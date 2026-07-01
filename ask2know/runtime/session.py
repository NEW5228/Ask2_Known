from pathlib import Path

from ask2know.data.dataset_loader import DatasetLoader
from ask2know.experience.pairwise import PairwiseExperienceManager
from ask2know.features.feature_config import (
    expand_feature_adjustments,
    initial_feature_weights,
    parse_feature_config,
    resolve_deep_feature_config,
    summarize_group_weights,
)
from ask2know.inference.prototype_model import PrototypeModel
from ask2know.inference.uncertainty import (
    is_globally_uncertain,
    saturated_feature_ratio,
    score_spread,
    top_gap,
)
from ask2know.learning.feedback_updater import apply_answer_to_weights, update_question_reward
from ask2know.learning.weights import AdaptiveWeights
from ask2know.questions.question_generator import generate_natural_question
from ask2know.questions.ask_resolver import (
    DEFAULT_ASK_CANDIDATE_TOP_K,
    DEFAULT_ASK_MAX_OPTIONS,
    DEFAULT_ASK_MAX_QUESTIONS,
    apply_dynamic_answer_to_predictions,
    apply_taxonomy_answer_to_predictions,
    build_runtime_dynamic_question,
    build_runtime_taxonomy_question,
)
from ask2know.questions.question_selector import QuestionSelector
from ask2know.sample_pool.manager import SamplePoolManager, _safe_name
from ask2know.utils.io_utils import ensure_dir, load_yaml, save_json
from run_demo import (
    VERSION,
    handle_sample_decision,
    make_class_understanding_summary,
    make_experience_report,
    make_experience_summary,
    render_class_understanding_markdown,
)


def _class_names(objects):
    return [item.get('name') for item in objects if item.get('name')]


def _pretty_weights(weights):
    return {key: round(float(value), 3) for key, value in weights.items()}


def _result_row(row):
    sources = {}
    for key in (
        'prototype_score',
        'subprototype_score',
        'knn_score',
        'text_semantic_score',
        'hierarchy_score',
        'taxonomy_score',
        'field_clip_score',
        'field_shape_score',
        'local_leaf_score',
        'pairwise_score',
        'crop_rerank_score',
        'pair_confusion_score',
        'concept_score',
    ):
        if row.get(key) is not None:
            sources[key] = round(float(row[key]), 4)
    nearest = []
    for item in row.get('nearest_samples') or []:
        nearest.append({
            'score': round(float(item.get('score', 0.0)), 4),
            'path': item.get('path'),
        })
    return {
        'label': row.get('label'),
        'score': round(float(row.get('score', 0.0)), 4),
        'taxonomy_path': list(row.get('taxonomy_path') or []),
        'detail': {k: round(float(v), 4) for k, v in (row.get('group_detail') or row.get('detail') or {}).items()},
        'system_detail': {k: round(float(v), 4) for k, v in (row.get('system_detail') or {}).items()},
        'sources': sources,
        'nearest_samples': nearest[:3],
    }


def _question_options(question):
    return [
        {
            'key': key,
            'text': text.format(a='{a}', b='{b}'),
        }
        for key, text, _ in question.get('options', [])
    ]


class LearningSession:
    """Stateful active-learning session shared by GUI entrypoints."""

    def __init__(self, config_path):
        self.config_path = Path(config_path).expanduser().resolve()
        self.cfg = None
        self.loader = None
        self.pool = None
        self.pairwise = None
        self.model = None
        self.q_selector = None
        self.aw = None
        self.objects = []
        self.feature_spec = {}
        self.train_samples = []
        self.unlabeled_samples = []
        self.unknown_samples = self.unlabeled_samples
        self.logs = []
        self.index = -1
        self.current_sample = None
        self.current_results = None
        self.current_state = None
        self.pending_question = None
        self.pending_generated = None
        self.pending_question_feedback = None
        self.question_turns = 0
        self.finished = False

    def initialize(self):
        if not self.config_path.exists():
            raise FileNotFoundError(f'配置文件不存在: {self.config_path}')

        self.cfg = load_yaml(self.config_path)
        dataset_dir = self.cfg['paths']['dataset_dir']
        output_dir = Path(self.cfg['paths']['output_dir'])
        project_root = self.cfg.get('paths', {}).get('project_root')
        ensure_dir(output_dir)
        ensure_dir(output_dir / 'logs')

        self.loader = DatasetLoader(dataset_dir)
        self.objects = self.loader.load_objects()
        concepts = self.loader.load_concepts()
        deep_feature_config = resolve_deep_feature_config(self.cfg)
        self.feature_spec = parse_feature_config(self.cfg, classes=_class_names(self.objects) or self.cfg.get('classes', []))

        self.pool = SamplePoolManager(project_root=project_root, output_dir=output_dir, dataset_dir=dataset_dir, version=VERSION)
        self.pairwise = PairwiseExperienceManager(metadata_dir=self.pool.metadata_dir, version=VERSION)
        self.pool.ensure_for_classes(_class_names(self.objects))
        self.pool.update_project_meta(
            project_name=self.cfg.get('task', {}).get('name', 'unknown'),
            classes=_class_names(self.objects),
        )

        if self.cfg.get('train_import', {}).get('auto_rename', True):
            self.pool.normalize_train_images(_class_names(self.objects))
        unlabeled_import_cfg = self.cfg.get('unlabeled_import', self.cfg.get('unknown_import', {'auto_rename': True}))
        if unlabeled_import_cfg.get('auto_rename', True):
            self.pool.normalize_unlabeled()

        self.train_samples = self.loader.load_train_samples()
        self.unlabeled_samples = self.loader.load_unlabeled_samples()
        self.unknown_samples = self.unlabeled_samples
        if not self.objects:
            raise RuntimeError('没有找到类别。请先新建项目或添加类别。')
        if not self.train_samples:
            raise RuntimeError('没有找到训练样本。请先为每个类别导入少量已知图片。')
        if not self.unlabeled_samples:
            raise RuntimeError('没有找到待学习 unlabeled 图片。请先在“添加数据集”中导入未标注图片。')

        initial_weights = initial_feature_weights(self.cfg, self.feature_spec)
        self.aw = AdaptiveWeights(
            initial_weights,
            self.cfg['learning'].get('update_step', 0.07),
            self.cfg['learning'].get('min_weight', 0.05),
            self.cfg['learning'].get('max_weight', 0.70),
        )
        self.aw.apply_concepts(self._expand_concept_feature_hints(concepts))
        self.model = PrototypeModel(
            self.feature_spec['scoring_features'],
            augmentation_config=self.cfg.get('augmentation', {}),
            concept_config=self.cfg.get('concepts', {'enable': True, 'score_weight': 0.25}),
            system_feature_names=self.feature_spec['system_features'],
            feature_groups=self.feature_spec['group_features'],
            similarity_config=self.cfg.get('similarity', {}),
            deep_feature_config=deep_feature_config,
            deep_cache_dir=output_dir / '.cache' / 'deep_features',
        ).fit(self.train_samples)
        self.q_selector = QuestionSelector(
            pairwise_manager=self.pairwise,
            enabled_features=self.feature_spec['display_features'],
        )
        return self.project_summary()

    def _expand_concept_feature_hints(self, concepts):
        expanded = []
        for item in concepts or []:
            copied = dict(item)
            copied['important_features'] = expand_feature_adjustments(item.get('important_features', []), self.feature_spec)
            copied['weak_features'] = expand_feature_adjustments(item.get('weak_features', []), self.feature_spec)
            expanded.append(copied)
        return expanded

    def project_summary(self):
        if self.cfg is None:
            return {}
        return {
            'config_path': str(self.config_path),
            'project_root': self.cfg.get('paths', {}).get('project_root'),
            'dataset_dir': self.cfg.get('paths', {}).get('dataset_dir'),
            'output_dir': self.cfg.get('paths', {}).get('output_dir'),
            'task_name': self.cfg.get('task', {}).get('name', 'unknown'),
            'classes': _class_names(self.objects),
            'display_features': list(self.feature_spec.get('display_features', [])),
            'train_count': len(self.train_samples),
            'unlabeled_count': len(self.unlabeled_samples),
            'unknown_count': len(self.unlabeled_samples),
        }

    def advance(self):
        if self.finished:
            return {'mode': 'done', 'message': '本轮学习已经结束。'}
        self.index += 1
        if self.index >= len(self.unlabeled_samples):
            return self.finish()

        self.current_sample = self.unlabeled_samples[self.index]
        self.pending_question = None
        self.pending_generated = None
        self.pending_question_feedback = None
        self.question_turns = 0
        return self._predict_current()

    def _build_taxonomy_question(self, results):
        question_cfg = self.cfg.get('question', {}) if self.cfg else {}
        return build_runtime_taxonomy_question(
            results,
            max_options=question_cfg.get('max_taxonomy_options', DEFAULT_ASK_MAX_OPTIONS),
            candidate_top_k=question_cfg.get('ask_candidate_top_k', DEFAULT_ASK_CANDIDATE_TOP_K),
        )

    def _build_dynamic_question(self, results):
        question_cfg = self.cfg.get('question', {}) if self.cfg else {}
        return build_runtime_dynamic_question(
            results,
            max_options=question_cfg.get('max_dynamic_options', question_cfg.get('max_taxonomy_options', DEFAULT_ASK_MAX_OPTIONS)),
            candidate_top_k=question_cfg.get('ask_candidate_top_k', DEFAULT_ASK_CANDIDATE_TOP_K),
        )

    def _predict_current(self):
        sample_path = self.current_sample['path']
        self.current_results = self.model.predict(sample_path, self.aw.export())
        if len(self.current_results) < 2:
            self.current_state = self._state('insufficient_classes', self.current_results)
            return self.current_state

        gap = top_gap(self.current_results)
        global_uncertain, reason = is_globally_uncertain(self.current_results, self.cfg)
        ask_threshold = self.cfg.get('confidence', {}).get('ask_user_threshold', 0.12)
        if global_uncertain:
            mode = 'global_uncertain'
        elif gap > ask_threshold:
            mode = 'confirm'
        else:
            mode = 'ask'
            weights = summarize_group_weights(self.aw.export(), self.feature_spec)
            taxonomy_question, taxonomy_generated = self._build_taxonomy_question(self.current_results)
            if self.cfg.get('question', {}).get('enable_taxonomy_ask', True) and taxonomy_question:
                self.pending_question = taxonomy_question
                self.pending_generated = taxonomy_generated
            elif self.cfg.get('question', {}).get('enable_dynamic_ask', True):
                dynamic_question, dynamic_generated = self._build_dynamic_question(self.current_results)
                if dynamic_question:
                    self.pending_question = dynamic_question
                    self.pending_generated = dynamic_generated
                else:
                    self.pending_question = self.q_selector.select(self.current_results[0], self.current_results[1], weights=weights)
                    self.pending_generated = generate_natural_question(
                        self.current_results[0],
                        self.current_results[1],
                        self.pending_question,
                        weights,
                        sample_path,
                        pairwise_manager=self.pairwise,
                    )
            else:
                self.pending_question = self.q_selector.select(self.current_results[0], self.current_results[1], weights=weights)
                self.pending_generated = generate_natural_question(
                    self.current_results[0],
                    self.current_results[1],
                    self.pending_question,
                    weights,
                    sample_path,
                    pairwise_manager=self.pairwise,
                )
        self.current_state = self._state(mode, self.current_results, reason=reason)
        return self.current_state

    def _state(self, mode, results, reason=''):
        question = None
        if self.pending_question and self.pending_generated:
            a = results[0].get('label') if results else ''
            b = results[1].get('label') if len(results) > 1 else ''
            question = {
                'id': self.pending_question.get('id'),
                'kind': self.pending_question.get('kind', 'feature_weight_update'),
                'question': self.pending_generated.get('question', ''),
                'evidence': self.pending_generated.get('evidence', ''),
                'concept_evidence': self.pending_generated.get('concept_evidence', ''),
                'selected_feature': self.pending_generated.get('selected_feature', ''),
                'options': [
                    {
                        'key': key,
                        'text': text.format(a=a, b=b),
                    }
                    for key, text, _ in self.pending_question.get('options', [])
                ],
            }
        return {
            'mode': mode,
            'sample_path': self.current_sample['path'] if self.current_sample else None,
            'sample_index': self.index + 1,
            'sample_total': len(self.unlabeled_samples),
            'classes': _class_names(self.objects),
            'results': [_result_row(row) for row in results[:5]],
            'weights': _pretty_weights(summarize_group_weights(self.aw.export(), self.feature_spec)),
            'gap': round(float(top_gap(results)), 4) if results else 0.0,
            'spread': round(float(score_spread(results)), 4) if results else 0.0,
            'saturated_ratio': round(float(saturated_feature_ratio(results)), 4) if results else 0.0,
            'uncertainty_reason': reason,
            'question': question,
            'question_feedback': self.pending_question_feedback,
            'summary': self.project_summary(),
        }

    def answer_question(self, selected_key):
        if not self.pending_question or not self.current_results:
            raise RuntimeError('当前没有待回答的问题。')
        selected_key = (selected_key or '').strip().upper()
        if not selected_key:
            selected_key = self.pending_question['options'][-1][0]

        old_results = self.current_results
        answered_question = self.pending_question
        selected = None
        for key, text, action in answered_question.get('options', []):
            if str(key).upper() == selected_key:
                selected = (key, text, action)
                break
        if selected is None:
            raise ValueError(f'无法识别的问题选项: {selected_key}')

        key, answer_text, action = selected
        before = self.aw.export()
        after = before
        if answered_question.get('kind') == 'taxonomy_resolution' or action.get('kind') == 'taxonomy_resolution':
            _reranked, matched = apply_taxonomy_answer_to_predictions(
                self.current_results,
                action.get('path_prefix') or [],
            )
            if not matched:
                raise ValueError('当前候选中没有匹配这个分层选项的类别。')
            self.current_results = matched
            self.question_turns += 1
            feedback_kind = 'taxonomy_resolution'
        elif answered_question.get('kind') == 'dynamic_disambiguation' or action.get('kind') == 'dynamic_disambiguation':
            reranked, matched = apply_dynamic_answer_to_predictions(
                self.current_results,
                action.get('labels') or [],
                score_bonus=action.get('score_bonus', 1.0),
            )
            if not matched:
                raise ValueError('当前候选中没有匹配这个动态选项的类别。')
            self.current_results = reranked
            self.question_turns += 1
            feedback_kind = 'dynamic_disambiguation'
        else:
            answer_text, before, after = apply_answer_to_weights(
                self.aw,
                answered_question,
                selected_key,
                feature_expander=lambda keys: expand_feature_adjustments(keys, self.feature_spec),
            )
            if answer_text is None:
                raise ValueError(f'无法识别的问题选项: {selected_key}')
            self.current_results = self.model.predict(self.current_sample['path'], self.aw.export())
            feedback_kind = 'feature_weight_update'

        answer_text = answer_text.format(a=old_results[0]['label'], b=old_results[1]['label'])
        self.pending_question_feedback = {
            'question_id': answered_question.get('id'),
            'generated_question': self.pending_generated,
            'answer': selected_key,
            'answer_text': answer_text,
            'kind': feedback_kind,
            'weights_before': _pretty_weights(summarize_group_weights(before, self.feature_spec)),
            'weights_after': _pretty_weights(summarize_group_weights(after, self.feature_spec)),
            'results_before': [_result_row(row) for row in old_results[:5]],
            'results_after': [_result_row(row) for row in self.current_results[:5]],
        }
        if feedback_kind == 'taxonomy_resolution':
            max_questions = int(self.cfg.get('question', {}).get('max_questions_per_sample', DEFAULT_ASK_MAX_QUESTIONS))
            if self.question_turns < max(1, max_questions):
                next_question, next_generated = self._build_taxonomy_question(self.current_results)
                if next_question:
                    self.pending_question = next_question
                    self.pending_generated = next_generated
                    self.current_state = self._state('ask', self.current_results)
                    return self.current_state
        self.pending_question = None
        self.pending_generated = None
        self.current_state = self._state('confirm_after_question', self.current_results)
        return self.current_state

    def decide_current(self, decision, label=None, correction_reason_ids=None, correction_note=''):
        if not self.current_sample or not self.current_results:
            raise RuntimeError('当前没有待处理样本。')

        predicted_label = self.current_results[0]['label'] if self.current_results else None
        mode = self.current_state.get('mode') if self.current_state else ''
        internal_decision, internal_label = self._normalize_decision(decision, label, predicted_label)
        pool_info = handle_sample_decision(
            internal_decision,
            internal_label,
            self.current_sample['path'],
            self.model,
            self.pool,
            self.objects,
            self.cfg['paths']['dataset_dir'],
            self.cfg.get('sample_pool', {}).get('enable', True),
            self.cfg.get('sample_pool', {}).get('move_unlabeled_after_decision', True),
        )

        correction_info = None
        corrected_label = pool_info.get('label') or internal_label
        if mode != 'global_uncertain' and internal_decision in ('class', 'new') and corrected_label and corrected_label != predicted_label:
            correction_info = self.record_correction(
                predicted_label,
                corrected_label,
                correction_reason_ids or ['other'],
                free_note=correction_note,
            )

        log_item = {
            'sample': self.current_sample['path'],
            'before': self.pending_question_feedback.get('results_before') if self.pending_question_feedback else self.current_results,
            'global_uncertain': mode == 'global_uncertain',
            'uncertainty_reason': self.current_state.get('uncertainty_reason') if self.current_state else '',
            'asked': bool(self.pending_question_feedback),
            'pool': pool_info,
            'correction_reason': correction_info,
            'weights_after': self.aw.export(),
        }
        if self.pending_question_feedback:
            old_results = self.pending_question_feedback.get('results_before') or []
            old_gap = 0.0
            if len(old_results) >= 2:
                old_gap = float(old_results[0]['score']) - float(old_results[1]['score'])
            new_gap = top_gap(self.current_results)
            helpful = pool_info.get('decision') == 'confirmed' and new_gap >= old_gap and pool_info.get('label') == predicted_label
            question_id = self.pending_question_feedback.get('question_id') or (self.pending_question or {}).get('id') or ''
            if self.cfg.get('question', {}).get('enable_question_reward', True):
                update_question_reward(self.q_selector.question_weights, question_id, helpful)
            self.pairwise.record_question_result(
                self.pending_question_feedback['results_before'][0]['label'],
                self.pending_question_feedback['results_before'][1]['label'],
                question_id,
                helpful,
            )
            log_item.update({
                'asked': True,
                'question': question_id,
                'generated_question': self.pending_question_feedback.get('generated_question'),
                'answer': self.pending_question_feedback['answer'],
                'answer_text': self.pending_question_feedback['answer_text'],
                'weights_before': self.pending_question_feedback['weights_before'],
                'after': self.current_results,
                'helpful': helpful,
                'question_state': self.q_selector.export(),
            })
        self.logs.append(log_item)
        return self.advance()

    def _normalize_decision(self, decision, label, predicted_label):
        value = str(decision or '').strip().lower()
        if value in ('correct', 'confirmed'):
            return 'confirmed', predicted_label
        if value in ('class', 'wrong'):
            if not label:
                raise ValueError('请选择正确类别。')
            return 'class', label
        if value == 'new':
            if not label:
                raise ValueError('请输入新类别名称。')
            return 'new', label
        if value in ('candidate', 'reject', 'rejected', 'skip'):
            mapped = 'reject' if value == 'rejected' else value
            return mapped, label or predicted_label
        return 'skip', None

    def record_correction(self, predicted_label, true_label, reason_ids, free_note=''):
        options = self.pairwise.correction_options(predicted_label, true_label)
        option_by_reason = {reason_id: (reason_id, text, action) for _, reason_id, text, action in options}
        selected = []
        inc = []
        dec = []
        for reason_id in reason_ids or ['other']:
            reason_id, reason_text, action = option_by_reason.get(reason_id, option_by_reason.get('other'))
            selected.append((reason_id, reason_text))
            inc.extend(action.get('increase', []))
            dec.extend(action.get('decrease', []))
        before, after = self.aw.update(
            expand_feature_adjustments(inc, self.feature_spec),
            expand_feature_adjustments(dec, self.feature_spec),
        )
        pair = self.pairwise.record_corrections(predicted_label, true_label, selected, free_note=free_note)
        return {
            'predicted': predicted_label,
            'true_label': true_label,
            'reasons': [{'reason_id': reason_id, 'reason_text': text} for reason_id, text in selected],
            'user_discriminative_note': free_note,
            'weights_before': before,
            'weights_after': after,
            'pairwise': pair,
        }

    def correction_options(self, predicted_label=None, true_label=None):
        labels = _class_names(self.objects)
        predicted_label = predicted_label or (self.current_results[0]['label'] if self.current_results else '')
        true_label = true_label or next((label for label in labels if label != predicted_label), predicted_label)
        return [
            {'key': key, 'reason_id': reason_id, 'text': text}
            for key, reason_id, text, _ in self.pairwise.correction_options(predicted_label, true_label)
        ]

    def finish(self):
        if self.finished:
            return {'mode': 'done', 'message': '本轮学习已经结束。', 'summary': self.project_summary()}
        self.finished = True
        output_dir = Path(self.cfg['paths']['output_dir'])
        final_weights = self.aw.export()
        question_state = self.q_selector.export()
        pairwise_state = self.pairwise.export()
        report = make_experience_report(
            self.cfg,
            self.logs,
            final_weights,
            question_state,
            self.objects,
            self.feature_spec,
            pairwise_state=pairwise_state,
        )
        experience_summary = make_experience_summary(pairwise_state, self.objects)
        class_summary = make_class_understanding_summary(self.model, self.objects, pairwise_state=pairwise_state)
        class_markdown = render_class_understanding_markdown(class_summary)

        save_json(output_dir / 'feature_weights.json', summarize_group_weights(final_weights, self.feature_spec))
        save_json(output_dir / 'internal_feature_weights.json', final_weights)
        save_json(output_dir / 'question_weights.json', question_state)
        save_json(output_dir / 'prototype_model.json', self.model.export())
        save_json(output_dir / 'logs' / 'demo_log.json', self.logs)
        save_json(output_dir / 'experience_report.json', report)
        save_json(output_dir / 'objects_runtime.json', {'objects': self.objects})
        save_json(output_dir / 'pairwise_experience_runtime.json', pairwise_state)
        save_json(output_dir / 'experience_summary.json', experience_summary)
        save_json(self.pool.metadata_dir / 'experience_summary.json', experience_summary)
        save_json(output_dir / 'class_understanding_summary.json', class_summary)
        (output_dir / 'class_understanding_summary.md').write_text(class_markdown, encoding='utf-8')
        save_json(self.pool.metadata_dir / 'class_understanding_summary.json', class_summary)
        return {
            'mode': 'done',
            'message': '本轮学习结束，报告已保存。',
            'output_dir': str(output_dir),
            'logs_count': len(self.logs),
            'summary': self.project_summary(),
        }


def add_class_to_project(project_root, class_name):
    import yaml

    project = Path(project_root).expanduser().resolve()
    storage_name = _safe_name(class_name)
    dataset_dir = project / 'datasets'
    train_dir = dataset_dir / 'train'
    config_path = project / 'configs' / 'task_config.yaml'
    objects_path = dataset_dir / 'objects.json'
    metadata_dir = project / 'metadata'
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (train_dir / storage_name).mkdir(parents=True, exist_ok=True)

    data = {'objects': []}
    if objects_path.exists():
        import json
        data = json.loads(objects_path.read_text(encoding='utf-8'))
    objects = data.setdefault('objects', [])
    if storage_name not in [item.get('name') for item in objects]:
        objects.append({
            'object_id': f'C{len(objects) + 1:03d}',
            'name': storage_name,
            'display_name': class_name,
            'description': f'added by desktop GUI v{VERSION}',
        })
        save_json(objects_path, data)

    if config_path.exists():
        cfg = load_yaml(config_path) or {}
        classes = cfg.setdefault('classes', [])
        if storage_name not in classes:
            classes.append(storage_name)
        cfg.setdefault('paths', {})['project_root'] = str(project).replace('\\', '/')
        cfg['paths']['dataset_dir'] = str(dataset_dir).replace('\\', '/')
        cfg['paths']['output_dir'] = str((project / 'outputs')).replace('\\', '/')
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return storage_name
