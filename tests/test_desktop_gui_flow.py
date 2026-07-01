from pathlib import Path

from PIL import Image

from app_desktop import apply_algorithm_profile_to_config, parse_class_names, unique_copy
from ask2know.data.dataset_loader import DatasetLoader
from ask2know.runtime.project import create_task_project
from ask2know.runtime.session import LearningSession, add_class_to_project


def _image(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (12, 12), color).save(path)
    return path


def test_parse_class_names_accepts_list_like_input():
    assert parse_class_names('apple banana,pear\napple;grape') == ['apple', 'banana', 'pear', 'grape']


def test_algorithm_profile_updates_task_config(tmp_path):
    created = create_task_project(
        name='profile_smoke',
        output=tmp_path,
        classes=['apple', 'banana'],
        feature_preset='fruit',
        features=['color', 'shape'],
    )
    config = Path(created['config_path'])

    legacy = apply_algorithm_profile_to_config(config, 'classic_similarity')
    assert legacy['algorithm_profile']['id'] == 'classic_similarity'
    assert legacy['question']['max_questions_per_sample'] == 1
    assert legacy['question']['enable_taxonomy_ask'] is False
    assert legacy['similarity']['recognition_mode'] == 'flat'

    latest = apply_algorithm_profile_to_config(config, 'interactive_hierarchy')
    assert latest['algorithm_profile']['id'] == 'interactive_hierarchy'
    assert latest['question']['max_questions_per_sample'] == 2
    assert latest['question']['ask_candidate_top_k'] == 10
    assert latest['question']['max_taxonomy_options'] == 8

def test_create_task_project_texture_preset_writes_release_defaults(tmp_path):
    created = create_task_project(
        name='texture_smoke',
        output=tmp_path,
        classes=['banded', 'woven'],
        feature_preset='texture',
    )

    config_text = Path(created['config_path']).read_text(encoding='utf-8')
    task_readme = (Path(created['project_root']) / 'README_task.md').read_text(encoding='utf-8')

    assert created['feature_preset'] == 'texture'
    assert created['features'] == ['color', 'shape', 'texture', 'surface', 'part']
    assert 'preset: texture' in config_text
    assert 'enable_dynamic_ask: true' in config_text
    assert 'max_dynamic_options: 8' in config_text
    assert '用于命令行交互学习的未知样本放入' in task_readme
    assert "{dataset_dir / 'unlabeled'}" not in task_readme
    assert '<class_name>' in task_readme

def test_desktop_project_import_and_session_smoke(tmp_path, monkeypatch):
    created = create_task_project(
        name='gui_smoke',
        output=tmp_path,
        classes=['apple', 'banana'],
        feature_preset='fruit',
        features=['color', 'shape'],
    )
    project = Path(created['project_root'])
    dataset = Path(created['dataset_dir'])
    config = Path(created['config_path'])

    added = [add_class_to_project(project, name) for name in ['pear', 'grape']]
    assert added == ['pear', 'grape']

    source = tmp_path / 'source_images'
    unique_copy(_image(source / 'apple_sample.jpg', (220, 30, 30)), dataset / 'train' / 'apple')
    unique_copy(_image(source / 'banana_sample.jpg', (240, 220, 30)), dataset / 'train' / 'banana')
    unique_copy(_image(source / 'unlabeled_sample.jpg', (230, 40, 40)), dataset / 'unlabeled')
    unique_copy(_image(source / 'eval_sample.jpg', (220, 30, 30)), dataset / 'unlabeled' / 'apple')

    loader = DatasetLoader(dataset)
    assert {item['name'] for item in loader.load_objects()} == {'apple', 'banana', 'pear', 'grape'}
    assert len(loader.load_train_samples()) == 2
    assert len(loader.load_unlabeled_samples()) == 1
    assert len(loader.load_eval_samples()) == 1

    class FakeModel:
        def __init__(self, *args, **kwargs):
            self.added = []

        def fit(self, samples):
            self.samples = samples
            return self

        def predict(self, image_path, weights):
            return [
                {
                    'label': 'apple',
                    'score': 0.91,
                    'prototype_score': 0.91,
                    'detail': {'color': 0.9},
                    'group_detail': {'color': 0.9},
                    'system_detail': {},
                    'nearest_samples': [{'score': 0.88, 'path': 'apple_sample.jpg'}],
                },
                {
                    'label': 'banana',
                    'score': 0.74,
                    'prototype_score': 0.74,
                    'detail': {'color': 0.7},
                    'group_detail': {'color': 0.7},
                    'system_detail': {},
                    'nearest_samples': [],
                },
            ]

        def add_confirmed_sample(self, label, image_path):
            self.added.append((label, image_path))

        def export(self):
            return {'fake': True}

    monkeypatch.setattr('ask2know.runtime.session.PrototypeModel', FakeModel)
    monkeypatch.setattr('ask2know.runtime.session.make_class_understanding_summary', lambda *args, **kwargs: {'classes': {}})
    monkeypatch.setattr('ask2know.runtime.session.render_class_understanding_markdown', lambda summary: '# summary\n')

    session = LearningSession(config)
    summary = session.initialize()
    assert summary['train_count'] == 2
    assert summary['unlabeled_count'] == 1

    state = session.advance()
    assert state['mode'] == 'confirm'
    assert state['results'][0]['label'] == 'apple'

    done = session.decide_current('correct')
    assert done['mode'] == 'done'
    assert (Path(done['output_dir']) / 'experience_report.json').exists()
