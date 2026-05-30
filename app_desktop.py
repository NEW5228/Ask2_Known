import shutil
import threading
import traceback
from pathlib import Path
from tkinter import (
    BooleanVar,
    END,
    LEFT,
    RIGHT,
    BOTH,
    X,
    Y,
    Tk,
    Toplevel,
    Listbox,
    filedialog,
    messagebox,
    simpledialog,
)
from tkinter import ttk

from PIL import Image, ImageTk

from ask2know.data.dataset_loader import IMAGE_EXTS, DatasetLoader
from ask2know.features.feature_config import PRESET_DEFAULT_GROUPS, USER_FEATURE_GROUPS
from ask2know.runtime.project import create_task_project
from ask2know.runtime.session import LearningSession, add_class_to_project
from ask2know.sample_pool.manager import _safe_name
from ask2know.utils.io_utils import load_yaml


def parse_class_names(raw):
    names = []
    seen = set()
    for item in str(raw or '').replace(',', ' ').replace(';', ' ').replace('\n', ' ').split():
        name = item.strip()
        if name and name not in seen:
            names.append(name)
            seen.add(name)
    return names


def unique_copy(src, dst_dir):
    src = Path(src)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    base = src.stem
    ext = src.suffix.lower()
    dst = dst_dir / f'{base}{ext}'
    idx = 1
    while dst.exists():
        dst = dst_dir / f'{base}_{idx}{ext}'
        idx += 1
    shutil.copy2(str(src), str(dst))
    return dst


class CreateProjectDialog:
    def __init__(self, parent):
        self.result = None
        self.win = Toplevel(parent)
        self.win.title('新建 Ask2Know 项目')
        self.win.transient(parent)
        self.win.grab_set()

        body = ttk.Frame(self.win, padding=12)
        body.pack(fill=BOTH, expand=True)

        self.name_var = ttk.Entry(body, width=42)
        self.preset_var = ttk.Combobox(body, values=['auto', 'general', 'fruit', 'pet', 'car', 'traffic_sign'], state='readonly')
        self.preset_var.set('auto')

        self._row(body, 0, '项目名', self.name_var)
        output_row = ttk.Frame(body)
        self.output_var = ttk.Entry(output_row, width=42)
        self.output_var.insert(0, str(Path.cwd()))
        self.output_var.pack(side=LEFT, fill=X, expand=True)
        ttk.Button(output_row, text='选择', command=self.choose_output).pack(side=RIGHT, padx=(6, 0))
        self._row(body, 1, '保存位置', output_row)

        class_box = ttk.Frame(body)
        class_input_row = ttk.Frame(class_box)
        self.class_input = ttk.Entry(class_input_row, width=32)
        self.class_input.pack(side=LEFT, fill=X, expand=True)
        ttk.Button(class_input_row, text='添加', command=self.add_class_name).pack(side=LEFT, padx=(6, 0))
        ttk.Button(class_input_row, text='删除选中', command=self.remove_selected_class).pack(side=LEFT, padx=(6, 0))
        class_input_row.pack(fill=X)
        ttk.Label(class_box, text='每次输入一个类别；也可以粘贴多个，用空格、逗号或换行分隔。').pack(anchor='w', pady=(3, 4))
        self.class_list = Listbox(class_box, height=6, exportselection=False)
        self.class_list.pack(fill=X)
        self.class_input.bind('<Return>', lambda _event: self.add_class_name())
        self._row(body, 2, '类别列表', class_box)
        self._row(body, 3, '预设', self.preset_var)

        self.feature_vars = {}
        features_frame = ttk.LabelFrame(body, text='用户可见特征', padding=8)
        features_frame.grid(row=4, column=0, columnspan=2, sticky='ew', pady=(8, 0))
        defaults = set(PRESET_DEFAULT_GROUPS['general'])
        for idx, name in enumerate(USER_FEATURE_GROUPS):
            var = BooleanVar(value=name in defaults)
            self.feature_vars[name] = var
            ttk.Checkbutton(features_frame, text=name, variable=var).grid(row=idx // 4, column=idx % 4, sticky='w', padx=8, pady=3)

        buttons = ttk.Frame(body)
        buttons.grid(row=5, column=0, columnspan=2, sticky='e', pady=(12, 0))
        ttk.Button(buttons, text='取消', command=self.win.destroy).pack(side=RIGHT)
        ttk.Button(buttons, text='创建', command=self.create).pack(side=RIGHT, padx=(0, 8))

        body.columnconfigure(1, weight=1)
        self.win.wait_window()

    def _row(self, parent, row, label, widget):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', pady=4, padx=(0, 8))
        widget.grid(row=row, column=1, sticky='ew', pady=4)

    def choose_output(self):
        path = filedialog.askdirectory(parent=self.win, title='选择项目保存位置')
        if path:
            self.output_var.delete(0, END)
            self.output_var.insert(0, path)

    def add_class_name(self):
        raw = self.class_input.get().strip()
        if not raw:
            return
        existing = set(self.class_list.get(0, END))
        for name in parse_class_names(raw):
            if name not in existing:
                self.class_list.insert(END, name)
                existing.add(name)
        self.class_input.delete(0, END)

    def remove_selected_class(self):
        for idx in reversed(self.class_list.curselection()):
            self.class_list.delete(idx)

    def create(self):
        self.add_class_name()
        name = self.name_var.get().strip()
        output = self.output_var.get().strip() or '.'
        classes = list(self.class_list.get(0, END))
        if len(classes) < 2:
            messagebox.showwarning('类别不足', 'GUI 新建项目至少需要 2 个类别，后续也可以继续新增类别。', parent=self.win)
            return
        features = [name for name, var in self.feature_vars.items() if var.get()]
        try:
            self.result = create_task_project(
                name=name,
                output=output,
                classes=classes,
                feature_preset=self.preset_var.get(),
                features=features,
            )
        except Exception as exc:
            messagebox.showerror('创建失败', str(exc), parent=self.win)
            return
        self.win.destroy()


class Ask2KnowDesktopApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Ask2Know 本地交互窗口')
        self.root.geometry('1180x760')
        self.session = None
        self.current_state = None
        self.photo = None
        self.config_path = None

        self.status_var = ttk.Label(root, text='请选择或新建项目。', anchor='w')
        self.status_var.pack(fill=X, side='bottom')

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=BOTH, expand=True)

        self.project_tab = ttk.Frame(self.notebook, padding=10)
        self.learn_tab = ttk.Frame(self.notebook, padding=10)
        self.report_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.project_tab, text='项目')
        self.notebook.add(self.learn_tab, text='学习')
        self.notebook.add(self.report_tab, text='报告')

        self._build_project_tab()
        self._build_learning_tab()
        self._build_report_tab()

    def _build_project_tab(self):
        top = ttk.Frame(self.project_tab)
        top.pack(fill=X)
        ttk.Button(top, text='新建项目', command=self.create_project).pack(side=LEFT)
        ttk.Button(top, text='打开配置', command=self.open_config).pack(side=LEFT, padx=6)
        ttk.Button(top, text='加载会话', command=self.initialize_session).pack(side=LEFT, padx=6)
        ttk.Button(top, text='导入训练图片', command=self.import_train_images).pack(side=LEFT, padx=(18, 6))
        ttk.Button(top, text='导入 unknown', command=self.import_unknown_images).pack(side=LEFT, padx=6)
        ttk.Button(top, text='导入评估图片', command=self.import_eval_images).pack(side=LEFT, padx=6)
        ttk.Button(top, text='新增类别', command=self.add_class).pack(side=LEFT, padx=(18, 0))

        self.config_var = ttk.Entry(self.project_tab)
        self.config_var.pack(fill=X, pady=(10, 8))

        self.project_info = ttk.Treeview(self.project_tab, columns=('value',), show='tree headings', height=18)
        self.project_info.heading('#0', text='项目项')
        self.project_info.heading('value', text='值')
        self.project_info.column('#0', width=220)
        self.project_info.column('value', width=720)
        self.project_info.pack(fill=BOTH, expand=True)

    def _build_learning_tab(self):
        outer = ttk.Frame(self.learn_tab)
        outer.pack(fill=BOTH, expand=True)
        left = ttk.Frame(outer)
        right = ttk.Frame(outer)
        left.pack(side=LEFT, fill=BOTH, expand=False)
        right.pack(side=RIGHT, fill=BOTH, expand=True, padx=(12, 0))

        controls = ttk.Frame(left)
        controls.pack(fill=X, pady=(0, 8))
        ttk.Button(controls, text='开始学习', command=self.start_learning).pack(side=LEFT)
        ttk.Button(controls, text='结束并保存报告', command=self.finish_session).pack(side=LEFT, padx=6)

        self.image_label = ttk.Label(left, text='图片预览', anchor='center')
        self.image_label.pack(fill=BOTH, expand=True)
        self.sample_label = ttk.Label(left, text='', wraplength=360)
        self.sample_label.pack(fill=X, pady=(8, 0))

        summary_frame = ttk.LabelFrame(right, text='当前状态', padding=8)
        summary_frame.pack(fill=X)
        self.state_text = ttk.Label(summary_frame, text='尚未开始。', wraplength=720, justify='left')
        self.state_text.pack(fill=X)

        result_frame = ttk.LabelFrame(right, text='预测结果', padding=8)
        result_frame.pack(fill=BOTH, expand=True, pady=8)
        self.result_tree = ttk.Treeview(result_frame, columns=('score', 'detail', 'sources', 'nearest'), show='tree headings', height=8)
        self.result_tree.heading('#0', text='类别')
        self.result_tree.heading('score', text='分数')
        self.result_tree.heading('detail', text='特征')
        self.result_tree.heading('sources', text='来源')
        self.result_tree.heading('nearest', text='最近样本')
        self.result_tree.column('#0', width=110)
        self.result_tree.column('score', width=70)
        self.result_tree.column('detail', width=220)
        self.result_tree.column('sources', width=210)
        self.result_tree.column('nearest', width=260)
        self.result_tree.pack(fill=BOTH, expand=True)

        self.question_frame = ttk.LabelFrame(right, text='系统提问', padding=8)
        self.question_frame.pack(fill=X, pady=(0, 8))
        self.question_text = ttk.Label(self.question_frame, text='当前没有问题。', wraplength=730, justify='left')
        self.question_text.pack(fill=X)
        self.option_frame = ttk.Frame(self.question_frame)
        self.option_frame.pack(fill=X, pady=(8, 0))

        decision = ttk.LabelFrame(right, text='用户确认', padding=8)
        decision.pack(fill=X)
        row = ttk.Frame(decision)
        row.pack(fill=X)
        ttk.Label(row, text='正确类别').pack(side=LEFT)
        self.class_var = ttk.Combobox(row, values=[], state='readonly', width=20)
        self.class_var.pack(side=LEFT, padx=6)
        ttk.Label(row, text='新类别').pack(side=LEFT, padx=(12, 0))
        self.new_class_var = ttk.Entry(row, width=20)
        self.new_class_var.pack(side=LEFT, padx=6)

        reason_row = ttk.Frame(decision)
        reason_row.pack(fill=X, pady=(8, 0))
        self.reason_vars = {}
        for reason in ('color', 'shape', 'texture', 'surface', 'part', 'text', 'sign', 'background', 'other'):
            var = BooleanVar(value=False)
            self.reason_vars[reason] = var
            ttk.Checkbutton(reason_row, text=reason, variable=var).pack(side=LEFT, padx=(0, 8))
        self.note_var = ttk.Entry(decision)
        self.note_var.pack(fill=X, pady=(8, 0))
        self.note_var.insert(0, '可选：写一句你区分这两个类别的依据')

        buttons = ttk.Frame(decision)
        buttons.pack(fill=X, pady=(8, 0))
        ttk.Button(buttons, text='正确', command=lambda: self.decide('correct')).pack(side=LEFT)
        ttk.Button(buttons, text='改为所选类别', command=lambda: self.decide('class')).pack(side=LEFT, padx=6)
        ttk.Button(buttons, text='作为新类别', command=lambda: self.decide('new')).pack(side=LEFT, padx=6)
        ttk.Button(buttons, text='暂存 candidate', command=lambda: self.decide('candidate')).pack(side=LEFT, padx=6)
        ttk.Button(buttons, text='拒绝样本', command=lambda: self.decide('reject')).pack(side=LEFT, padx=6)
        ttk.Button(buttons, text='跳过', command=lambda: self.decide('skip')).pack(side=LEFT, padx=6)

    def _build_report_tab(self):
        self.report_text = ttk.Label(self.report_tab, text='报告会在结束学习后生成。', justify='left', wraplength=1000)
        self.report_text.pack(fill=X, anchor='nw')

    def set_status(self, text):
        self.status_var.configure(text=text)

    def run_worker(self, title, func, on_success):
        self.set_status(title)

        def work():
            try:
                result = func()
            except Exception as exc:
                tb = traceback.format_exc()
                self.root.after(0, lambda exc=exc, tb=tb: self.show_error(exc, tb))
                return
            self.root.after(0, lambda: on_success(result))

        threading.Thread(target=work, daemon=True).start()

    def show_error(self, exc, tb):
        self.set_status('操作失败。')
        messagebox.showerror('错误', f'{exc}\n\n{tb}')

    def create_project(self):
        dialog = CreateProjectDialog(self.root)
        if not dialog.result:
            return
        self.config_path = dialog.result['config_path']
        self.config_var.delete(0, END)
        self.config_var.insert(0, self.config_path)
        self.refresh_project_info()
        self.set_status('项目已创建，请导入训练图片和 unknown 图片。')

    def open_config(self):
        path = filedialog.askopenfilename(
            title='选择 task_config.yaml',
            filetypes=[('YAML', '*.yaml *.yml'), ('All files', '*.*')],
        )
        if not path:
            return
        self.config_path = path
        self.config_var.delete(0, END)
        self.config_var.insert(0, path)
        self.refresh_project_info()

    def config(self):
        path = self.config_var.get().strip()
        if not path:
            raise RuntimeError('请先选择 task_config.yaml。')
        return load_yaml(path), Path(path)

    def refresh_project_info(self):
        self.project_info.delete(*self.project_info.get_children())
        try:
            cfg, config_path = self.config()
            dataset_dir = Path(cfg['paths']['dataset_dir'])
            loader = DatasetLoader(dataset_dir)
            objects = loader.load_objects()
            train_samples = loader.load_train_samples()
            unknown_samples = loader.load_unknown_samples()
            eval_samples = loader.load_eval_samples()
        except Exception as exc:
            self.project_info.insert('', END, text='读取失败', values=(str(exc),))
            return
        items = {
            '配置文件': str(config_path),
            '项目目录': cfg.get('paths', {}).get('project_root', ''),
            '数据目录': str(dataset_dir),
            '输出目录': cfg.get('paths', {}).get('output_dir', ''),
            '类别': ', '.join([item.get('name', '') for item in objects]),
            '训练样本数': str(len(train_samples)),
            'unknown 样本数': str(len(unknown_samples)),
            '评估样本数': str(len(eval_samples)),
        }
        for key, value in items.items():
            self.project_info.insert('', END, text=key, values=(value,))
        class_root = self.project_info.insert('', END, text='类别明细', values=('每类训练样本数',), open=True)
        counts = {}
        for sample in train_samples:
            counts[sample.get('label')] = counts.get(sample.get('label'), 0) + 1
        for item in objects:
            name = item.get('name', '')
            display = item.get('display_name') or name
            self.project_info.insert(class_root, END, text=display, values=(f'{counts.get(name, 0)} 张',))

    def initialize_session(self):
        path = self.config_var.get().strip()
        if not path:
            messagebox.showwarning('缺少配置', '请先选择 task_config.yaml。')
            return

        def build():
            session = LearningSession(path)
            summary = session.initialize()
            return session, summary

        def done(result):
            self.session, summary = result
            self.current_state = None
            self.class_var.configure(values=summary.get('classes', []))
            if summary.get('classes'):
                self.class_var.set(summary['classes'][0])
            self.refresh_project_info()
            self.set_status('会话已加载，可以开始学习。')

        self.run_worker('正在加载模型和提取训练特征...', build, done)

    def initialize_and_advance(self):
        path = self.config_var.get().strip()
        if not path:
            messagebox.showwarning('缺少配置', '请先选择 task_config.yaml。')
            return

        def build_and_advance():
            session = LearningSession(path)
            session.initialize()
            state = session.advance()
            return session, state

        def done(result):
            self.session, state = result
            classes = state.get('classes') or []
            self.class_var.configure(values=classes)
            if classes:
                self.class_var.set(classes[0])
            self.render_state(state)

        self.run_worker('正在加载模型、提取训练特征并预测第一张 unknown 图片...', build_and_advance, done)

    def import_train_images(self):
        try:
            cfg, _ = self.config()
            dataset_dir = Path(cfg['paths']['dataset_dir'])
            loader = DatasetLoader(dataset_dir)
            classes = [item.get('name') for item in loader.load_objects()]
        except Exception as exc:
            messagebox.showerror('导入失败', str(exc))
            return
        if not classes:
            messagebox.showwarning('没有类别', '请先新建项目或新增类别。')
            return
        cls = simpledialog.askstring('训练类别', '请输入要导入到哪个类别：\n' + ', '.join(classes), initialvalue=classes[0])
        if not cls:
            return
        safe = _safe_name(cls)
        paths = self.choose_images()
        if not paths:
            return
        dst_dir = dataset_dir / 'train' / safe
        count = self.copy_many(paths, dst_dir)
        self.refresh_project_info()
        self.set_status(f'已导入 {count} 张训练图片到 {safe}。')

    def import_unknown_images(self):
        try:
            cfg, _ = self.config()
            dst_dir = Path(cfg['paths']['dataset_dir']) / 'unknown'
        except Exception as exc:
            messagebox.showerror('导入失败', str(exc))
            return
        paths = self.choose_images()
        if not paths:
            return
        count = self.copy_many(paths, dst_dir)
        self.refresh_project_info()
        self.set_status(f'已导入 {count} 张 unknown 图片。')

    def import_eval_images(self):
        try:
            cfg, _ = self.config()
            dataset_dir = Path(cfg['paths']['dataset_dir'])
            classes = [item.get('name') for item in DatasetLoader(dataset_dir).load_objects()]
        except Exception as exc:
            messagebox.showerror('导入失败', str(exc))
            return
        cls = simpledialog.askstring('评估类别', '请输入评估图片真实类别：\n' + ', '.join(classes), initialvalue=classes[0] if classes else '')
        if not cls:
            return
        paths = self.choose_images()
        if not paths:
            return
        count = self.copy_many(paths, dataset_dir / 'unlabeled' / _safe_name(cls))
        self.refresh_project_info()
        self.set_status(f'已导入 {count} 张评估图片到 {cls}。')

    def choose_images(self):
        return filedialog.askopenfilenames(
            title='选择图片',
            filetypes=[('Images', '*.jpg *.jpeg *.png *.bmp *.webp'), ('All files', '*.*')],
        )

    def copy_many(self, paths, dst_dir):
        count = 0
        for path in paths:
            if Path(path).suffix.lower() not in IMAGE_EXTS:
                continue
            unique_copy(path, dst_dir)
            count += 1
        return count

    def add_class(self):
        try:
            cfg, _ = self.config()
            project_root = cfg.get('paths', {}).get('project_root')
            if not project_root:
                raise RuntimeError('当前配置缺少 paths.project_root。')
        except Exception as exc:
            messagebox.showerror('新增失败', str(exc))
            return
        raw = simpledialog.askstring(
            '新增类别',
            '请输入新类别名称；可以一次输入多个，用空格、逗号或换行分隔：',
        )
        if not raw:
            return
        names = parse_class_names(raw)
        if not names:
            return
        try:
            storage_names = [add_class_to_project(project_root, name) for name in names]
        except Exception as exc:
            messagebox.showerror('新增失败', str(exc))
            return
        self.refresh_project_info()
        self.set_status(f'已新增类别：{", ".join(storage_names)}。重新加载会话后生效。')

    def start_learning(self):
        if self.session is None:
            self.initialize_and_advance()
            return
        self.run_worker('正在预测下一张 unknown 图片...', self.session.advance, self.render_state)

    def finish_session(self):
        if self.session is None:
            return
        self.run_worker('正在保存报告...', self.session.finish, self.render_state)

    def answer_question(self, key):
        self.run_worker(f'正在应用问题答案 {key} 并重新预测...', lambda: self.session.answer_question(key), self.render_state)

    def decide(self, decision):
        if self.session is None or self.current_state is None:
            messagebox.showwarning('尚未开始', '请先开始学习。')
            return
        label = self.class_var.get().strip()
        if decision == 'new':
            label = self.new_class_var.get().strip()
        reasons = [key for key, var in self.reason_vars.items() if var.get()]
        note = self.note_var.get().strip()
        if note.startswith('可选：'):
            note = ''
        self.run_worker(
            '正在保存用户反馈并进入下一张...',
            lambda: self.session.decide_current(decision, label=label, correction_reason_ids=reasons, correction_note=note),
            self.render_state,
        )

    def render_state(self, state):
        self.current_state = state
        mode = state.get('mode')
        if mode == 'done':
            self.render_done(state)
            return
        self.set_status(f'当前样本 {state.get("sample_index")}/{state.get("sample_total")}，模式：{mode}')
        self.sample_label.configure(text=state.get('sample_path') or '')
        self.render_image(state.get('sample_path'))
        self.render_results(state.get('results') or [])
        self.render_question(state.get('question'))
        classes = state.get('classes') or []
        self.class_var.configure(values=classes)
        if classes and not self.class_var.get():
            self.class_var.set(classes[0])
        status = [
            f'模式: {mode}',
            f'Top gap: {state.get("gap")}  Spread: {state.get("spread")}  Saturated: {state.get("saturated_ratio")}',
            f'不确定原因: {state.get("uncertainty_reason")}',
            f'当前特征权重: {state.get("weights")}',
        ]
        feedback = state.get('question_feedback')
        if feedback:
            status.append('问题回答: ' + feedback.get('answer_text', ''))
            status.append('权重更新: ' + str(feedback.get('weights_before')) + ' -> ' + str(feedback.get('weights_after')))
        self.state_text.configure(text='\n'.join([line for line in status if line]))
        self.refresh_project_info()

    def render_done(self, state):
        self.set_status(state.get('message', '完成。'))
        self.state_text.configure(text=state.get('message', '完成。'))
        self.question_text.configure(text='当前没有问题。')
        self.render_question(None)
        output_dir = state.get('output_dir', '')
        text = [
            state.get('message', '本轮学习结束。'),
            f'输出目录: {output_dir}',
            f'本轮日志数量: {state.get("logs_count", 0)}',
            '',
            '主要报告文件:',
            f'{output_dir}/experience_report.json',
            f'{output_dir}/feature_weights.json',
            f'{output_dir}/class_understanding_summary.md',
        ]
        self.report_text.configure(text='\n'.join(text))
        self.notebook.select(self.report_tab)
        self.refresh_project_info()

    def render_image(self, path):
        self.photo = None
        if not path:
            self.image_label.configure(image='', text='图片预览')
            return
        try:
            image = Image.open(path)
            image.thumbnail((380, 480))
            self.photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=self.photo, text='')
        except Exception as exc:
            self.image_label.configure(image='', text=f'无法预览图片:\n{exc}')

    def render_results(self, rows):
        self.result_tree.delete(*self.result_tree.get_children())
        for row in rows:
            nearest = ''
            if row.get('nearest_samples'):
                top = row['nearest_samples'][0]
                nearest = f'{top.get("score")}: {top.get("path")}'
            self.result_tree.insert(
                '',
                END,
                text=row.get('label', ''),
                values=(
                    row.get('score', ''),
                    row.get('detail', ''),
                    row.get('sources', ''),
                    nearest,
                ),
            )

    def render_question(self, question):
        for child in self.option_frame.winfo_children():
            child.destroy()
        if not question:
            self.question_text.configure(text='当前没有问题。')
            return
        parts = [
            question.get('evidence', ''),
            question.get('concept_evidence', ''),
            question.get('question', ''),
        ]
        self.question_text.configure(text='\n\n'.join([item for item in parts if item]))
        for option in question.get('options') or []:
            ttk.Button(
                self.option_frame,
                text=f"{option['key']}. {option['text']}",
                command=lambda key=option['key']: self.answer_question(key),
            ).pack(fill=X, pady=2)


def main():
    root = Tk()
    app = Ask2KnowDesktopApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
