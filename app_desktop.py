import shutil
import threading
import traceback
from datetime import datetime
from pathlib import Path
import yaml
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
    StringVar,
    filedialog,
    messagebox,
    simpledialog,
    Canvas,
)
from tkinter import ttk

from PIL import Image, ImageTk

from ask2know.data.dataset_loader import IMAGE_EXTS, DatasetLoader
from ask2know.deployment import (
    build_deployment_bundle,
    build_deployment_bundle_from_model_cache,
    load_deployment_bundle,
    predict_with_loaded_bundle,
)
from ask2know.evaluation import evaluate_unknown_audit_logs
from ask2know.features.feature_config import PRESET_DEFAULT_GROUPS, USER_FEATURE_GROUPS
from ask2know.runtime.project import create_task_project
from ask2know.runtime.session import LearningSession, add_class_to_project
from ask2know.sample_pool.manager import _safe_name
from ask2know.utils.io_utils import load_json, load_yaml, save_json
from scripts.package_model import build_package
from scripts.predict_folder import flatten_row, image_paths, write_csv


ALGORITHM_PROFILES = {
    'interactive_hierarchy': {
        'label': '分层交互识别',
        'description': '分层候选、最多两轮 ASK、支持在线反馈。',
    },
    'classic_similarity': {
        'label': '经典相似度识别',
        'description': '保留旧版原型相似度与单轮确认流程，适合兼容旧项目。',
    },
}

ALGORITHM_PROFILE_ALIASES = {
    'latest_online_ask2': 'interactive_hierarchy',
    'legacy_04631': 'classic_similarity',
}


def apply_algorithm_profile_to_config(config_path, profile_id):
    cfg = load_yaml(config_path)
    profile_id = ALGORITHM_PROFILE_ALIASES.get(profile_id, profile_id)
    profile_id = profile_id if profile_id in ALGORITHM_PROFILES else 'interactive_hierarchy'
    cfg['algorithm_profile'] = {
        'id': profile_id,
        'name': ALGORITHM_PROFILES[profile_id]['label'],
        'description': ALGORITHM_PROFILES[profile_id]['description'],
    }
    question_cfg = cfg.setdefault('question', {})
    similarity_cfg = cfg.setdefault('similarity', {})
    if profile_id == 'classic_similarity':
        question_cfg['max_questions_per_sample'] = 1
        question_cfg['enable_taxonomy_ask'] = False
        question_cfg['ask_candidate_top_k'] = 5
        question_cfg['max_taxonomy_options'] = 5
        similarity_cfg['recognition_mode'] = 'flat'
    else:
        question_cfg['max_questions_per_sample'] = 2
        question_cfg['enable_taxonomy_ask'] = True
        question_cfg['ask_candidate_top_k'] = 10
        question_cfg['max_taxonomy_options'] = 8
        similarity_cfg['recognition_mode'] = 'multilayer'
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return cfg


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
        self.algorithm_options = [
            (profile_id, profile['label'])
            for profile_id, profile in ALGORITHM_PROFILES.items()
        ]
        self.algorithm_label_to_id = {label: profile_id for profile_id, label in self.algorithm_options}
        self.algorithm_var = ttk.Combobox(
            body,
            values=[label for _, label in self.algorithm_options],
            state='readonly',
        )
        self.algorithm_var.set(ALGORITHM_PROFILES['interactive_hierarchy']['label'])

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
        self._row(body, 4, '算法模式', self.algorithm_var)

        self.feature_vars = {}
        features_frame = ttk.LabelFrame(body, text='用户可见特征', padding=8)
        features_frame.grid(row=5, column=0, columnspan=2, sticky='ew', pady=(8, 0))
        defaults = set(PRESET_DEFAULT_GROUPS['general'])
        for idx, name in enumerate(USER_FEATURE_GROUPS):
            var = BooleanVar(value=name in defaults)
            self.feature_vars[name] = var
            ttk.Checkbutton(features_frame, text=name, variable=var).grid(row=idx // 4, column=idx % 4, sticky='w', padx=8, pady=3)

        buttons = ttk.Frame(body)
        buttons.grid(row=6, column=0, columnspan=2, sticky='e', pady=(12, 0))
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
            profile_id = self.algorithm_label_to_id.get(self.algorithm_var.get(), 'interactive_hierarchy')
            apply_algorithm_profile_to_config(self.result['config_path'], profile_id)
        except Exception as exc:
            messagebox.showerror('创建失败', str(exc), parent=self.win)
            return
        self.win.destroy()


class Ask2KnowDesktopApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Ask2Know')
        self.root.geometry('1220x780')
        self.root.minsize(1040, 680)
        self._configure_style()
        self.session = None
        self.current_state = None
        self.photo = None
        self.deploy_photo = None
        self.config_path = None
        self.deploy_model_path = None
        self.deploy_model = None
        self.deploy_weights = None
        self.deploy_bundle = None
        self.nav_buttons = {}

        self.status_var = ttk.Label(root, text='请选择或新建项目。', anchor='w', style='Status.TLabel')
        self.status_var.pack(fill=X, side='bottom')

        self.shell = ttk.Frame(root, style='App.TFrame')
        self.shell.pack(fill=BOTH, expand=True)

        self.launch_frame = ttk.Frame(self.shell, padding=0, style='Launch.TFrame')
        self.workbench_frame = ttk.Frame(self.shell, padding=(14, 10, 14, 10), style='App.TFrame')
        self._build_launch_page(self.launch_frame)
        self._build_workbench(self.workbench_frame)
        self.show_launch_page()

    def _build_workbench(self, parent):
        self.main_frame = parent

        self.notebook = ttk.Notebook(self.main_frame, style='Workbench.TNotebook')

        self.project_tab = ttk.Frame(self.notebook, padding=12, style='Page.TFrame')
        self.dataset_tab = ttk.Frame(self.notebook, padding=12, style='Page.TFrame')
        self.trainset_tab = ttk.Frame(self.notebook, padding=12, style='Page.TFrame')
        self.learn_tab = ttk.Frame(self.notebook, padding=12, style='Page.TFrame')
        self.report_tab = ttk.Frame(self.notebook, padding=12, style='Page.TFrame')
        self.validate_tab = ttk.Frame(self.notebook, padding=12, style='Page.TFrame')
        self.deploy_tab = ttk.Frame(self.notebook, padding=12, style='Page.TFrame')
        self.analysis_tab = ttk.Frame(self.notebook, padding=12, style='Page.TFrame')
        self.notebook.add(self.project_tab, text='项目管理')
        self.notebook.add(self.dataset_tab, text='添加数据集')
        self.notebook.add(self.trainset_tab, text='创建训练集')
        self.notebook.add(self.learn_tab, text='训练模型')
        self.notebook.add(self.report_tab, text='训练报告')
        self.notebook.add(self.validate_tab, text='评估模型')
        self.notebook.add(self.analysis_tab, text='分析图片')
        self.notebook.add(self.deploy_tab, text='模型导出')

        self._build_app_header(self.main_frame)
        self.notebook.pack(fill=BOTH, expand=True)

        self._build_project_tab()
        self._build_dataset_tab()
        self._build_trainset_tab()
        self._build_learning_tab()
        self._build_report_tab()
        self._build_validate_tab()
        self._build_deploy_tab()
        self._build_analysis_tab()
        self.select_workbench_tab(self.project_tab, '项目管理')

    def _build_launch_page(self, parent):
        hero = ttk.Frame(parent, padding=(34, 30), style='Launch.TFrame')
        hero.pack(fill=BOTH, expand=True)

        left = ttk.Frame(hero, padding=(28, 24), style='Hero.TFrame')
        right = ttk.Frame(hero, padding=(26, 24), style='Launch.TFrame')
        left.pack(side=LEFT, fill=BOTH, expand=True)
        right.pack(side=RIGHT, fill=BOTH, expand=False, padx=(22, 0))

        ttk.Label(left, text='Ask2Know 工作台', style='HeroTitle.TLabel').pack(anchor='w')
        ttk.Label(
            left,
            text='低样本视觉识别与主动教学系统',
            style='HeroSubtitle.TLabel',
        ).pack(anchor='w', pady=(10, 0))
        ttk.Label(
            left,
            text='用于创建项目、管理训练数据、配置算法、训练模型、评估效果并分析图片。请选择右侧入口开始。',
            style='HeroBody.TLabel',
            wraplength=560,
        ).pack(anchor='w', pady=(28, 0))

        accent = ttk.Frame(left, style='HeroAccent.TFrame')
        accent.pack(fill=X, pady=(34, 0))
        for label, text in (
            ('项目', '创建或加载 task_config.yaml 项目'),
            ('训练', '导入数据并选择算法配置'),
            ('使用', '导出模型或直接分析图片'),
        ):
            block = ttk.Frame(accent, padding=(12, 10), style='HeroMetric.TFrame')
            block.pack(side=LEFT, fill=X, expand=True, padx=(0, 10))
            ttk.Label(block, text=label, style='HeroMetricTitle.TLabel').pack(anchor='w')
            ttk.Label(block, text=text, style='HeroMetricBody.TLabel', wraplength=150).pack(anchor='w', pady=(4, 0))

        card = ttk.Frame(right, padding=22, style='StartCard.TFrame')
        card.pack(fill=Y)
        ttk.Label(card, text='开始使用', style='StartTitle.TLabel').pack(anchor='w')
        ttk.Label(
            card,
            text='选择一个入口，系统会切换到项目工作台。',
            style='StartHint.TLabel',
            wraplength=310,
        ).pack(anchor='w', pady=(6, 18))
        ttk.Button(
            card,
            text='创建项目',
            command=self.create_project,
            style='Primary.TButton',
            width=28,
        ).pack(fill=X, pady=(0, 10))
        ttk.Button(
            card,
            text='加载项目',
            command=self.open_config,
            style='Secondary.TButton',
            width=28,
        ).pack(fill=X)
        ttk.Separator(card).pack(fill=X, pady=18)
        ttk.Label(
            card,
            text='创建项目时可以选择经典相似度识别，或分层交互识别；加载项目后也可以在“创建训练集”中调整。',
            style='StartHint.TLabel',
            wraplength=310,
        ).pack(anchor='w')

    def show_launch_page(self):
        self.workbench_frame.pack_forget()
        self.launch_frame.pack(fill=BOTH, expand=True)
        self.set_status('请选择创建项目或加载项目。')

    def show_workbench(self):
        self.launch_frame.pack_forget()
        self.workbench_frame.pack(fill=BOTH, expand=True)

    def _configure_style(self):
        self.root.configure(bg='#edf3f8')
        style = ttk.Style(self.root)
        try:
            style.theme_use('clam')
        except Exception:
            pass
        base_font = ('Microsoft YaHei UI', 10)
        title_font = ('Microsoft YaHei UI', 13, 'bold')
        section_font = ('Microsoft YaHei UI', 10, 'bold')
        style.configure('.', font=base_font)
        style.configure('App.TFrame', background='#edf3f8')
        style.configure('Launch.TFrame', background='#edf3f8')
        style.configure('Hero.TFrame', background='#17324d', relief='flat')
        style.configure('HeroAccent.TFrame', background='#17324d')
        style.configure('HeroMetric.TFrame', background='#24445f', relief='flat')
        style.configure('StartCard.TFrame', background='#ffffff', relief='solid', borderwidth=1)
        style.configure('Page.TFrame', background='#f8fafc')
        style.configure('Toolbar.TFrame', background='#dbe8f1')
        style.configure('TFrame', background='#f8fafc')
        style.configure('TLabel', background='#f8fafc', foreground='#1f2933')
        style.configure('Muted.TLabel', background='#f8fafc', foreground='#667085')
        style.configure('Title.TLabel', background='#edf3f8', foreground='#17324d', font=title_font)
        style.configure('Subtitle.TLabel', background='#edf3f8', foreground='#667085')
        style.configure('PageTitle.TLabel', background='#f8fafc', foreground='#17324d', font=title_font)
        style.configure('PageSubtitle.TLabel', background='#f8fafc', foreground='#667085')
        style.configure('HeroTitle.TLabel', background='#17324d', foreground='#ffffff', font=('Microsoft YaHei UI', 30, 'bold'))
        style.configure('HeroSubtitle.TLabel', background='#17324d', foreground='#b6e3ff', font=('Microsoft YaHei UI', 15, 'bold'))
        style.configure('HeroBody.TLabel', background='#17324d', foreground='#d7e5ee', font=('Microsoft YaHei UI', 11))
        style.configure('HeroChip.TLabel', background='#2a4a68', foreground='#ffffff', padding=(12, 7))
        style.configure('HeroMetricTitle.TLabel', background='#24445f', foreground='#ffffff', font=('Microsoft YaHei UI', 11, 'bold'))
        style.configure('HeroMetricBody.TLabel', background='#24445f', foreground='#d7e5ee')
        style.configure('StartTitle.TLabel', background='#ffffff', foreground='#17324d', font=('Microsoft YaHei UI', 17, 'bold'))
        style.configure('StartHint.TLabel', background='#ffffff', foreground='#667085')
        style.configure('Status.TLabel', background='#dbe8f1', foreground='#344054', padding=(12, 6))
        style.configure('TLabelframe', background='#f8fafc', bordercolor='#cbd8e2', relief='solid')
        style.configure('TLabelframe.Label', background='#f8fafc', foreground='#17324d', font=section_font)
        style.configure('TButton', padding=(10, 5), borderwidth=1)
        style.configure('Primary.TButton', padding=(14, 7), background='#1f6feb', foreground='#ffffff')
        style.map('Primary.TButton', background=[('active', '#1d4ed8'), ('pressed', '#1e40af')], foreground=[('active', '#ffffff')])
        style.configure('Secondary.TButton', padding=(14, 7), background='#e7eef5', foreground='#17324d')
        style.map('Secondary.TButton', background=[('active', '#dbe8f1'), ('pressed', '#cbd8e2')])
        style.configure('Nav.TButton', padding=(12, 7), background='#dbe8f1', foreground='#17324d')
        style.configure('NavActive.TButton', padding=(12, 7), background='#17324d', foreground='#ffffff')
        style.map('NavActive.TButton', background=[('active', '#17324d')], foreground=[('active', '#ffffff')])
        style.configure('TCheckbutton', background='#f8fafc', foreground='#1f2933')
        style.configure('TEntry', fieldbackground='#ffffff', padding=(6, 5))
        style.configure('Treeview', rowheight=26, fieldbackground='#ffffff', background='#ffffff', foreground='#1f2933')
        style.configure('Treeview.Heading', background='#dbe8f1', foreground='#344054', font=section_font)
        style.configure('Workbench.TNotebook', background='#edf3f8', borderwidth=0)
        style.layout('Workbench.TNotebook.Tab', [])

    def _build_app_header(self, parent):
        header = ttk.Frame(parent, style='App.TFrame')
        header.pack(fill=X, pady=(0, 10))
        brand = ttk.Frame(header, style='App.TFrame')
        brand.pack(side=LEFT, fill=Y)
        ttk.Label(brand, text='Ask2Know', style='Title.TLabel').pack(anchor='w')
        ttk.Label(brand, text='项目工作台', style='Subtitle.TLabel').pack(anchor='w')

        nav = ttk.Frame(header, padding=(10, 8), style='Toolbar.TFrame')
        nav.pack(side=RIGHT, fill=X, expand=True, padx=(18, 0))
        nav_items = [
            ('项目管理', self.project_tab),
            ('添加数据集', self.dataset_tab),
            ('创建训练集', self.trainset_tab),
            ('训练模型', self.learn_tab),
            ('评估模型', self.validate_tab),
            ('分析图片', self.analysis_tab),
            ('模型导出', self.deploy_tab),
        ]
        for label, tab in nav_items:
            button = ttk.Button(
                nav,
                text=label,
                command=lambda target=tab, name=label: self.select_workbench_tab(target, name),
                style='Nav.TButton',
            )
            button.pack(side=LEFT, padx=(0, 8))
            self.nav_buttons[label] = button
        ttk.Button(nav, text='返回首页', command=self.show_launch_page, style='Secondary.TButton').pack(side=RIGHT)

    def select_workbench_tab(self, tab, label=None):
        self.notebook.select(tab)
        active = label
        for name, button in self.nav_buttons.items():
            button.configure(style='NavActive.TButton' if name == active else 'Nav.TButton')

    def _build_project_tab(self):
        workflow = ttk.LabelFrame(self.project_tab, text='项目管理', padding=10)
        workflow.pack(fill=X, pady=(0, 10))
        project_actions = ttk.Frame(workflow)
        project_actions.pack(fill=X)
        ttk.Button(project_actions, text='打开项目配置', command=self.open_config, style='Primary.TButton').pack(side=LEFT)
        ttk.Button(project_actions, text='加载项目', command=self.initialize_session).pack(side=LEFT, padx=(8, 0))
        ttk.Button(project_actions, text='新增类别', command=self.add_class).pack(side=LEFT, padx=(8, 0))

        config_frame = ttk.LabelFrame(self.project_tab, text='当前配置', padding=10)
        config_frame.pack(fill=X, pady=(0, 10))
        self.config_var = ttk.Entry(config_frame)
        self.config_var.pack(fill=X)

        info_frame = ttk.LabelFrame(self.project_tab, text='项目内容', padding=8)
        info_frame.pack(fill=BOTH, expand=True)
        info_actions = ttk.Frame(info_frame)
        info_actions.pack(fill=X, pady=(0, 6))
        ttk.Button(info_actions, text='删除选中类别', command=self.remove_selected_project_classes).pack(side=LEFT)
        ttk.Label(
            info_actions,
            text='删除类别只会从项目清单移除，不会删除本地图片文件。',
            style='Muted.TLabel',
        ).pack(side=LEFT, padx=(10, 0))

        info_table = ttk.Frame(info_frame)
        info_table.pack(fill=BOTH, expand=True)
        self.project_class_items = {}
        self.project_info = ttk.Treeview(info_table, columns=('value',), show='tree headings', height=18, selectmode='extended')
        self.project_info.heading('#0', text='项目项')
        self.project_info.heading('value', text='值')
        self.project_info.column('#0', width=240, minwidth=160, stretch=False)
        self.project_info.column('value', width=820, minwidth=240, stretch=False)
        info_y_scroll = ttk.Scrollbar(info_table, orient='vertical', command=self.project_info.yview)
        info_x_scroll = ttk.Scrollbar(info_table, orient='horizontal', command=self.project_info.xview)
        self.project_info.configure(yscrollcommand=info_y_scroll.set, xscrollcommand=info_x_scroll.set)
        self.project_info.grid(row=0, column=0, sticky='nsew')
        info_y_scroll.grid(row=0, column=1, sticky='ns')
        info_x_scroll.grid(row=1, column=0, sticky='ew')
        info_table.rowconfigure(0, weight=1)
        info_table.columnconfigure(0, weight=1)

    def _build_dataset_tab(self):
        header = ttk.Frame(self.dataset_tab)
        header.pack(fill=X, pady=(0, 12))
        ttk.Label(header, text='添加数据集', style='PageTitle.TLabel').pack(anchor='w')
        ttk.Label(
            header,
            text='把训练图片和未标注图片导入当前项目。训练图片按类别保存，unlabeled 图片用于后续学习和评估。',
            style='PageSubtitle.TLabel',
        ).pack(anchor='w', pady=(3, 0))

        actions = ttk.LabelFrame(self.dataset_tab, text='数据导入', padding=12)
        actions.pack(fill=X)
        ttk.Button(actions, text='导入单类训练图片', command=self.import_train_images, style='Primary.TButton').grid(row=0, column=0, sticky='ew', padx=(0, 8), pady=(0, 8))
        ttk.Button(actions, text='批量导入训练文件夹', command=self.import_train_folder).grid(row=0, column=1, sticky='ew', padx=(0, 8), pady=(0, 8))
        ttk.Button(actions, text='导入未标注图片', command=self.import_unlabeled_images).grid(row=0, column=2, sticky='ew', pady=(0, 8))
        ttk.Label(
            actions,
            text='批量导入时，文件夹名会作为类别名；项目中不存在的类别会提示是否自动新增。',
            style='Muted.TLabel',
        ).grid(row=1, column=0, columnspan=3, sticky='w')
        for col in range(3):
            actions.columnconfigure(col, weight=1)

        summary = ttk.LabelFrame(self.dataset_tab, text='当前项目数据', padding=10)
        summary.pack(fill=BOTH, expand=True, pady=(10, 0))
        ttk.Label(
            summary,
            text='导入后可回到“项目管理”查看类别和训练样本数量。',
            style='Muted.TLabel',
        ).pack(anchor='w')
        ttk.Button(summary, text='刷新项目内容', command=self.refresh_project_info).pack(anchor='w', pady=(10, 0))

    def _build_trainset_tab(self):
        header = ttk.Frame(self.trainset_tab)
        header.pack(fill=X, pady=(0, 12))
        ttk.Label(header, text='创建训练集', style='PageTitle.TLabel').pack(anchor='w')
        ttk.Label(
            header,
            text='选择当前项目的训练和识别算法。配置会写入 task_config.yaml，后续训练、评估和导出都会使用这套设置。',
            style='PageSubtitle.TLabel',
        ).pack(anchor='w', pady=(3, 0))

        self.train_algorithm_var = StringVar(value=ALGORITHM_PROFILES['interactive_hierarchy']['label'])
        self.train_mode_var = StringVar(value='在线学习')
        self.train_questions_var = StringVar(value='2')
        self.train_candidate_var = StringVar(value='10')
        self.train_options_var = StringVar(value='8')

        form = ttk.LabelFrame(self.trainset_tab, text='训练集和算法配置', padding=12)
        form.pack(fill=X)
        algorithm_values = [profile['label'] for profile in ALGORITHM_PROFILES.values()]
        ttk.Label(form, text='算法模式').grid(row=0, column=0, sticky='w', padx=(0, 10), pady=(0, 8))
        self.train_algorithm_combo = ttk.Combobox(form, textvariable=self.train_algorithm_var, values=algorithm_values, state='readonly')
        self.train_algorithm_combo.grid(row=0, column=1, sticky='ew', pady=(0, 8))
        ttk.Label(form, text='学习方式').grid(row=1, column=0, sticky='w', padx=(0, 10), pady=(0, 8))
        ttk.Combobox(form, textvariable=self.train_mode_var, values=['在线学习', '离线/静态'], state='readonly').grid(row=1, column=1, sticky='ew', pady=(0, 8))
        ttk.Label(form, text='ASK 最大问题数').grid(row=2, column=0, sticky='w', padx=(0, 10), pady=(0, 8))
        ttk.Combobox(form, textvariable=self.train_questions_var, values=['1', '2'], state='readonly').grid(row=2, column=1, sticky='ew', pady=(0, 8))
        ttk.Label(form, text='候选 top-k').grid(row=3, column=0, sticky='w', padx=(0, 10), pady=(0, 8))
        ttk.Combobox(form, textvariable=self.train_candidate_var, values=['5', '10'], state='readonly').grid(row=3, column=1, sticky='ew', pady=(0, 8))
        ttk.Label(form, text='每问最大选项').grid(row=4, column=0, sticky='w', padx=(0, 10), pady=(0, 8))
        ttk.Combobox(form, textvariable=self.train_options_var, values=['5', '8', '10'], state='readonly').grid(row=4, column=1, sticky='ew', pady=(0, 8))

        buttons = ttk.Frame(form)
        buttons.grid(row=5, column=1, sticky='w', pady=(8, 0))
        ttk.Button(buttons, text='读取当前配置', command=self.refresh_trainset_options).pack(side=LEFT)
        ttk.Button(buttons, text='应用训练集配置', command=self.apply_trainset_options, style='Primary.TButton').pack(side=LEFT, padx=(8, 0))
        ttk.Button(buttons, text='加载训练模型', command=self.initialize_session).pack(side=LEFT, padx=(8, 0))
        form.columnconfigure(1, weight=1)

        info = ttk.LabelFrame(self.trainset_tab, text='说明', padding=10)
        info.pack(fill=X, pady=(10, 0))
        ttk.Label(
            info,
            text='经典相似度识别会使用一问/较小候选配置；分层交互识别会使用 top10、最多 8 个选项和最多两问。',
            style='Muted.TLabel',
            wraplength=960,
        ).pack(anchor='w')

    def _build_learning_tab(self):
        outer = ttk.Frame(self.learn_tab)
        outer.pack(fill=BOTH, expand=True)
        left = ttk.Frame(outer)
        right = ttk.Frame(outer)
        left.configure(width=460)
        left.pack(side=LEFT, fill=BOTH, expand=False)
        left.pack_propagate(False)
        right.pack(side=RIGHT, fill=BOTH, expand=True, padx=(12, 0))

        right_scroll_area = ttk.Frame(right)
        right_scroll_area.pack(fill=BOTH, expand=True, pady=(0, 8))
        right_canvas = Canvas(right_scroll_area, bg='#ffffff', highlightthickness=0)
        right_scrollbar = ttk.Scrollbar(right_scroll_area, orient='vertical', command=right_canvas.yview)
        right_content = ttk.Frame(right_canvas)
        right_window = right_canvas.create_window((0, 0), window=right_content, anchor='nw')
        right_canvas.configure(yscrollcommand=right_scrollbar.set)
        right_content.bind(
            '<Configure>',
            lambda _event: right_canvas.configure(scrollregion=right_canvas.bbox('all')),
        )
        right_canvas.bind(
            '<Configure>',
            lambda event: right_canvas.itemconfigure(right_window, width=event.width),
        )
        right_canvas.pack(side=LEFT, fill=BOTH, expand=True)
        right_scrollbar.pack(side=RIGHT, fill=Y)
        self.learning_right_canvas = right_canvas

        controls = ttk.LabelFrame(left, text='学习控制', padding=8)
        controls.pack(fill=X, pady=(0, 8))
        ttk.Button(controls, text='开始学习', command=self.start_learning, style='Primary.TButton').pack(side=LEFT)
        ttk.Button(controls, text='结束并保存报告', command=self.finish_session).pack(side=LEFT, padx=6)

        preview_frame = ttk.LabelFrame(left, text='样本预览', padding=8)
        preview_frame.pack(fill=BOTH, expand=True)
        self.image_label = ttk.Label(preview_frame, text='图片预览', anchor='center')
        self.image_label.pack(fill=BOTH, expand=True)
        self.sample_label = ttk.Label(preview_frame, text='', wraplength=430, style='Muted.TLabel')
        self.sample_label.pack(fill=X, pady=(8, 0))

        prediction_frame = ttk.LabelFrame(right_content, text='模型判断', padding=8)
        prediction_frame.pack(fill=X, pady=(0, 6))
        self.prediction_text = ttk.Label(
            prediction_frame,
            text='尚未开始预测。',
            wraplength=720,
            justify='left',
            font=('Microsoft YaHei UI', 10, 'bold'),
        )
        self.prediction_text.pack(fill=X)

        summary_frame = ttk.LabelFrame(right_content, text='当前状态', padding=8)
        summary_frame.pack(fill=X)
        self.state_text = ttk.Label(summary_frame, text='尚未开始。', wraplength=720, justify='left')
        self.state_text.pack(fill=X)

        result_frame = ttk.LabelFrame(right_content, text='预测结果', padding=8)
        result_frame.pack(fill=X, pady=6)
        result_table = ttk.Frame(result_frame)
        result_table.pack(fill=BOTH, expand=True)
        self.result_tree = ttk.Treeview(
            result_table,
            columns=('score', 'detail', 'sources', 'nearest'),
            show='tree headings',
            height=5,
        )
        self.result_tree.heading('#0', text='类别')
        self.result_tree.heading('score', text='分数')
        self.result_tree.heading('detail', text='特征')
        self.result_tree.heading('sources', text='来源')
        self.result_tree.heading('nearest', text='最近样本')
        self.result_tree.column('#0', width=130, minwidth=100, stretch=False)
        self.result_tree.column('score', width=70, minwidth=60, stretch=False)
        self.result_tree.column('detail', width=220, minwidth=160, stretch=False)
        self.result_tree.column('sources', width=220, minwidth=160, stretch=False)
        self.result_tree.column('nearest', width=360, minwidth=220, stretch=False)
        result_y_scroll = ttk.Scrollbar(result_table, orient='vertical', command=self.result_tree.yview)
        result_x_scroll = ttk.Scrollbar(result_table, orient='horizontal', command=self.result_tree.xview)
        self.result_tree.configure(yscrollcommand=result_y_scroll.set, xscrollcommand=result_x_scroll.set)
        self.result_tree.grid(row=0, column=0, sticky='nsew')
        result_y_scroll.grid(row=0, column=1, sticky='ns')
        result_x_scroll.grid(row=1, column=0, sticky='ew')
        result_table.rowconfigure(0, weight=1)
        result_table.columnconfigure(0, weight=1)

        self.question_frame = ttk.LabelFrame(right_content, text='系统提问', padding=8)
        self.question_frame.pack(fill=X, pady=(0, 6))
        self.question_text = ttk.Label(self.question_frame, text='当前没有问题。', wraplength=730, justify='left')
        self.question_text.pack(fill=X)
        self.option_frame = ttk.Frame(self.question_frame)
        self.option_frame.pack(fill=X, pady=(8, 0))

        decision = ttk.LabelFrame(right, text='确认模型判断', padding=8)
        decision.pack(fill=X, side='bottom')
        ttk.Label(
            decision,
            text='如果模型判断正确，直接点击确认；如果不正确，选择真实类别后提交修正。',
            style='Muted.TLabel',
        ).pack(fill=X, pady=(0, 6))
        row = ttk.Frame(decision)
        row.pack(fill=X)
        ttk.Label(row, text='真实类别').pack(side=LEFT)
        self.class_var = ttk.Combobox(row, values=[], state='readonly', width=20)
        self.class_var.pack(side=LEFT, padx=6)
        ttk.Label(row, text='新类别').pack(side=LEFT, padx=(12, 0))
        self.new_class_var = ttk.Entry(row, width=20)
        self.new_class_var.pack(side=LEFT, padx=6)

        reason_row = ttk.Frame(decision)
        reason_row.pack(fill=X, pady=(6, 0))
        self.reason_vars = {}
        for idx, reason in enumerate(('color', 'shape', 'texture', 'surface', 'part', 'text', 'sign', 'background', 'other')):
            var = BooleanVar(value=False)
            self.reason_vars[reason] = var
            ttk.Checkbutton(reason_row, text=reason, variable=var).grid(
                row=idx // 5,
                column=idx % 5,
                sticky='w',
                padx=(0, 8),
                pady=(0 if idx < 5 else 4, 0),
            )
        for col in range(5):
            reason_row.columnconfigure(col, weight=1)
        self.note_var = ttk.Entry(decision)
        self.note_var.pack(fill=X, pady=(6, 0))
        self.note_var.insert(0, '可选：写一句你区分这两个类别的依据')

        buttons = ttk.Frame(decision)
        buttons.pack(fill=X, pady=(6, 0))
        button_defs = [
            ('模型判断正确', 'correct', 'Primary.TButton'),
            ('改为所选类别', 'class', None),
            ('作为新类别', 'new', None),
            ('暂存 candidate', 'candidate', None),
            ('拒绝样本', 'reject', None),
            ('跳过', 'skip', None),
        ]
        for idx, (text, decision_key, style_name) in enumerate(button_defs):
            button = ttk.Button(
                buttons,
                text=text,
                command=lambda key=decision_key: self.decide(key),
                style=style_name or 'TButton',
            )
            if decision_key == 'correct':
                self.correct_button = button
            button.grid(row=idx // 3, column=idx % 3, sticky='ew', padx=(0 if idx % 3 == 0 else 6, 0), pady=(0 if idx < 3 else 6, 0))
        for col in range(3):
            buttons.columnconfigure(col, weight=1)

    def _build_report_tab(self):
        header = ttk.Frame(self.report_tab)
        header.pack(fill=X, pady=(0, 12))
        ttk.Label(header, text='学习报告', style='PageTitle.TLabel').pack(anchor='w')
        ttk.Label(header, text='结束学习后，这里会显示当前项目的学习结果和保存位置。', style='PageSubtitle.TLabel').pack(anchor='w', pady=(3, 0))

        report_frame = ttk.LabelFrame(self.report_tab, text='报告内容', padding=10)
        report_frame.pack(fill=BOTH, expand=True)
        self.report_text = ttk.Label(report_frame, text='报告会在结束学习后生成。', justify='left', wraplength=1000)
        self.report_text.pack(fill=X, anchor='nw')

    def _build_validate_tab(self):
        header = ttk.Frame(self.validate_tab)
        header.pack(fill=X, pady=(0, 12))
        ttk.Label(header, text='验证模型', style='PageTitle.TLabel').pack(anchor='w')
        ttk.Label(header, text='根据已确认的 unlabeled 学习记录验证当前模型。', style='PageSubtitle.TLabel').pack(anchor='w', pady=(3, 0))

        validate_frame = ttk.LabelFrame(self.validate_tab, text='验证', padding=10)
        validate_frame.pack(fill=X)
        ttk.Button(
            validate_frame,
            text='验证模型',
            command=self.validate_current_project,
            style='Primary.TButton',
            width=18,
        ).pack(anchor='w')

        result_frame = ttk.LabelFrame(self.validate_tab, text='验证结果', padding=10)
        result_frame.pack(fill=X, pady=(10, 0))
        self.validation_summary = ttk.Label(result_frame, text='尚未验证。', justify='left', wraplength=1000)
        self.validation_summary.pack(fill=X, anchor='nw')

    def _build_deploy_tab(self):
        model_frame = ttk.LabelFrame(self.deploy_tab, text='离线模型', padding=8)
        model_frame.pack(fill=X)
        self.deploy_model_var = ttk.Entry(model_frame)
        self.deploy_model_var.pack(side=LEFT, fill=X, expand=True)
        ttk.Button(model_frame, text='选择模型', command=self.choose_deploy_model).pack(side=LEFT, padx=6)
        ttk.Button(model_frame, text='加载模型', command=self.load_deploy_model).pack(side=LEFT)

        body = ttk.Frame(self.deploy_tab)
        body.pack(fill=BOTH, expand=True, pady=(10, 0))
        left = ttk.Frame(body)
        right = ttk.Frame(body)
        left.pack(side=LEFT, fill=BOTH, expand=False)
        right.pack(side=RIGHT, fill=BOTH, expand=True, padx=(12, 0))

        single = ttk.LabelFrame(left, text='单张图片识别', padding=8)
        single.pack(fill=X)
        self.deploy_image_var = ttk.Entry(single, width=54)
        self.deploy_image_var.pack(fill=X)
        image_buttons = ttk.Frame(single)
        image_buttons.pack(fill=X, pady=(6, 0))
        ttk.Button(image_buttons, text='选择图片', command=self.choose_deploy_image).pack(side=LEFT)
        ttk.Button(image_buttons, text='识别图片', command=self.predict_deploy_image).pack(side=LEFT, padx=6)

        self.deploy_preview = ttk.Label(left, text='图片预览', anchor='center')
        self.deploy_preview.pack(fill=BOTH, expand=True, pady=(10, 0))

        folder = ttk.LabelFrame(left, text='批量识别', padding=8)
        folder.pack(fill=X, pady=(10, 0))
        ttk.Label(folder, text='输入文件夹').pack(anchor='w')
        self.deploy_folder_var = ttk.Entry(folder, width=54)
        self.deploy_folder_var.pack(fill=X)
        ttk.Label(folder, text='输出 CSV').pack(anchor='w', pady=(6, 0))
        self.deploy_csv_var = ttk.Entry(folder, width=54)
        self.deploy_csv_var.pack(fill=X)
        folder_buttons = ttk.Frame(folder)
        folder_buttons.pack(fill=X, pady=(6, 0))
        ttk.Button(folder_buttons, text='选择文件夹', command=self.choose_deploy_folder).pack(side=LEFT)
        ttk.Button(folder_buttons, text='选择输出', command=self.choose_deploy_csv).pack(side=LEFT, padx=6)
        ttk.Button(folder_buttons, text='批量识别', command=self.predict_deploy_folder).pack(side=LEFT)

        package = ttk.LabelFrame(left, text='导出 Python 离线包', padding=8)
        package.pack(fill=X, pady=(10, 0))
        self.package_output_var = ttk.Entry(package, width=54)
        self.package_output_var.pack(fill=X)
        package_buttons = ttk.Frame(package)
        package_buttons.pack(fill=X, pady=(6, 0))
        self.package_server_var = BooleanVar(value=True)
        ttk.Checkbutton(package_buttons, text='包含服务脚本', variable=self.package_server_var).pack(side=LEFT)
        ttk.Button(package_buttons, text='选择输出目录', command=self.choose_package_output).pack(side=LEFT, padx=6)
        ttk.Button(package_buttons, text='生成离线包', command=self.package_deploy_model).pack(side=LEFT)

        result_frame = ttk.LabelFrame(right, text='识别结果', padding=8)
        result_frame.pack(fill=BOTH, expand=True)
        self.deploy_summary = ttk.Label(result_frame, text='请先加载离线模型。', justify='left', wraplength=760)
        self.deploy_summary.pack(fill=X)
        self.deploy_result_tree = ttk.Treeview(
            result_frame,
            columns=('score', 'margin', 'sources', 'nearest'),
            show='tree headings',
            height=14,
        )
        self.deploy_result_tree.heading('#0', text='类别')
        self.deploy_result_tree.heading('score', text='分数')
        self.deploy_result_tree.heading('margin', text='Top2差距')
        self.deploy_result_tree.heading('sources', text='证据')
        self.deploy_result_tree.heading('nearest', text='最近样本')
        self.deploy_result_tree.column('#0', width=180)
        self.deploy_result_tree.column('score', width=80)
        self.deploy_result_tree.column('margin', width=80)
        self.deploy_result_tree.column('sources', width=260)
        self.deploy_result_tree.column('nearest', width=280)
        self.deploy_result_tree.pack(fill=BOTH, expand=True, pady=(8, 0))

    def _build_analysis_tab(self):
        header = ttk.Frame(self.analysis_tab)
        header.pack(fill=X, pady=(0, 10))
        ttk.Label(header, text='分析图片', style='PageTitle.TLabel').pack(anchor='w')
        ttk.Label(
            header,
            text='加载训练后导出的模型，添加图片后直接查看识别结果。',
            style='PageSubtitle.TLabel',
        ).pack(anchor='w', pady=(3, 0))

        model_frame = ttk.LabelFrame(self.analysis_tab, text='已训练模型', padding=8)
        model_frame.pack(fill=X)
        self.deploy_model_var = ttk.Entry(model_frame)
        self.deploy_model_var.pack(side=LEFT, fill=X, expand=True)
        ttk.Button(model_frame, text='使用当前项目模型', command=self.load_current_project_for_analysis).pack(side=LEFT, padx=(6, 0))
        ttk.Button(model_frame, text='选择模型', command=self.choose_deploy_model).pack(side=LEFT, padx=6)
        ttk.Button(model_frame, text='加载模型', command=self.load_deploy_model, style='Primary.TButton').pack(side=LEFT)

        body = ttk.Frame(self.analysis_tab)
        body.pack(fill=BOTH, expand=True, pady=(10, 0))
        left = ttk.Frame(body)
        right = ttk.Frame(body)
        left.pack(side=LEFT, fill=BOTH, expand=False)
        right.pack(side=RIGHT, fill=BOTH, expand=True, padx=(12, 0))

        single = ttk.LabelFrame(left, text='单张图片分析', padding=8)
        single.pack(fill=X)
        self.deploy_image_var = ttk.Entry(single, width=54)
        self.deploy_image_var.pack(fill=X)
        image_buttons = ttk.Frame(single)
        image_buttons.pack(fill=X, pady=(6, 0))
        ttk.Button(image_buttons, text='添加图片', command=self.choose_deploy_image).pack(side=LEFT)
        ttk.Button(image_buttons, text='分析图片', command=self.predict_deploy_image).pack(side=LEFT, padx=6)

        self.deploy_preview = ttk.Label(left, text='图片预览', anchor='center')
        self.deploy_preview.pack(fill=BOTH, expand=True, pady=(10, 0))

        folder = ttk.LabelFrame(left, text='批量分析', padding=8)
        folder.pack(fill=X, pady=(10, 0))
        ttk.Label(folder, text='输入文件夹').pack(anchor='w')
        self.deploy_folder_var = ttk.Entry(folder, width=54)
        self.deploy_folder_var.pack(fill=X)
        ttk.Label(folder, text='输出 CSV').pack(anchor='w', pady=(6, 0))
        self.deploy_csv_var = ttk.Entry(folder, width=54)
        self.deploy_csv_var.pack(fill=X)
        folder_buttons = ttk.Frame(folder)
        folder_buttons.pack(fill=X, pady=(6, 0))
        ttk.Button(folder_buttons, text='选择文件夹', command=self.choose_deploy_folder).pack(side=LEFT)
        ttk.Button(folder_buttons, text='选择输出', command=self.choose_deploy_csv).pack(side=LEFT, padx=6)
        ttk.Button(folder_buttons, text='批量分析', command=self.predict_deploy_folder).pack(side=LEFT)

        result_frame = ttk.LabelFrame(right, text='分析结果', padding=8)
        result_frame.pack(fill=BOTH, expand=True)
        self.deploy_summary = ttk.Label(result_frame, text='请先加载训练好的模型。', justify='left', wraplength=760)
        self.deploy_summary.pack(fill=X)
        self.deploy_result_tree = ttk.Treeview(
            result_frame,
            columns=('score', 'margin', 'sources', 'nearest'),
            show='tree headings',
            height=14,
        )
        self.deploy_result_tree.heading('#0', text='类别')
        self.deploy_result_tree.heading('score', text='分数')
        self.deploy_result_tree.heading('margin', text='Top2差距')
        self.deploy_result_tree.heading('sources', text='证据')
        self.deploy_result_tree.heading('nearest', text='最近样本')
        self.deploy_result_tree.column('#0', width=180)
        self.deploy_result_tree.column('score', width=80)
        self.deploy_result_tree.column('margin', width=80)
        self.deploy_result_tree.column('sources', width=260)
        self.deploy_result_tree.column('nearest', width=280)
        self.deploy_result_tree.pack(fill=BOTH, expand=True, pady=(8, 0))

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
        if hasattr(self, 'export_config_var'):
            self.export_config_var.set(self.config_path)
        self.refresh_project_info()
        self.refresh_trainset_options()
        self.show_workbench()
        self.select_workbench_tab(self.project_tab, '项目管理')
        self.set_status('项目已创建，请导入训练图片和 unlabeled 图片。')

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
        if hasattr(self, 'export_config_var'):
            self.export_config_var.set(path)
        self.refresh_project_info()
        self.refresh_trainset_options()
        self.show_workbench()
        self.select_workbench_tab(self.project_tab, '项目管理')

    def config(self):
        path = self.config_var.get().strip()
        if not path:
            raise RuntimeError('请先选择 task_config.yaml。')
        return load_yaml(path), Path(path)

    def refresh_trainset_options(self):
        try:
            cfg, _ = self.config()
        except Exception as exc:
            messagebox.showerror('读取失败', str(exc))
            return
        profile = cfg.get('algorithm_profile') or {}
        profile_id = profile.get('id') or ('classic_similarity' if cfg.get('similarity', {}).get('recognition_mode') == 'flat' else 'interactive_hierarchy')
        profile_id = ALGORITHM_PROFILE_ALIASES.get(profile_id, profile_id)
        profile_id = profile_id if profile_id in ALGORITHM_PROFILES else 'interactive_hierarchy'
        self.train_algorithm_var.set(ALGORITHM_PROFILES[profile_id]['label'])
        runtime_mode = cfg.get('runtime', {}).get('learning_mode', 'online')
        self.train_mode_var.set('离线/静态' if runtime_mode == 'offline' else '在线学习')
        question_cfg = cfg.get('question') or {}
        self.train_questions_var.set(str(question_cfg.get('max_questions_per_sample', 2)))
        self.train_candidate_var.set(str(question_cfg.get('ask_candidate_top_k', 10)))
        self.train_options_var.set(str(question_cfg.get('max_taxonomy_options', 8)))
        self.set_status('已读取当前训练集配置。')

    def apply_trainset_options(self):
        try:
            _cfg, config_path = self.config()
        except Exception as exc:
            messagebox.showerror('应用失败', str(exc))
            return
        label_to_profile = {
            profile['label']: profile_id
            for profile_id, profile in ALGORITHM_PROFILES.items()
        }
        profile_id = label_to_profile.get(self.train_algorithm_var.get(), 'interactive_hierarchy')
        cfg = apply_algorithm_profile_to_config(config_path, profile_id)
        question_cfg = cfg.setdefault('question', {})
        question_cfg['max_questions_per_sample'] = int(self.train_questions_var.get() or question_cfg.get('max_questions_per_sample', 2))
        question_cfg['ask_candidate_top_k'] = int(self.train_candidate_var.get() or question_cfg.get('ask_candidate_top_k', 10))
        question_cfg['max_taxonomy_options'] = int(self.train_options_var.get() or question_cfg.get('max_taxonomy_options', 8))
        mode = 'offline' if self.train_mode_var.get() == '离线/静态' else 'online'
        cfg.setdefault('runtime', {})['learning_mode'] = mode
        cfg.setdefault('online_learning', {})['enable'] = mode == 'online'
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        self.refresh_project_info()
        self.set_status('训练集和算法配置已应用。')

    def refresh_project_info(self):
        self.project_info.delete(*self.project_info.get_children())
        self.project_class_items = {}
        try:
            cfg, config_path = self.config()
            dataset_dir = Path(cfg['paths']['dataset_dir'])
            loader = DatasetLoader(dataset_dir)
            objects = loader.load_objects()
            train_samples = loader.load_train_samples()
            unlabeled_samples = loader.load_unlabeled_samples()
        except Exception as exc:
            self.project_info.insert('', END, text='读取失败', values=(str(exc),))
            return
        items = {
            '配置文件': str(config_path),
            '项目目录': cfg.get('paths', {}).get('project_root', ''),
            '数据目录': str(dataset_dir),
            '输出目录': cfg.get('paths', {}).get('output_dir', ''),
            '算法模式': (cfg.get('algorithm_profile') or {}).get('name', cfg.get('similarity', {}).get('recognition_mode', '')),
            '学习方式': cfg.get('runtime', {}).get('learning_mode', 'online'),
            '类别': ', '.join([item.get('name', '') for item in objects]),
            '训练样本数': str(len(train_samples)),
            'unlabeled 样本数': str(len(unlabeled_samples)),
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
            iid = self.project_info.insert(class_root, END, text=display, values=(f'{counts.get(name, 0)} 张',), tags=('class',))
            self.project_class_items[iid] = name

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

        self.run_worker('正在加载模型、提取训练特征并预测第一张 unlabeled 图片...', build_and_advance, done)

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

    def import_train_folder(self):
        try:
            cfg, _ = self.config()
            dataset_dir = Path(cfg['paths']['dataset_dir'])
            project_root = cfg.get('paths', {}).get('project_root')
            if not project_root:
                raise RuntimeError('当前配置缺少 paths.project_root。')
            loader = DatasetLoader(dataset_dir)
            objects = loader.load_objects()
        except Exception as exc:
            messagebox.showerror('导入失败', str(exc))
            return

        root = filedialog.askdirectory(title='选择包含类别子文件夹的训练图片目录')
        if not root:
            return
        source_root = Path(root)
        class_dirs = [path for path in source_root.iterdir() if path.is_dir()]
        if not class_dirs:
            messagebox.showwarning('没有类别文件夹', '请选择包含类别子文件夹的目录。')
            return

        class_map = {}
        for item in objects:
            name = item.get('name')
            display_name = item.get('display_name')
            if name:
                class_map[name] = name
            if display_name:
                class_map[display_name] = name

        plan = []
        missing = []
        total_images = 0
        for class_dir in class_dirs:
            image_paths = [
                path for path in class_dir.rglob('*')
                if path.is_file() and path.suffix.lower() in IMAGE_EXTS
            ]
            if not image_paths:
                continue
            class_name = class_map.get(class_dir.name) or class_map.get(_safe_name(class_dir.name))
            if not class_name:
                class_name = _safe_name(class_dir.name)
                missing.append(class_dir.name)
            total_images += len(image_paths)
            plan.append((class_dir.name, class_name, image_paths))

        if not plan:
            messagebox.showwarning('没有图片', '类别子文件夹中没有可导入的图片。')
            return
        if missing:
            preview = ', '.join(missing[:12])
            suffix = '...' if len(missing) > 12 else ''
            ok = messagebox.askyesno(
                '自动新增类别',
                f'发现 {len(missing)} 个项目中不存在的类别，将按文件夹名自动新增：\n{preview}{suffix}\n\n是否继续导入？',
            )
            if not ok:
                return

        def import_all():
            imported = 0
            by_class = {}
            known = set(class_map.values())
            for display_name, class_name, image_paths in plan:
                storage_name = class_name
                if storage_name not in known:
                    storage_name = add_class_to_project(project_root, display_name)
                    known.add(storage_name)
                dst_dir = dataset_dir / 'train' / storage_name
                count = self.copy_many(image_paths, dst_dir)
                imported += count
                by_class[storage_name] = by_class.get(storage_name, 0) + count
            return imported, len(by_class), by_class

        def done(result):
            imported, class_count, _by_class = result
            self.refresh_project_info()
            self.set_status(f'已批量导入 {imported} 张训练图片，覆盖 {class_count} 个类别。')
            messagebox.showinfo('导入完成', f'已导入 {imported} 张训练图片。\n类别数：{class_count}')

        self.run_worker(f'正在批量导入 {total_images} 张训练图片...', import_all, done)

    def import_unlabeled_images(self):
        try:
            cfg, _ = self.config()
            dst_dir = Path(cfg['paths']['dataset_dir']) / 'unlabeled'
        except Exception as exc:
            messagebox.showerror('导入失败', str(exc))
            return
        paths = self.choose_images()
        if not paths:
            return
        count = self.copy_many(paths, dst_dir)
        self.refresh_project_info()
        self.set_status(f'已导入 {count} 张 unlabeled 图片。')

    def import_unknown_images(self):
        self.import_unlabeled_images()

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

    def remove_selected_project_classes(self):
        selected = [
            self.project_class_items[item_id]
            for item_id in self.project_info.selection()
            if item_id in self.project_class_items
        ]
        selected = list(dict.fromkeys(selected))
        if not selected:
            messagebox.showwarning('未选择类别', '请在“类别明细”中选中要删除的类别。')
            return

        try:
            cfg, config_path = self.config()
            dataset_dir = Path(cfg['paths']['dataset_dir'])
            objects_path = dataset_dir / 'objects.json'
            objects = DatasetLoader(dataset_dir).load_objects()
        except Exception as exc:
            messagebox.showerror('删除失败', str(exc))
            return

        remaining = [item for item in objects if item.get('name') not in selected]
        if len(remaining) < 2:
            messagebox.showwarning('类别不足', '项目至少需要保留 2 个类别用于学习。')
            return

        names_text = ', '.join(selected)
        ok = messagebox.askyesno(
            '删除类别',
            f'将从项目清单中删除以下类别：\n{names_text}\n\n本地训练图片文件不会被删除。是否继续？',
        )
        if not ok:
            return

        try:
            save_json(objects_path, {'objects': remaining})
            cfg['classes'] = [name for name in cfg.get('classes', []) if name not in selected]
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        except Exception as exc:
            messagebox.showerror('删除失败', str(exc))
            return

        self.session = None
        self.current_state = None
        self.class_var.configure(values=[])
        self.class_var.set('')
        self.refresh_project_info()
        self.set_status(f'已从项目清单删除类别：{names_text}。请重新加载项目。')

    def start_learning(self):
        if self.session is None:
            self.initialize_and_advance()
            return
        self.run_worker('正在预测下一张 unlabeled 图片...', self.session.advance, self.render_state)

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
        results = state.get('results') or []
        self.render_results(results)
        self.render_prediction_summary(results, state)
        self.render_question(state.get('question'))
        classes = state.get('classes') or []
        self.class_var.configure(values=classes)
        predicted_label = results[0].get('label') if results else ''
        if predicted_label and predicted_label in classes:
            self.class_var.set(predicted_label)
        elif classes and not self.class_var.get():
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
        self.prediction_text.configure(text='学习已结束。')
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

    def load_current_project_for_analysis(self):
        config_path = str(self.config_path or self.config_var.get().strip() or '')
        if not config_path:
            messagebox.showwarning('缺少项目', '请先创建或加载项目。')
            return

        def build():
            _cfg, model_path, _package_dir, cache_path = self._default_deploy_paths(
                config_path,
                export_name='current_project_analysis_model',
            )
            if cache_path:
                path, _bundle = build_deployment_bundle_from_model_cache(
                    config_path,
                    cache_path,
                    output_path=model_path,
                    include_sample_features=True,
                )
            else:
                path, _bundle = build_deployment_bundle(
                    config_path,
                    output_path=model_path,
                    include_sample_features=True,
                )
            model, weights, bundle = load_deployment_bundle(path)
            return Path(path).expanduser().resolve(), model, weights, bundle

        def done(result):
            self.deploy_model_path, self.deploy_model, self.deploy_weights, self.deploy_bundle = result
            self.deploy_model_var.delete(0, END)
            self.deploy_model_var.insert(0, str(self.deploy_model_path))
            classes = self.deploy_bundle.get('classes') or []
            self.deploy_summary.configure(text=f'当前项目模型已加载：{self.deploy_model_path}\n类别数: {len(classes)}')
            self.set_status('当前项目模型已加载，可以添加图片分析。')

        self.run_worker('正在加载当前项目模型用于图片分析...', build, done)

    def choose_deploy_model(self):
        path = filedialog.askopenfilename(
            title='选择 Ask2Know 模型',
            filetypes=[('Ask2Know model', '*.a2kmodel.json'), ('JSON', '*.json'), ('All files', '*.*')],
        )
        if path:
            self.deploy_model_var.delete(0, END)
            self.deploy_model_var.insert(0, path)

    def load_deploy_model(self):
        path = self.deploy_model_var.get().strip()
        if not path:
            messagebox.showwarning('缺少模型', '请先选择 .a2kmodel.json 模型文件。')
            return

        def build():
            model, weights, bundle = load_deployment_bundle(path)
            return Path(path).expanduser().resolve(), model, weights, bundle

        def done(result):
            self.deploy_model_path, self.deploy_model, self.deploy_weights, self.deploy_bundle = result
            classes = self.deploy_bundle.get('classes') or []
            task = self.deploy_bundle.get('task') or {}
            self.deploy_summary.configure(
                text=f'模型已加载: {self.deploy_model_path}\n任务: {task.get("name", "")}\n类别数: {len(classes)}'
            )
            self.set_status('离线模型已加载。')

        self.run_worker('正在加载离线模型...', build, done)

    def _require_deploy_model(self):
        if self.deploy_model is None or self.deploy_weights is None or self.deploy_bundle is None:
            raise RuntimeError('请先加载离线模型。')

    def choose_deploy_image(self):
        path = filedialog.askopenfilename(
            title='选择要识别的图片',
            filetypes=[('Images', '*.jpg *.jpeg *.png *.bmp *.webp'), ('All files', '*.*')],
        )
        if path:
            target_path = path
            try:
                cfg, _ = self.config()
                unknown_dir = Path(cfg['paths']['dataset_dir']) / 'unknown'
                target_path = str(unique_copy(path, unknown_dir))
                self.set_status(f'已添加分析图片到 unknown：{target_path}')
            except Exception:
                target_path = path
            self.deploy_image_var.delete(0, END)
            self.deploy_image_var.insert(0, target_path)
            self.render_deploy_image(target_path)

    def render_deploy_image(self, path):
        self.deploy_photo = None
        if not path:
            self.deploy_preview.configure(image='', text='图片预览')
            return
        try:
            image = Image.open(path)
            image.thumbnail((380, 300))
            self.deploy_photo = ImageTk.PhotoImage(image)
            self.deploy_preview.configure(image=self.deploy_photo, text='')
        except Exception as exc:
            self.deploy_preview.configure(image='', text=f'无法预览图片:\n{exc}')

    def predict_deploy_image(self):
        image_path = self.deploy_image_var.get().strip()
        if not image_path:
            messagebox.showwarning('缺少图片', '请先选择要识别的图片。')
            return

        def predict():
            self._require_deploy_model()
            return predict_with_loaded_bundle(
                self.deploy_model,
                self.deploy_weights,
                self.deploy_bundle,
                self.deploy_model_path,
                image_path,
                top_k=5,
            )

        self.run_worker('正在识别图片...', predict, self.render_deploy_prediction)

    def render_deploy_prediction(self, result):
        predictions = result.get('predictions') or []
        self.deploy_result_tree.delete(*self.deploy_result_tree.get_children())
        self.deploy_summary.configure(
            text=(
                f'图片: {result.get("image")}\n'
                f'预测: {result.get("predicted_label")}  '
                f'置信度: {result.get("confidence")}  '
                f'Top2差距: {result.get("top2_margin")}'
            )
        )
        for row in predictions:
            sources = row.get('sources') or {}
            source_text = ', '.join(f'{key}={value}' for key, value in sources.items())
            nearest = ''
            if row.get('nearest_samples'):
                item = row['nearest_samples'][0]
                nearest = f'{item.get("score")}: {item.get("path")}'
            self.deploy_result_tree.insert(
                '',
                END,
                text=row.get('label', ''),
                values=(row.get('score', ''), result.get('top2_margin', ''), source_text, nearest),
            )
        self.render_deploy_image(result.get('image'))
        self.set_status('图片识别完成。')

    def choose_deploy_folder(self):
        path = filedialog.askdirectory(title='选择要批量识别的图片文件夹')
        if path:
            self.deploy_folder_var.delete(0, END)
            self.deploy_folder_var.insert(0, path)
            if not self.deploy_csv_var.get().strip():
                self.deploy_csv_var.insert(0, str(Path(path) / 'predictions.csv'))

    def choose_deploy_csv(self):
        path = filedialog.asksaveasfilename(
            title='选择 CSV 输出文件',
            defaultextension='.csv',
            filetypes=[('CSV', '*.csv'), ('All files', '*.*')],
        )
        if path:
            self.deploy_csv_var.delete(0, END)
            self.deploy_csv_var.insert(0, path)

    def predict_deploy_folder(self):
        folder_path = self.deploy_folder_var.get().strip()
        csv_path = self.deploy_csv_var.get().strip()
        if not folder_path or not csv_path:
            messagebox.showwarning('缺少路径', '请选择输入文件夹和输出 CSV。')
            return

        def predict():
            self._require_deploy_model()
            paths = image_paths(folder_path, recursive=False)
            if not paths:
                raise RuntimeError('输入文件夹中没有图片。')
            flat_rows = []
            for path in paths:
                result = predict_with_loaded_bundle(
                    self.deploy_model,
                    self.deploy_weights,
                    self.deploy_bundle,
                    self.deploy_model_path,
                    path,
                    top_k=5,
                )
                flat_rows.append(flatten_row(result, 5))
            write_csv(csv_path, flat_rows, 5)
            return len(flat_rows), Path(csv_path).expanduser().resolve()

        def done(result):
            count, output = result
            self.deploy_summary.configure(text=f'批量识别完成: {count} 张图片\nCSV: {output}')
            self.set_status(f'批量识别完成: {count} 张。')

        self.run_worker('正在批量识别图片...', predict, done)

    def choose_package_output(self):
        parent = filedialog.askdirectory(title='选择离线包父目录')
        if not parent:
            return
        model_path = Path(self.deploy_model_var.get().strip() or 'ask2know_model.a2kmodel.json')
        output = Path(parent) / f'{model_path.stem}_python_package'
        self.package_output_var.delete(0, END)
        self.package_output_var.insert(0, str(output))

    def package_deploy_model(self):
        model_path = self.deploy_model_var.get().strip()
        output_dir = self.package_output_var.get().strip()
        if not model_path or not output_dir:
            messagebox.showwarning('缺少路径', '请选择模型文件和离线包输出目录。')
            return

        def package():
            return build_package(
                model_path,
                output_dir,
                include_server=bool(self.package_server_var.get()),
            )

        def done(result):
            self.deploy_summary.configure(text=f'Python 离线包已生成:\n{result}')
            self.set_status('Python 离线包已生成。')

        self.run_worker('正在生成 Python 离线包...', package, done)

    def _build_deploy_tab(self):
        header = ttk.Frame(self.deploy_tab)
        header.pack(fill=X, pady=(0, 10))
        ttk.Label(
            header,
            text='导出离线模型包',
            style='PageTitle.TLabel',
        ).pack(anchor='w')
        ttk.Label(
            header,
            text='完成项目训练后，一键生成可离线使用的 Python 模型包。',
            style='PageSubtitle.TLabel',
        ).pack(anchor='w', pady=(3, 0))

        self.export_config_var = StringVar(value=str(self.config_path or ''))
        self.export_cache_var = StringVar(value='')
        self.export_model_output_var = StringVar(value='')
        self.export_dir_var = StringVar(value='')
        self.export_name_var = StringVar(value='')
        self.package_model_var = StringVar(value='')
        self.package_output_var = StringVar(value='')
        self.export_include_samples_var = BooleanVar(value=True)
        self.package_server_var = BooleanVar(value=False)

        export_frame = ttk.LabelFrame(self.deploy_tab, text='一键导出', padding=10)
        export_frame.pack(fill=X, pady=(0, 10))

        ttk.Label(export_frame, text='当前项目').grid(row=0, column=0, sticky='nw', padx=(0, 10), pady=(0, 8))
        ttk.Label(
            export_frame,
            textvariable=self.export_config_var,
            foreground='#333333',
            wraplength=900,
            justify='left',
        ).grid(row=0, column=1, sticky='ew', pady=(0, 8))

        project_buttons = ttk.Frame(export_frame)
        project_buttons.grid(row=1, column=1, sticky='w', pady=(0, 12))
        ttk.Button(project_buttons, text='使用当前项目', command=self.use_current_export_project).pack(side=LEFT)
        ttk.Button(project_buttons, text='选择项目配置', command=self.choose_export_config).pack(side=LEFT, padx=(8, 0))

        ttk.Label(export_frame, text='导出位置').grid(row=2, column=0, sticky='w', padx=(0, 10), pady=(0, 8))
        self.export_dir_entry = ttk.Entry(export_frame, textvariable=self.export_dir_var)
        self.export_dir_entry.grid(row=2, column=1, sticky='ew', pady=(0, 8))
        ttk.Button(export_frame, text='选择', command=self.choose_export_dir).grid(row=2, column=2, sticky='e', padx=(8, 0), pady=(0, 8))

        ttk.Label(export_frame, text='模型名称').grid(row=3, column=0, sticky='w', padx=(0, 10), pady=(0, 8))
        self.export_name_entry = ttk.Entry(export_frame, textvariable=self.export_name_var)
        self.export_name_entry.grid(row=3, column=1, sticky='ew', pady=(0, 8))
        ttk.Label(
            export_frame,
            text='可选；不填则使用默认位置和默认名称。',
            style='Muted.TLabel',
        ).grid(row=4, column=1, sticky='w', pady=(0, 10))

        options = ttk.Frame(export_frame)
        options.grid(row=5, column=1, sticky='w', pady=(0, 14))
        ttk.Checkbutton(options, text='包含 kNN 样本特征', variable=self.export_include_samples_var).pack(side=LEFT)
        ttk.Checkbutton(options, text='包含本地服务脚本', variable=self.package_server_var).pack(side=LEFT, padx=(18, 0))

        ttk.Button(
            export_frame,
            text='一键导出离线模型包',
            command=self.export_full_deploy_package,
            style='Primary.TButton',
            width=20,
        ).grid(row=6, column=1, sticky='w')
        export_frame.columnconfigure(1, weight=1)

        result_frame = ttk.LabelFrame(self.deploy_tab, text='导出结果', padding=10)
        result_frame.pack(fill=X)
        self.export_summary = ttk.Label(
            result_frame,
            text='状态：尚未导出。',
            justify='left',
            wraplength=1000,
        )
        self.export_summary.pack(fill=X, anchor='nw')

    def choose_export_config(self):
        path = filedialog.askopenfilename(
            title='选择 task_config.yaml',
            filetypes=[('YAML', '*.yaml *.yml'), ('All files', '*.*')],
        )
        if not path:
            return
        self.export_config_var.set(path)
        self.config_path = path
        if hasattr(self, 'config_var'):
            self.config_var.delete(0, END)
            self.config_var.insert(0, path)
        self.refresh_project_info()

    def use_current_export_project(self):
        if not self.config_path:
            messagebox.showwarning('缺少项目', '请先在“项目”页新建项目或打开配置。')
            return
        self.export_config_var.set(str(self.config_path))

    def choose_export_dir(self):
        path = filedialog.askdirectory(title='选择模型导出位置')
        if path:
            self.export_dir_var.set(path)

    def _default_deploy_paths(self, config_path, export_dir=None, export_name=None):
        config_path = Path(config_path).expanduser().resolve()
        cfg = load_yaml(config_path)
        task_name = cfg.get('task', {}).get('name') or config_path.parent.parent.name or 'ask2know_model'
        task_name = _safe_name(task_name)
        project_root = config_path.parent.parent if config_path.parent.name == 'configs' else config_path.parent
        deploy_dir = Path(export_dir).expanduser().resolve() if export_dir else project_root / 'outputs_deploy'
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        clean_export_name = str(export_name or '').strip()
        for suffix in ('.a2kmodel.json', '.json'):
            if clean_export_name.lower().endswith(suffix):
                clean_export_name = clean_export_name[:-len(suffix)]
                break
        base_name = _safe_name(clean_export_name) if clean_export_name else f'{task_name}_offline_model_{stamp}'
        model_path = deploy_dir / f'{base_name}.a2kmodel.json'
        package_dir = deploy_dir / f'{base_name}_offline_model_package'

        output_dir = Path(cfg.get('paths', {}).get('output_dir', project_root / 'outputs'))
        if not output_dir.is_absolute():
            output_dir = output_dir.resolve()
        cache_path = output_dir / 'prototype_model_cache.json'
        if not cache_path.exists():
            cache_path = None
        return cfg, model_path, package_dir, cache_path

    def _validation_status_path(self, config_path):
        config_path = Path(config_path).expanduser().resolve()
        cfg = load_yaml(config_path)
        project_root = config_path.parent.parent if config_path.parent.name == 'configs' else config_path.parent
        output_dir = Path(cfg.get('paths', {}).get('output_dir', project_root / 'outputs'))
        if not output_dir.is_absolute():
            output_dir = output_dir.resolve()
        return output_dir / 'validation_status.json'

    def _require_evaluated_before_export(self, config_path):
        status_path = self._validation_status_path(config_path)
        if not status_path.exists():
            messagebox.showwarning('需要先评估模型', '请先进入“评估模型”并完成评估，然后再导出模型。')
            return False
        return True

    def choose_export_cache(self):
        path = filedialog.askopenfilename(
            title='选择 prototype_model_cache.json',
            filetypes=[('JSON', '*.json'), ('All files', '*.*')],
        )
        if path:
            self.export_cache_var.set(path)

    def choose_export_model_output(self):
        path = filedialog.asksaveasfilename(
            title='保存 .a2kmodel.json',
            defaultextension='.json',
            filetypes=[('Ask2Know model', '*.a2kmodel.json'), ('JSON', '*.json'), ('All files', '*.*')],
        )
        if path:
            self.export_model_output_var.set(path)

    def validate_current_project(self):
        config_path = self.export_config_var.get().strip() or str(self.config_path or '')
        if not config_path:
            messagebox.showwarning('缺少项目', '请先在“项目”页新建项目或打开配置。')
            return
        self.export_config_var.set(config_path)

        def validate():
            cfg = load_yaml(config_path)
            output_dir = Path(cfg['paths']['output_dir'])
            active_session = self.session
            if active_session is not None and Path(active_session.config_path) == Path(config_path).expanduser().resolve():
                logs = list(getattr(active_session, 'logs', []) or [])
                source = 'current_session'
            else:
                log_path = output_dir / 'logs' / 'demo_log.json'
                logs = load_json(log_path) if log_path.exists() else []
                source = 'saved_log'
            report = evaluate_unknown_audit_logs(cfg, logs, output_dir)
            return report, output_dir / 'unlabeled_validation_report.json', source

        def done(result):
            report, report_path, source = result
            standard = report.get('validation_standard') or {}
            passed = bool(standard.get('passed', False))
            status_text = '验证通过' if passed else '验证未通过'
            self.validation_summary.configure(text=status_text)
            self.refresh_project_info()
            self.set_status(status_text + '。')

        self.run_worker('正在验证当前模型...', validate, done)

    def export_deploy_model(self):
        config_path = self.export_config_var.get().strip() or str(self.config_path or '')
        cache_path = self.export_cache_var.get().strip()
        include_samples = bool(self.export_include_samples_var.get())
        if not config_path:
            messagebox.showwarning('缺少项目', '请先选择项目配置。')
            return
        if not self._require_evaluated_before_export(config_path):
            return
        export_dir = self.export_dir_var.get().strip()
        export_name = self.export_name_var.get().strip()
        _, default_model_path, _, default_cache_path = self._default_deploy_paths(config_path, export_dir, export_name)
        output_path = self.export_model_output_var.get().strip() or str(default_model_path)
        cache_path = cache_path or (str(default_cache_path) if default_cache_path else '')

        def export():
            if cache_path:
                path, bundle = build_deployment_bundle_from_model_cache(
                    config_path,
                    cache_path,
                    output_path=output_path,
                    include_sample_features=include_samples,
                )
            else:
                path, bundle = build_deployment_bundle(
                    config_path,
                    output_path=output_path,
                    include_sample_features=include_samples,
                )
            return Path(path).resolve(), bundle

        def done(result):
            path, bundle = result
            self.package_model_var.set(str(path))
            if not self.package_output_var.get().strip():
                self.package_output_var.set(str(path.parent / f'{path.stem}_offline_model_package'))
            classes = bundle.get('classes') or []
            self.export_summary.configure(text=f'模型文件已导出：\n{path}\n类别数：{len(classes)}')
            self.set_status('模型文件已导出。')

        self.run_worker('正在导出模型文件...', export, done)

    def export_full_deploy_package(self):
        config_path = self.export_config_var.get().strip() or str(self.config_path or '')
        if not config_path:
            messagebox.showwarning('缺少项目', '请先在“项目”页新建项目或打开配置。')
            return
        if not self._require_evaluated_before_export(config_path):
            return
        self.export_config_var.set(config_path)
        include_samples = bool(self.export_include_samples_var.get())
        include_server = bool(self.package_server_var.get())
        export_dir = self.export_dir_var.get().strip()
        export_name = self.export_name_var.get().strip()

        def export_and_package():
            cfg, model_path, package_dir, cache_path = self._default_deploy_paths(config_path, export_dir, export_name)
            if cache_path:
                path, bundle = build_deployment_bundle_from_model_cache(
                    config_path,
                    cache_path,
                    output_path=model_path,
                    include_sample_features=include_samples,
                )
            else:
                path, bundle = build_deployment_bundle(
                    config_path,
                    output_path=model_path,
                    include_sample_features=include_samples,
                )
            package_path = build_package(path, package_dir, include_server=include_server)
            return Path(path).resolve(), Path(package_path).resolve(), bundle, cache_path

        def done(result):
            model_path, package_path, bundle, cache_path = result
            self.export_model_output_var.set(str(model_path))
            self.package_model_var.set(str(model_path))
            self.package_output_var.set(str(package_path))
            self.export_cache_var.set(str(cache_path) if cache_path else '')
            classes = bundle.get('classes') or []
            cache_text = '已使用训练缓存' if cache_path else '重新构建模型特征'
            self.export_summary.configure(
                text=(
                    f'状态：导出完成\n'
                    f'模型文件：{model_path}\n'
                    f'离线包目录：{package_path}\n'
                    f'类别数：{len(classes)}\n'
                    f'导出方式：{cache_text}'
                )
            )
            self.set_status('离线模型包导出完成。')

        self.run_worker('正在一键导出离线模型包...', export_and_package, done)

    def choose_package_model(self):
        path = filedialog.askopenfilename(
            title='选择 .a2kmodel.json',
            filetypes=[('Ask2Know model', '*.a2kmodel.json'), ('JSON', '*.json'), ('All files', '*.*')],
        )
        if path:
            self.package_model_var.set(path)

    def choose_package_output(self):
        parent = filedialog.askdirectory(title='选择离线包父目录')
        if not parent:
            return
        model_path = Path(self.package_model_var.get().strip() or self.export_model_output_var.get().strip() or 'ask2know_model.a2kmodel.json')
        output = Path(parent) / f'{model_path.stem}_offline_model_package'
        self.package_output_var.set(str(output))

    def package_deploy_model(self):
        model_path = self.package_model_var.get().strip()
        output_dir = self.package_output_var.get().strip()
        if not model_path or not output_dir:
            messagebox.showwarning('缺少路径', '请先选择模型文件和离线包输出目录。')
            return

        def package():
            return build_package(
                model_path,
                output_dir,
                include_server=bool(self.package_server_var.get()),
            )

        def done(result):
            self.export_summary.configure(text=f'Python 离线包已生成：\n{result}')
            self.set_status('Python 离线包已生成。')

        self.run_worker('正在生成 Python 离线包...', package, done)

    def render_image(self, path):
        self.photo = None
        if not path:
            self.image_label.configure(image='', text='图片预览')
            return
        try:
            image = Image.open(path)
            image.thumbnail((440, 540))
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

    def render_prediction_summary(self, rows, state):
        if not rows:
            self.prediction_text.configure(text='模型判断：暂无预测结果。')
            if hasattr(self, 'correct_button'):
                self.correct_button.configure(text='模型判断正确')
            return
        top = rows[0]
        second = rows[1] if len(rows) > 1 else None
        text = f"模型判断：{top.get('label', '')}    分数：{top.get('score', '')}"
        if second:
            text += f"\n第二候选：{second.get('label', '')}    分数：{second.get('score', '')}    Top2 差距：{state.get('gap', '')}"
        self.prediction_text.configure(text=text)
        if hasattr(self, 'correct_button'):
            label = top.get('label', '')
            self.correct_button.configure(text=f'确认：{label}' if label else '模型判断正确')

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
