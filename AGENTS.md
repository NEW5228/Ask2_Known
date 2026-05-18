# Repository Guidelines

## Project Structure & Module Organization

Ask2Know is a Python package for low-sample active teaching. Core code lives in `ask2know/`, split by responsibility: `data/` loads datasets, `features/` extracts image signals, `inference/` scores prototypes and uncertainty, `questions/` generates/selects prompts, `learning/` updates weights, `sample_pool/` manages accepted/rejected samples, and `experience/` records pairwise feedback. Top-level entry points are `run_demo.py` for interactive learning and `train.py` as a placeholder runner. Utility scripts live in `scripts/`, reusable configs in `configs/`, and design/user documentation in `docs/`.

## Build, Test, and Development Commands

Create and activate a Python 3.9+ environment before installing dependencies.

```bash
pip install -r requirements.txt
pip install -e .
python scripts/init_task.py --name fruit_test --classes apple banana --output ./work
python run_demo.py --config ./work/fruit_test/configs/task_config.yaml
python run_demo.py --config ./work/fruit_test/configs/task_config.yaml --preview
python -m build
```

`pip install -e .` installs the local package for development. `init_task.py` creates a task folder with dataset, metadata, and config files. `run_demo.py` runs the main workflow; `--preview` opens images during review. `python -m build` builds package artifacts if the `build` package is installed.

## Coding Style & Naming Conventions

Use standard Python style with 4-space indentation, `snake_case` functions and variables, and `PascalCase` classes. Prefer `pathlib.Path` for filesystem paths and keep JSON/YAML access through helpers in `ask2know/utils/io_utils.py` where practical. Keep modules focused on one domain concept and avoid adding heavy ML dependencies; this prototype currently relies on OpenCV, NumPy, and PyYAML. Preserve UTF-8 text handling because docs, prompts, and generated task files include Chinese copy.

## 0.4.0 Architecture Direction

Ask2Know should evolve as an embedding + similarity + user feedback pipeline without losing active teaching. Treat CLIP/DINO/ResNet/MobileNet-style embeddings as optional adapters and internal scoring signals, not replacements for the explainable concept layer. Prototype similarity, k-NN nearest-sample evidence, concept prototypes, confidence/uncertainty, active questions, pairwise experience, and online user feedback should remain coordinated parts of the same loop. User-facing questions should prefer explainable features such as color, shape, texture, surface, part, text, sign, and quality rather than asking users to reason about raw embeddings.

## Testing Guidelines

No committed test suite is present yet. Add new tests under `tests/` using `pytest` when changing behavior, especially for dataset loading, file normalization, scoring, and feedback updates. Name files `test_<module>.py` and test functions `test_<behavior>()`. Prefer temporary directories for sample task projects instead of writing into `configs/` or repository data paths.

## Commit & Pull Request Guidelines

Git history currently uses short, imperative summaries such as `Add project description for Ask2Know`. Follow that style: start with a verb, keep the subject concise, and mention the affected area when helpful. Pull requests should include a clear summary, manual test commands run, any config/schema changes, and screenshots or sample terminal output when changing interactive image workflows.

## Security & Configuration Tips

Do not commit local task datasets, generated outputs, virtual environments, or private image collections. Keep project-specific paths in task config files, and use relative paths in examples when possible so demos work across machines.

## Local File Safety

Do not bulk-delete files or directories. Do not use `del /s`, `rd /s`, `rmdir /s`, `Remove-Item -Recurse`, or `rm -rf`. If deletion is required, delete only one explicit file path at a time, for example `Remove-Item "C:\path\to\file.txt"`. If a task requires batch deletion, stop and ask the user to handle it manually.
