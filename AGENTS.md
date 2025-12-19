# Repository Guidelines

## 项目结构与模块组织
- 代码集中在仓库根目录的 Python 模块：`analyzer.py`（入口示例）、`framing_scorer.py`、`relative_framing.py`、`bias_teacher.py`、`text_processor.py` 与工具集 `utils.py`、配置 `config.py`。自动化脚本在 `setup_autodl.py` 与 `setup_autodl.sh`，依赖清单为 `requirements.txt` 与 `requirements_autodl.txt`。
- 建议新增 `tests/` 存放用例，`data/` 用于样本输入输出（必要时在 `.gitignore` 里排除大文件或敏感数据）。

## 环境、构建与开发命令
- 创建与启用虚拟环境：`python -m venv .venv && source .venv/bin/activate`。
- 安装依赖：`pip install -r requirements.txt`（自动 DL 场景用 `requirements_autodl.txt`）。如需国内源，自行在 pip 命令追加 `-i`。
- 运行分析脚本示例：`python analyzer.py --help`；自动化环境初始化：`python setup_autodl.py` 或 `bash setup_autodl.sh`（确认路径与权限）。
- 代码质量自检：`python -m compileall .` 快速发现语法错误。

## 代码风格与命名约定
- 遵循 PEP 8，四空格缩进；函数/变量用 `snake_case`，类用 `PascalCase`，常量全大写。
- 使用类型标注与文档字符串描述输入输出；仅保留必要注释，避免重复描述显而易见的逻辑。
- 保持模块单一职责，避免重复逻辑，抽取通用处理放入 `utils.py`；新增配置集中于 `config.py`。

## 测试指南
- 当前无内置测试，建议使用 `pytest`：在 `tests/` 下按 `test_*.py` 命名，覆盖核心打分逻辑（`framing_scorer.py`）、文本预处理（`text_processor.py`）和偏差调整（`bias_teacher.py`）。
- 推荐本地运行 `pytest -q`，新增功能需提供边界与负例；如引入外部资源，使用可复现的固定样本或 mock。

## 提交与 Pull Request 规范
- 提交信息使用祈使句简述变化，例如 `Add framing scorer thresholds`；关联 Issue 时追加 `#123`。
- PR 需包含：变更摘要、测试结果（命令与输出简述）、风险或兼容性说明、必要时的前后对比（日志或样例）。
- 禁止提交密钥或未清理的临时数据；变更依赖请同步更新相关 `requirements` 文件并说明原因。

## 配置与安全提示
- 不要在仓库中硬编码凭据；运行时通过环境变量或本地未提交的配置文件注入，必要时在 `.gitignore` 补充规则。
- 自动化脚本可能写入本地路径或下载模型/数据，执行前确认目标目录和磁盘配额，避免覆盖重要文件。
