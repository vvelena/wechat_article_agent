# WeChat Agent

一个用于搜索、抓取和分析微信公众号文章及普通网页的命令行 Agent。程序通过自然语言接收任务，将分析结果保存为 CSV，并可根据当天数据生成 Word 日报。

## 主要功能

- 按关键词搜索微信公众号文章
- 按最近若干天筛选文章，并对链接去重
- 抓取、分析微信公众号文章或普通网页
- 批量处理 `data/links.csv` 中的链接
- 将分析结果保存为结构化 CSV
- 清空指定结果表或全部结果表
- 根据当天收集的数据生成 Word 日报

## 环境要求

- Windows 10/11
- Python 3.10 或更高版本，推荐 Python 3.11
- 可访问 DeepSeek API
- 可访问目标网页及搜狗微信搜索

## 项目结构

```text
wechat_agent/
├─ app/
│  ├─ core/                 # 路径、CSV 管理和统一结果结构
│  ├─ report/               # 日报生成逻辑及 Word 模板
│  ├─ web/                  # 普通网页处理
│  └─ wechat/               # 微信文章搜索、抓取和分析
├─ data/
│  ├─ links.csv             # 等待批量处理的微信文章链接
│  ├─ web_results.csv       # 普通网页分析结果
│  └─ wechat_results.csv    # 微信文章分析结果（运行后生成）
├─ reports/                 # 生成的日报
├─ scripts/                 # 数据维护脚本
├─ tests/                   # 自动化测试
├─ main.py                  # 程序入口
└─ requirements.txt         # Python 依赖
```

请保留 `app/report/templates/daily_report_template.docx`，生成 Word 日报时需要使用该模板。

## 安装

以下命令均在项目根目录执行。

### 1. 创建虚拟环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

如果 PowerShell 禁止运行激活脚本，可先在当前窗口执行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

### 2. 安装 Python 依赖

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. 安装 Playwright 浏览器

```powershell
python -m playwright install chromium
```

## 配置 API Key

在项目根目录新建 `.env` 文件：

```env
DEEPSEEK_API_KEY=替换为你的DeepSeek_API_Key
DEEPSEEK_MODEL=deepseek-v4-flash
```

`DEEPSEEK_MODEL` 可以省略，程序默认使用 `deepseek-v4-flash`。

> `.env` 中含有 API 密钥，已被 `.gitignore` 忽略。请勿将自己的 `.env` 上传到代码仓库或直接发给其他人。同事应使用自己的密钥。

## 启动

```powershell
python main.py
```

启动后直接输入自然语言任务，例如：

```text
搜索最近7天量子计算融资相关的微信公众号文章，最多5篇并分析保存
```

```text
搜索 IBM 量子计算最新进展的普通网页，最多3篇并总结
```

```text
分析这个网页：https://example.com/article
```

```text
处理 data/links.csv 中的全部微信文章
```

```text
将今天收集的数据整理成Word日报
```

输入 `exit`、`quit` 或 `退出` 可结束程序。

## 批量处理链接

需要批量分析微信文章时，确保 `data/links.csv` 的第一列列名为 `article_url`，并将链接逐行写入该列。例如：

```csv
article_url
https://mp.weixin.qq.com/s/example1
https://mp.weixin.qq.com/s/example2
```

程序搜索到的新链接也会自动加入该文件。

随后在程序中输入：

```text
处理 data/links.csv 中的全部微信文章
```

## 输出文件

- 微信文章结果：`data/wechat_results.csv`
- 普通网页结果：`data/web_results.csv`
- Word 日报：`reports/`

CSV 文件可能包含收集到的文章内容和业务数据。在把项目交给其他人之前，请确认这些数据是否允许共享；如不需要共享，可以仅保留 CSV 表头或删除运行产生的数据文件。

## 数据维护

清理微信结果中的重复记录：

```powershell
python scripts/cleanup_wechat_duplicates.py
```

也可以在主程序中输入自然语言命令：

```text
清空微信结果表的数据
```

或：

```text
清空全部表格数据
```

清空操作不可撤销，执行前请备份需要保留的 CSV 文件。

## 运行测试

项目测试使用 Python 内置的 `unittest`：

```powershell
python -m unittest discover -s tests -v
```

## 常见问题

### 提示没有找到 `DEEPSEEK_API_KEY`

确认项目根目录存在 `.env`，并且内容格式如下：

```env
DEEPSEEK_API_KEY=你的实际密钥
```

等号两边不要添加多余空格，也不要把值写成示例文字。

### 提示找不到 Chromium 或浏览器可执行文件

重新执行：

```powershell
python -m playwright install chromium
```

### 搜索或抓取失败

可能原因包括网络不可达、网页限制自动访问、链接已失效、页面需要验证或目标站点结构发生变化。可以稍后重试，或将有效文章链接加入 `data/links.csv` 后批量处理。

### 无法生成 Word 日报

请检查：

- `app/report/templates/daily_report_template.docx` 是否存在
- `data/` 中是否有当天的有效分析结果
- `reports/` 目录是否可写
- `python-docx` 是否已正确安装

## 交付给他人时的注意事项

压缩项目时建议包含源代码、模板、`requirements.txt` 和本 README，不要包含以下内容：

```text
.env
.git/
.venv/
.uv-cache/
__pycache__/
*.pyc
```
