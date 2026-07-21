import asyncio
import csv
import json
import os
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import (
    parse_qsl,
    urlencode,
    urlparse,
    urlunparse,
)

import pandas as pd
from agents import (
    Agent,
    Runner,
    function_tool,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
)
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import AsyncOpenAI
from playwright.sync_api import sync_playwright


# ============================================================
# 1. 基础配置
# ============================================================

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not DEEPSEEK_API_KEY:
    raise ValueError(
        "没有找到 DEEPSEEK_API_KEY。\n"
        "请检查项目根目录中的 .env 文件。"
    )

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = DATA_DIR / "wechat_results.csv"
LINKS_FILE = DATA_DIR / "links.csv"

MODEL_NAME = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
MAX_CONTENT_LENGTH = 15_000
ARTICLE_INTERVAL_SECONDS = 3

CSV_FIELDS = [
    "publish_time",
    "title",
    "account",
    "article_url",
    "category",
    "companies",
    "keywords",
    "summary",
    "importance",
    "reason",
    "relevance_score",
    "quality_score",
    "importance_score",
    "source_reliability_score",
    "originality_score",
    "source_type",
    "technology_route",
    "evidence_level",
    "is_promotional",
    "selection_reason",
]

set_tracing_disabled(True)


# ============================================================
# 2. 配置 DeepSeek
# ============================================================

deepseek_client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
)

deepseek_model = OpenAIChatCompletionsModel(
    model=MODEL_NAME,
    openai_client=deepseek_client,
)


# ============================================================
# 3. 通用辅助函数
# ============================================================

def parse_wechat_publish_time(
    publish_time_text: str,
) -> datetime | None:
    """
    将微信公众号发布时间转换为 datetime。
    """

    text = str(publish_time_text).strip()

    if not text:
        return None

    formats = [
        "%Y年%m月%d日 %H:%M",
        "%Y年%m月%d日",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d",
    ]

    for date_format in formats:
        try:
            return datetime.strptime(
                text,
                date_format,
            )
        except ValueError:
            continue

    return None


def is_within_last_days(
    publish_time_text: str,
    days: int = 7,
) -> bool:
    """
    按自然日期判断文章是否在最近指定天数内。

    例如今天是 7 月 21 日，
    days=7 时保留 7 月 15 日至 7 月 21 日。
    """

    publish_datetime = parse_wechat_publish_time(
        publish_time_text
    )

    if publish_datetime is None:
        return False

    today = datetime.now().date()

    earliest_date = today - timedelta(
        days=days - 1
    )

    publish_date = publish_datetime.date()

    return (
        earliest_date
        <= publish_date
        <= today
    )

def normalize_wechat_url(url: str) -> str:
    """
    标准化微信公众号文章链接，同时保留访问文章所需参数。

    支持：
    1. https://mp.weixin.qq.com/s/文章ID
    2. https://mp.weixin.qq.com/s?__biz=...&mid=...&idx=...&sn=...
    3. https://mp.weixin.qq.com/s?src=...&timestamp=...&ver=...&signature=...
    """

    cleaned = str(url).strip()
    parsed = urlparse(cleaned)

    if parsed.scheme not in {"http", "https"}:
        return cleaned

    hostname = (parsed.hostname or "").lower()

    if hostname != "mp.weixin.qq.com":
        return cleaned

    # 形式一：/s/文章ID
    if (
        parsed.path.startswith("/s/")
        and len(parsed.path) > 3
    ):
        normalized = parsed._replace(
            scheme="https",
            netloc="mp.weixin.qq.com",
            query="",
            fragment="",
        )

        return urlunparse(normalized).rstrip("/")

    # 形式二、三：/s?参数
    if parsed.path == "/s":
        query_items = parse_qsl(
            parsed.query,
            keep_blank_values=False,
        )

        allowed_keys = {
            "__biz",
            "mid",
            "idx",
            "sn",
            "chksm",
            "src",
            "timestamp",
            "ver",
            "signature",
            "new",
        }

        filtered_query = [
            (key, value)
            for key, value in query_items
            if key in allowed_keys
        ]

        normalized_query = urlencode(
            filtered_query
        )

        normalized = parsed._replace(
            scheme="https",
            netloc="mp.weixin.qq.com",
            query=normalized_query,
            fragment="",
        )

        return urlunparse(normalized)

    return cleaned


def is_valid_wechat_url(url: str) -> bool:
    """
    判断是否为可访问的微信公众号文章链接。
    """

    parsed = urlparse(
        str(url).strip()
    )

    hostname = (
        parsed.hostname or ""
    ).lower()

    if (
        parsed.scheme not in {"http", "https"}
        or hostname != "mp.weixin.qq.com"
    ):
        return False

    # 形式一：/s/文章ID
    if (
        parsed.path.startswith("/s/")
        and len(parsed.path) > 3
    ):
        return True

    if parsed.path != "/s":
        return False

    query_params = dict(
        parse_qsl(
            parsed.query,
            keep_blank_values=False,
        )
    )

    # 形式二：标准微信文章参数
    standard_keys = {
        "__biz",
        "mid",
        "idx",
        "sn",
    }

    if standard_keys.issubset(
        query_params.keys()
    ):
        return True

    # 形式三：搜狗签名链接
    signed_keys = {
        "src",
        "timestamp",
        "ver",
        "signature",
    }

    if signed_keys.issubset(
        query_params.keys()
    ):
        return True

    return False


def validate_score(value: int, field_name: str) -> int:
    """
    确保评分为 0 到 100 的整数。
    """
    try:
        score = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{field_name} 必须是整数，当前值为：{value}"
        ) from error

    if not 0 <= score <= 100:
        raise ValueError(
            f"{field_name} 必须在 0 到 100 之间，当前值为：{score}"
        )

    return score


def ensure_output_schema() -> None:
    """
    检查已有 CSV 表头是否与当前代码一致。

    如果旧 CSV 缺少评分字段，会明确报错，避免继续写入错位数据。
    """
    if not OUTPUT_FILE.exists():
        return

    if OUTPUT_FILE.stat().st_size == 0:
        OUTPUT_FILE.unlink()
        return

    with OUTPUT_FILE.open(
        mode="r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.reader(file)
        existing_header = next(reader, [])

    if existing_header != CSV_FIELDS:
        missing_fields = [
            field for field in CSV_FIELDS
            if field not in existing_header
        ]

        extra_fields = [
            field for field in existing_header
            if field not in CSV_FIELDS
        ]

        details = []

        if missing_fields:
            details.append(
                "缺少字段：" + "、".join(missing_fields)
            )

        if extra_fields:
            details.append(
                "多余字段：" + "、".join(extra_fields)
            )

        detail_text = "\n".join(details)

        raise ValueError(
            "现有 data/wechat_results.csv 的表头与当前代码不一致。\n"
            f"{detail_text}\n"
            "请先备份并删除旧 CSV，再重新运行程序。"
        )


def load_existing_urls() -> set[str]:
    """
    读取结果 CSV 中已保存的链接。
    """
    if not OUTPUT_FILE.exists():
        return set()

    ensure_output_schema()

    existing_urls: set[str] = set()

    with OUTPUT_FILE.open(
        mode="r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            url = normalize_wechat_url(
                row.get("article_url", "")
            )

            if url:
                existing_urls.add(url)

    return existing_urls


def article_already_saved(url: str) -> bool:
    """
    在调用浏览器和 DeepSeek 前检查文章是否已处理。
    """
    normalized_url = normalize_wechat_url(url)
    return normalized_url in load_existing_urls()


# ============================================================
# 4. 工具一：爬取微信公众号文章
# ============================================================

@function_tool
def scrape_wechat_article(url: str) -> str:
    """
    打开微信公众号文章并提取标题、公众号、发布时间、正文和链接。
    """

    print("\n[工具调用] 正在打开微信公众号文章……")
    print(f"[文章链接] {url}")

    normalized_url = normalize_wechat_url(url)

    if not is_valid_wechat_url(normalized_url):
        return json.dumps(
            {
                "success": False,
                "error": (
                    "链接格式不正确。请输入完整的微信公众号文章链接。"
                ),
            },
            ensure_ascii=False,
        )

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=False
            )

            context = browser.new_context(
                viewport={
                    "width": 1280,
                    "height": 900,
                },
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/150.0.0.0 Safari/537.36"
                ),
            )

            page = context.new_page()

            page.goto(
                normalized_url,
                wait_until="domcontentloaded",
                timeout=60_000,
            )

            page.wait_for_selector(
                "#js_content",
                timeout=30_000,
            )

            html = page.content()

            context.close()
            browser.close()

        soup = BeautifulSoup(
            html,
            "html.parser",
        )

        title_tag = soup.select_one("#activity-name")
        account_tag = soup.select_one("#js_name")
        publish_time_tag = (
            soup.select_one("#publish_time")
            or soup.select_one("em#publish_time")
        )
        content_tag = soup.select_one("#js_content")

        title = (
            title_tag.get_text(" ", strip=True)
            if title_tag
            else "未获取到标题"
        )

        account = (
            account_tag.get_text(" ", strip=True)
            if account_tag
            else "未获取到公众号"
        )

        publish_time = (
            publish_time_tag.get_text(" ", strip=True)
            if publish_time_tag
            else "未获取到发布时间"
        )

        content = (
            content_tag.get_text("\n", strip=True)
            if content_tag
            else ""
        )

        content_lines = [
            line.strip()
            for line in content.splitlines()
            if line.strip()
        ]

        content = "\n".join(content_lines)

        if not content:
            return json.dumps(
                {
                    "success": False,
                    "error": "网页打开成功，但没有提取到正文。",
                    "article_url": normalized_url,
                },
                ensure_ascii=False,
            )

        content = content[:MAX_CONTENT_LENGTH]

        article_data = {
            "success": True,
            "title": title,
            "account": account,
            "publish_time": publish_time,
            "article_url": normalized_url,
            "content": content,
        }

        print("[工具完成] 文章提取成功")
        print(f"[标题] {title}")
        print(f"[公众号] {account}")
        print(f"[发布时间] {publish_time}")
        print(f"[正文长度] {len(content)} 字符")

        return json.dumps(
            article_data,
            ensure_ascii=False,
        )

    except Exception as error:
        print(f"[工具失败] {error}")

        return json.dumps(
            {
                "success": False,
                "error": str(error),
                "article_url": normalized_url,
            },
            ensure_ascii=False,
        )


# ============================================================
# 5. 工具二：保存分析结果
# ============================================================

@function_tool
def save_article_analysis(
    title: str,
    account: str,
    publish_time: str,
    article_url: str,
    category: str,
    companies: str,
    keywords: str,
    summary: str,
    importance: str,
    reason: str,
    relevance_score: int,
    quality_score: int,
    importance_score: int,
    source_reliability_score: int,
    originality_score: int,
    source_type: str,
    technology_route: str,
    evidence_level: str,
    is_promotional: str,
    selection_reason: str,
) -> str:
    """
    将量子行业文章分析结果保存到 CSV。
    如果链接已存在，则不重复保存。
    """

    print("\n[工具调用] 正在保存分析结果……")

    try:
        ensure_output_schema()

        normalized_url = normalize_wechat_url(
            article_url
        )

        if normalized_url in load_existing_urls():
            message = (
                "保存状态：duplicate。"
                "该文章已存在于 CSV，本次未重复写入。"
            )

            print(f"[跳过保存] {message}")
            return message

        relevance_score = validate_score(
            relevance_score,
            "relevance_score",
        )
        quality_score = validate_score(
            quality_score,
            "quality_score",
        )
        importance_score = validate_score(
            importance_score,
            "importance_score",
        )
        source_reliability_score = validate_score(
            source_reliability_score,
            "source_reliability_score",
        )
        originality_score = validate_score(
            originality_score,
            "originality_score",
        )

        allowed_importance = {"高", "中", "低"}
        allowed_evidence = {"高", "中", "低"}
        allowed_promotional = {"是", "否"}

        if importance not in allowed_importance:
            raise ValueError(
                "importance 只能是：高、中、低。"
            )

        if evidence_level not in allowed_evidence:
            raise ValueError(
                "evidence_level 只能是：高、中、低。"
            )

        if is_promotional not in allowed_promotional:
            raise ValueError(
                "is_promotional 只能是：是、否。"
            )

        row = {
            "publish_time": str(publish_time).strip(),
            "title": str(title).strip(),
            "account": str(account).strip(),
            "article_url": normalized_url,
            "category": str(category).strip(),
            "companies": str(companies).strip(),
            "keywords": str(keywords).strip(),
            "summary": str(summary).strip(),
            "importance": importance,
            "reason": str(reason).strip(),
            "relevance_score": relevance_score,
            "quality_score": quality_score,
            "importance_score": importance_score,
            "source_reliability_score": source_reliability_score,
            "originality_score": originality_score,
            "source_type": str(source_type).strip(),
            "technology_route": str(technology_route).strip(),
            "evidence_level": evidence_level,
            "is_promotional": is_promotional,
            "selection_reason": str(selection_reason).strip(),
        }

        file_exists = OUTPUT_FILE.exists()

        with OUTPUT_FILE.open(
            mode="a",
            newline="",
            encoding="utf-8-sig",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=CSV_FIELDS,
                extrasaction="raise",
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)

        print("[评分]")
        print(f"  相关性：{relevance_score}")
        print(f"  文章质量：{quality_score}")
        print(f"  事件重要性：{importance_score}")
        print(f"  来源可靠性：{source_reliability_score}")
        print(f"  原创程度：{originality_score}")

        result = (
            "保存状态：saved。"
            f"本次成功新增一条记录，文件位置：{OUTPUT_FILE.resolve()}"
        )

        print(f"[工具完成] {result}")
        return result

    except Exception as error:
        error_message = f"保存状态：failed。保存失败：{error}"

        print(f"[工具失败] {error_message}")
        return error_message


# ============================================================
# 6. 创建文章分析 Agent
# ============================================================

wechat_agent = Agent(
    name="量子行业微信公众号资讯分析智能体",
    model=deepseek_model,

    instructions="""
你是一个量子行业微信公众号资讯采集与分析智能体。

你必须严格完成：
抓取文章 → 判断相关性 → 提取结构化信息 → 统一评分 → 保存 CSV。

【工具调用规则】

1. 每次任务必须先调用 scrape_wechat_article。
2. 如果 scrape_wechat_article 返回 success=false：
   - 说明失败原因；
   - 不得调用 save_article_analysis；
   - 不得编造文章信息。
3. 如果 success=true：
   - 阅读标题、公众号、发布时间和正文；
   - 进行分析；
   - 只调用一次 save_article_analysis。
4. 调用 save_article_analysis 后，无论返回 saved、duplicate 或 failed，
   都必须停止调用工具。
5. 不得为了确认保存状态重复调用保存工具。

【量子行业范围】

包括但不限于：
量子计算、量子通信、量子测量、量子精密测量、量子传感、
量子芯片、量子软件、量子算法、量子云平台、量子安全、
后量子密码、超导量子、离子阱、中性原子、光量子、
硅量子点、拓扑量子、量子退火、相干伊辛机。

如果文章与量子行业无关：
- category="非量子行业"
- importance="低"
- relevance_score 应低于 40
- reason 和 selection_reason 说明关联较弱
- 仍保存，便于后续审计和排除。

【category】

只能优先从以下类别选择一个：
融资、投资、并购、合作、合同订单、产品发布、技术研发、
科研进展、政策、政府项目、产业园区、市场动态、人才招聘、
会议活动、量子科普、非量子行业、其他。

【主体提取】

companies：
- 提取重要公司、科研机构、大学、投资机构或政府部门；
- 多个主体用中文顿号“、”分隔；
- 只保留与核心事件直接相关的主体；
- 没有明确主体时填写“无”；
- 不得编造。

【关键词】

keywords：
- 提取 3 到 6 个关键词；
- 优先包括技术路线、公司、融资轮次、产品、政策、合作类型、应用场景；
- 使用中文顿号“、”分隔。

【摘要】

summary：
- 中文，不超过 150 字；
- 回答“谁、发生了什么、涉及什么技术或产品、结果或影响是什么”；
- 不得添加原文没有的信息。

【重要程度】

importance 只能为“高”“中”“低”。

高：
国家级政策、重大技术突破、具有代表性的大型融资、
核心产品发布、大型合同、行业并购、重大商业化事件。

中：
一般融资、企业合作、产品更新、研究进展、区域项目、
商业落地案例。

低：
普通宣传、活动预告、常规会议、科普、重复报道、
信息量较少或与量子行业关联较弱。

reason：
用一句话解释重要程度。

【五项评分】

所有评分必须为 0 到 100 的整数，不能留空。

1. relevance_score：量子行业相关性
- 90-100：核心内容直接是量子技术、产品或产业事件
- 70-89：核心主体属于量子行业，但技术内容较少
- 40-69：部分涉及量子，但不是核心
- 0-39：关联较弱或无关

2. quality_score：信息质量
综合主体、金额、轮次、产品、技术、政策、合同、时间、
事实、数据和宣传程度。
- 85-100：多个具体且可核实事实
- 70-84：事件和主体明确，细节较完整
- 50-69：概括性报道
- 30-49：宣传性强、事实较少
- 0-29：信息极少或标题党

3. importance_score：行业事件重要性
- 90-100：国家政策、重大突破、重大并购、大型融资、核心产品、重大合同
- 70-89：重要融资、合作、落地、科研突破、区域政策
- 50-69：一般合作、更新、研究或市场拓展
- 30-49：普通活动、会议或宣传
- 0-29：行业影响很弱

4. source_reliability_score：来源可靠性
- 90-100：政府、权威政策、国家科研机构、大学官方、公司官方公告
- 75-89：知名行业媒体、投资机构、专业研究机构
- 55-74：普通商业媒体、综合媒体、转载平台
- 30-54：宣传稿、自媒体、来源不明确
- 0-29：来源缺失或明显不可靠

注意：
公众号名称本身不足以证明文章中的所有陈述均已被独立核实。
公司官方稿可作为原始来源，但若宣传性强，可降低 quality_score。

5. originality_score：原创程度
- 85-100：首发、独家、官方原始发布或有独家信息
- 65-84：有采访、原创整理或分析
- 40-64：主要整理已有信息
- 20-39：明显转载或高度重复
- 0-19：几乎无原创信息

【结构化字段】

source_type 优先选择：
政府、公司官方、科研机构、大学、投资机构、
行业媒体、综合媒体、自媒体、其他。

technology_route 优先选择：
超导量子、离子阱、中性原子、光量子、硅量子点、
拓扑量子、量子退火、相干伊辛机、量子软件、量子算法、
量子通信、量子测量、量子传感、后量子密码、量子综合、无法判断。

若涉及多个路线，用中文顿号“、”分隔。

evidence_level 只能为“高”“中”“低”：
- 高：有明确金额、政策、合同、参数、研究结果、官方公告或多个具体事实
- 中：主体和事件明确，但关键数据不完整
- 低：宣传、观点、活动介绍或缺少具体事实

is_promotional 只能为“是”或“否”。
大量愿景和宣传词、但缺少产品、合同、金额、技术成果或落地事实时，
通常填写“是”。

selection_reason：
用一句话说明是否值得进入量子行业日报或周报。
不要仅根据标题判断，要根据正文的事实密度、相关性和行业影响判断。

【保存要求】

调用 save_article_analysis 时必须完整传入：
title、account、publish_time、article_url、category、companies、
keywords、summary、importance、reason、relevance_score、
quality_score、importance_score、source_reliability_score、
originality_score、source_type、technology_route、evidence_level、
is_promotional、selection_reason。

【最终展示】

最终回答必须忠实展示：
- 标题
- 公众号
- 发布时间
- 分类
- 涉及主体
- 技术路线
- 关键词
- 摘要
- 重要程度
- 五项评分
- 是否宣传稿
- 入选理由
- 保存工具返回的真实状态

如果保存状态是 duplicate，只能说“此前已保存，本次已跳过”；
如果是 saved，只能说“本次新增成功”；
不得自行推测数据库状态。
""",

    tools=[
        scrape_wechat_article,
        save_article_analysis,
    ],
)


# ============================================================
# 7. 批量处理辅助函数
# ============================================================

def add_url_to_links_file(url: str) -> None:
    """
    将单篇输入的链接追加到 data/links.csv。
    如果链接已经存在，则不重复写入。
    """

    normalized_url = normalize_wechat_url(url)

    if LINKS_FILE.exists():
        df = pd.read_csv(LINKS_FILE)

        if "article_url" not in df.columns:
            raise ValueError(
                "links.csv 中必须包含 article_url 列。"
            )

        existing_urls = {
            normalize_wechat_url(str(item))
            for item in df["article_url"].dropna()
        }

        if normalized_url in existing_urls:
            print("[链接记录] 该链接已存在于 links.csv。")
            return

        new_row = pd.DataFrame(
            {
                "article_url": [normalized_url]
            }
        )

        new_row.to_csv(
            LINKS_FILE,
            mode="a",
            header=False,
            index=False,
            encoding="utf-8-sig",
        )

    else:
        pd.DataFrame(
            {
                "article_url": [normalized_url]
            }
        ).to_csv(
            LINKS_FILE,
            index=False,
            encoding="utf-8-sig",
        )

    print("[链接记录] 已写入 data/links.csv。")


def load_article_urls() -> list[str]:
    """
    从 data/links.csv 读取并去重微信公众号文章链接。
    """
    if not LINKS_FILE.exists():
        raise FileNotFoundError(
            f"没有找到链接文件：{LINKS_FILE.resolve()}\n"
            "请创建 data/links.csv，并设置第一列列名为 article_url。"
        )

    df = pd.read_csv(LINKS_FILE)

    if "article_url" not in df.columns:
        raise ValueError(
            "links.csv 中必须包含 article_url 列。"
        )

    urls: list[str] = []
    seen: set[str] = set()

    for raw_url in df["article_url"].dropna():
        normalized_url = normalize_wechat_url(
            str(raw_url)
        )

        if not normalized_url:
            continue

        if not is_valid_wechat_url(normalized_url):
            print(
                f"[忽略无效链接] {raw_url}"
            )
            continue

        if normalized_url in seen:
            continue

        seen.add(normalized_url)
        urls.append(normalized_url)

    return urls


@dataclass
class ProcessResult:
    status: str
    url: str
    message: str


async def process_one_article(
    url: str,
    current_number: int,
    total_number: int,
) -> ProcessResult:
    """
    处理一篇文章。

    在调用浏览器和 DeepSeek 前检查 CSV，已保存的链接直接跳过，
    避免重复爬取和消耗 API。
    """
    normalized_url = normalize_wechat_url(url)

    print("\n" + "=" * 60)
    print(
        f"正在处理第 {current_number}/{total_number} 篇文章"
    )
    print(f"链接：{normalized_url}")
    print("=" * 60)

    try:
        if article_already_saved(normalized_url):
            message = (
                "该文章已存在于结果 CSV，"
                "已在调用浏览器和 DeepSeek 前跳过。"
            )

            print(f"[预检查跳过] {message}")

            return ProcessResult(
                status="skipped",
                url=normalized_url,
                message=message,
            )

        task = f"""
请采集、分析并保存下面这篇微信公众号文章：

{normalized_url}
"""

        result = await Runner.run(
            starting_agent=wechat_agent,
            input=task,
            max_turns=8,
        )

        final_output = str(result.final_output)

        print("\nAgent 处理结果：")
        print(final_output)

        if "保存状态：saved" in final_output:
            status = "saved"
        elif "保存状态：duplicate" in final_output:
            status = "skipped"
        elif "保存状态：failed" in final_output:
            status = "failed"
        else:
            # 最终状态以结果 CSV 为准，避免模型最终措辞变化导致统计错误。
            status = (
                "saved"
                if article_already_saved(normalized_url)
                else "failed"
            )

        return ProcessResult(
            status=status,
            url=normalized_url,
            message=final_output,
        )

    except Exception as error:
        message = (
            f"{type(error).__name__}: {error}"
        )

        print("\n该文章处理失败：")
        print(message)

        return ProcessResult(
            status="failed",
            url=normalized_url,
            message=message,
        )


# ============================================================
# 8. 主程序
# ============================================================

async def main() -> None:
    print("=" * 60)
    print("量子行业微信公众号资讯分析 Agent")
    print(f"模型：{MODEL_NAME}")
    print("=" * 60)

    try:
        ensure_output_schema()
    except Exception as error:
        print("\nCSV 检查失败：")
        print(error)
        return

    print("\n请选择运行方式：")
    print("1. 输入单篇文章链接")
    print("2. 批量读取 data/links.csv")

    choice = input(
        "\n请输入 1 或 2："
    ).strip()

    if choice == "1":
        url = normalize_wechat_url(
            input(
                "\n请粘贴一篇微信公众号文章链接：\n"
            )
        )

        if not url:
            print("没有输入文章链接。")
            return

        if not is_valid_wechat_url(url):
            print(
                "链接无效。请输入完整的微信公众号文章链接。"
            )
            return

        add_url_to_links_file(url)

        result = await process_one_article(
            url=url,
            current_number=1,
            total_number=1,
        )

        print("\n" + "=" * 60)
        print("本次运行结束")
        print("=" * 60)
        print(f"状态：{result.status}")
        print(f"结果文件：{OUTPUT_FILE.resolve()}")

    elif choice == "2":
        try:
            urls = load_article_urls()
        except Exception as error:
            print("\n读取 links.csv 失败：")
            print(error)
            return

        if not urls:
            print("links.csv 中没有有效的微信公众号文章链接。")
            return

        print(
            f"\n共读取到 {len(urls)} 个不重复的有效链接。"
        )

        results: list[ProcessResult] = []

        for index, url in enumerate(
            urls,
            start=1,
        ):
            result = await process_one_article(
                url=url,
                current_number=index,
                total_number=len(urls),
            )

            results.append(result)

            if index < len(urls):
                await asyncio.sleep(
                    ARTICLE_INTERVAL_SECONDS
                )

        saved_count = sum(
            result.status == "saved"
            for result in results
        )
        skipped_count = sum(
            result.status == "skipped"
            for result in results
        )
        failed_count = sum(
            result.status == "failed"
            for result in results
        )

        print("\n" + "=" * 60)
        print("批量处理结束")
        print("=" * 60)
        print(f"有效链接总数：{len(urls)}")
        print(f"新增保存：{saved_count}")
        print(f"重复跳过：{skipped_count}")
        print(f"处理失败：{failed_count}")
        print(f"结果文件：{OUTPUT_FILE.resolve()}")

        if failed_count:
            print("\n失败链接：")

            for result in results:
                if result.status == "failed":
                    print(f"- {result.url}")
                    print(f"  {result.message}")

    else:
        print("输入错误，只能输入 1 或 2。")


if __name__ == "__main__":
    asyncio.run(main())
