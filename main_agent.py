import asyncio
import csv
import json
import os
from pathlib import Path
from urllib.parse import quote_plus, urlparse

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

import search_playwright as search_module
import wechat_agent as article_module


# ============================================================
# 1. 基础配置
# ============================================================

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL_NAME = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

if not DEEPSEEK_API_KEY:
    raise ValueError(
        "没有找到 DEEPSEEK_API_KEY，请检查项目根目录中的 .env 文件。"
    )

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

WEB_RESULTS_FILE = DATA_DIR / "web_results.csv"

set_tracing_disabled(True)

deepseek_client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
)

deepseek_model = OpenAIChatCompletionsModel(
    model=MODEL_NAME,
    openai_client=deepseek_client,
)


# ============================================================
# 2. 微信公众号搜索工具
# ============================================================

def build_wechat_search_queries(keyword: str) -> list[str]:
    """
    根据用户主题生成由精确到宽泛的搜索词。
    避免只搜索一个过窄的完整短语。
    """

    cleaned = " ".join(
        str(keyword).strip().split()
    )

    if not cleaned:
        return []

    queries = [cleaned]

    replacements = {
        "量子计算 融资": [
            "量子 融资",
            "量子科技 融资",
            "量子芯片 融资",
            "量子计算 投资",
        ],
        "量子计算融资": [
            "量子计算 融资",
            "量子 融资",
            "量子科技 融资",
            "量子芯片 融资",
            "量子计算 投资",
        ],
        "量子计算 合作": [
            "量子 合作",
            "量子科技 合作",
            "量子计算 战略合作",
        ],
        "量子计算合作": [
            "量子计算 合作",
            "量子 合作",
            "量子科技 合作",
        ],
        "量子计算 政策": [
            "量子科技 政策",
            "量子 政策",
            "量子产业 政策",
        ],
        "量子计算政策": [
            "量子计算 政策",
            "量子科技 政策",
            "量子产业 政策",
        ],
    }

    queries.extend(
        replacements.get(cleaned, [])
    )

    # 对“主题 + 事件”类命令做通用拆分。
    event_words = [
        "融资",
        "投资",
        "并购",
        "合作",
        "合同",
        "订单",
        "中标",
        "政策",
        "突破",
        "产品",
        "发布",
        "商业化",
    ]

    for event_word in event_words:
        if event_word not in cleaned:
            continue

        subject = cleaned.replace(
            event_word,
            " ",
        )
        subject = " ".join(
            subject.split()
        )

        if subject:
            queries.extend(
                [
                    f"{subject} {event_word}",
                    f"量子 {event_word}",
                    f"量子科技 {event_word}",
                ]
            )

    # 保留顺序去重。
    return list(
        dict.fromkeys(
            query
            for query in queries
            if query.strip()
        )
    )


@function_tool
def search_wechat_articles(
    keyword: str,
    days: int = 7,
    max_results: int = 5,
) -> str:
    """
    搜索微信公众号文章，并把新链接加入 data/links.csv。

    会自动扩展多个相关搜索词，再合并去重。
    """

    if not keyword.strip():
        return json.dumps(
            {
                "success": False,
                "error": "搜索关键词不能为空。",
            },
            ensure_ascii=False,
        )

    days = max(1, min(int(days), 365))
    max_results = max(1, min(int(max_results), 20))

    search_queries = build_wechat_search_queries(
        keyword
    )

    print("\n[总控工具] 搜索微信公众号文章")
    print(f"[用户主题] {keyword}")
    print(f"[搜索词] {'、'.join(search_queries)}")
    print(f"[时间范围] 最近 {days} 个自然日")
    print(f"[最多保留] {max_results} 篇")

    search_module.SEARCH_DAYS = days

    existing_links = (
        search_module.load_existing_links()
    )

    all_found_links = []
    query_details = []

    for search_query in search_queries:
        if len(all_found_links) >= max_results:
            break

        remaining = (
            max_results
            - len(all_found_links)
        )

        try:
            results = (
                search_module
                .search_sogou_with_playwright(
                    query=search_query,
                    max_results=remaining,
                )
            )
        except Exception as error:
            query_details.append(
                {
                    "query": search_query,
                    "success": False,
                    "error": str(error),
                    "found_count": 0,
                }
            )
            continue

        query_details.append(
            {
                "query": search_query,
                "success": True,
                "found_count": len(results),
            }
        )

        for url in results:
            if url in all_found_links:
                continue

            all_found_links.append(url)

            if len(all_found_links) >= max_results:
                break

    new_links = [
        url
        for url in all_found_links
        if url not in existing_links
    ]

    search_module.append_new_links(
        new_links
    )

    if all_found_links and not new_links:
        result_status = "found_but_all_duplicate"
    elif new_links:
        result_status = "found_new"
    else:
        result_status = "not_found"

    return json.dumps(
        {
            "success": True,
            "status": result_status,
            "keyword": keyword,
            "search_queries": search_queries,
            "days": days,
            "found_count": len(all_found_links),
            "new_count": len(new_links),
            "duplicate_count": (
                len(all_found_links)
                - len(new_links)
            ),
            "found_links": all_found_links,
            "new_links": new_links,
            "query_details": query_details,
            "links_file": str(
                search_module
                .LINKS_FILE
                .resolve()
            ),
        },
        ensure_ascii=False,
    )


# ============================================================
# 3. 微信文章分析工具
# ============================================================

@function_tool
async def analyze_wechat_articles(
    urls: list[str],
) -> str:
    """
    批量爬取、分析并保存微信公众号文章。

    参数：
        urls：微信公众号文章链接列表。
    """

    if not urls:
        return json.dumps(
            {
                "success": False,
                "error": "没有提供文章链接。",
            },
            ensure_ascii=False,
        )

    unique_urls = []
    seen = set()

    for raw_url in urls:
        normalized = article_module.normalize_wechat_url(
            str(raw_url)
        )

        if not article_module.is_valid_wechat_url(
            normalized
        ):
            continue

        if normalized in seen:
            continue

        seen.add(normalized)
        unique_urls.append(normalized)

    if not unique_urls:
        return json.dumps(
            {
                "success": False,
                "error": "没有有效的微信公众号文章链接。",
            },
            ensure_ascii=False,
        )

    saved = 0
    skipped = 0
    failed = 0
    details = []

    for index, url in enumerate(
        unique_urls,
        start=1,
    ):
        existed_before = article_module.article_already_saved(
            url
        )

        try:
            result = await article_module.process_one_article(
                url=url,
                current_number=index,
                total_number=len(unique_urls),
            )

            existed_after = article_module.article_already_saved(
                url
            )

            if existed_before:
                status = "skipped"
                skipped += 1
            elif existed_after:
                status = "saved"
                saved += 1
            else:
                status = "failed"
                failed += 1

            details.append(
                {
                    "url": url,
                    "status": status,
                    "message": (
                        getattr(result, "message", None)
                        or str(result)
                    ),
                }
            )

        except Exception as error:
            failed += 1
            details.append(
                {
                    "url": url,
                    "status": "failed",
                    "message": (
                        f"{type(error).__name__}: {error}"
                    ),
                }
            )

        if index < len(unique_urls):
            await asyncio.sleep(
                article_module.ARTICLE_INTERVAL_SECONDS
            )

    return json.dumps(
        {
            "success": failed < len(unique_urls),
            "total": len(unique_urls),
            "saved": saved,
            "skipped": skipped,
            "failed": failed,
            "details": details,
            "result_file": str(
                article_module.OUTPUT_FILE.resolve()
            ),
        },
        ensure_ascii=False,
    )


@function_tool
async def analyze_links_file() -> str:
    """
    读取 data/links.csv 中的全部链接，并批量分析。
    """

    try:
        urls = article_module.load_article_urls()
    except Exception as error:
        return json.dumps(
            {
                "success": False,
                "error": str(error),
            },
            ensure_ascii=False,
        )

    return await analyze_wechat_articles.on_invoke_tool(
        None,
        json.dumps(
            {
                "urls": urls,
            },
            ensure_ascii=False,
        ),
    )


# ============================================================
# 4. 普通网页搜索工具
# ============================================================

@function_tool
def search_web_pages(
    keyword: str,
    max_results: int = 5,
) -> str:
    """
    使用浏览器搜索普通网页，不使用搜索 API。
    """

    if not keyword.strip():
        return json.dumps(
            {
                "success": False,
                "error": "搜索关键词不能为空。",
            },
            ensure_ascii=False,
        )

    max_results = max(1, min(int(max_results), 20))
    results = []

    search_url = (
        "https://www.bing.com/search?q="
        + quote_plus(keyword.strip())
    )

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=False
        )

        context = browser.new_context(
            viewport={
                "width": 1280,
                "height": 900,
            }
        )

        page = context.new_page()

        try:
            page.goto(
                search_url,
                wait_until="domcontentloaded",
                timeout=60_000,
            )

            page.wait_for_timeout(3000)

            links = page.locator(
                "li.b_algo h2 a"
            )

            count = min(
                links.count(),
                max_results,
            )

            for index in range(count):
                link = links.nth(index)

                title = link.inner_text().strip()
                url = (
                    link.get_attribute("href")
                    or ""
                ).strip()

                if not url.startswith(
                    ("http://", "https://")
                ):
                    continue

                results.append(
                    {
                        "title": title,
                        "url": url,
                    }
                )

        finally:
            context.close()
            browser.close()

    return json.dumps(
        {
            "success": True,
            "keyword": keyword,
            "count": len(results),
            "results": results,
        },
        ensure_ascii=False,
    )


# ============================================================
# 5. 普通网页抓取和保存工具
# ============================================================

@function_tool
def scrape_webpage(url: str) -> str:
    """
    打开普通网页，提取标题和正文。
    """

    parsed = urlparse(url.strip())

    if parsed.scheme not in {"http", "https"}:
        return json.dumps(
            {
                "success": False,
                "error": "网页地址必须以 http:// 或 https:// 开头。",
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
                }
            )

            page = context.new_page()

            page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=60_000,
            )

            page.wait_for_timeout(2500)

            html = page.content()
            final_url = page.url

            context.close()
            browser.close()

        soup = BeautifulSoup(
            html,
            "html.parser",
        )

        for tag in soup(
            [
                "script",
                "style",
                "noscript",
                "svg",
                "nav",
                "footer",
            ]
        ):
            tag.decompose()

        title = (
            soup.title.get_text(
                " ",
                strip=True,
            )
            if soup.title
            else ""
        )

        article_tag = (
            soup.select_one("article")
            or soup.select_one("main")
            or soup.body
        )

        content = (
            article_tag.get_text(
                "\n",
                strip=True,
            )
            if article_tag
            else ""
        )

        lines = [
            line.strip()
            for line in content.splitlines()
            if line.strip()
        ]

        content = "\n".join(lines)[:15000]

        if not content:
            return json.dumps(
                {
                    "success": False,
                    "error": "网页打开成功，但没有提取到正文。",
                    "url": final_url,
                },
                ensure_ascii=False,
            )

        return json.dumps(
            {
                "success": True,
                "title": title,
                "url": final_url,
                "content": content,
            },
            ensure_ascii=False,
        )

    except Exception as error:
        return json.dumps(
            {
                "success": False,
                "error": str(error),
                "url": url,
            },
            ensure_ascii=False,
        )


@function_tool
def save_webpage_analysis(
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
    将普通网页按照与微信公众号文章一致的完整结构保存。
    """

    fields = [
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

    article_url = article_url.strip()

    if WEB_RESULTS_FILE.exists():
        with WEB_RESULTS_FILE.open(
            mode="r",
            newline="",
            encoding="utf-8-sig",
        ) as file:
            for row in csv.DictReader(file):
                if (
                    row.get("article_url", "").strip()
                    == article_url
                ):
                    return (
                        "保存状态：duplicate。"
                        "此前已保存，本次已跳过。"
                    )

    def clamp_score(value: int) -> int:
        return max(
            0,
            min(int(value), 100),
        )

    if importance not in {"高", "中", "低"}:
        raise ValueError(
            "importance 只能是：高、中、低。"
        )

    if evidence_level not in {"高", "中", "低"}:
        raise ValueError(
            "evidence_level 只能是：高、中、低。"
        )

    if is_promotional not in {"是", "否"}:
        raise ValueError(
            "is_promotional 只能是：是、否。"
        )

    row = {
        "publish_time": str(publish_time).strip(),
        "title": str(title).strip(),
        "account": str(account).strip(),
        "article_url": article_url,
        "category": str(category).strip(),
        "companies": str(companies).strip(),
        "keywords": str(keywords).strip(),
        "summary": str(summary).strip(),
        "importance": importance,
        "reason": str(reason).strip(),
        "relevance_score": clamp_score(
            relevance_score
        ),
        "quality_score": clamp_score(
            quality_score
        ),
        "importance_score": clamp_score(
            importance_score
        ),
        "source_reliability_score": clamp_score(
            source_reliability_score
        ),
        "originality_score": clamp_score(
            originality_score
        ),
        "source_type": str(source_type).strip(),
        "technology_route": str(
            technology_route
        ).strip(),
        "evidence_level": evidence_level,
        "is_promotional": is_promotional,
        "selection_reason": str(
            selection_reason
        ).strip(),
    }

    file_exists = WEB_RESULTS_FILE.exists()

    with WEB_RESULTS_FILE.open(
        mode="a",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

    return (
        "保存状态：saved。"
        f"本次新增成功，文件位置：{WEB_RESULTS_FILE.resolve()}"
    )


# ============================================================
# 6. 总控 Agent
# ============================================================

main_agent = Agent(
    name="资讯搜索抓取总控智能体",
    model=deepseek_model,

    instructions="""
你是资讯搜索、抓取、分析和保存的总控智能体。

以下规则是全局最高优先级规则。
无论用户输入的是微信公众号文章、普通网页、浏览器搜索结果、
公司官网、新闻页面、单个链接、批量链接或自然语言搜索命令，
凡是进入“内容分析”阶段，都必须遵守同一套完整规则。
不得因为来源不同而简化字段、评分或保存步骤。

【一、全局执行顺序】

任何需要分析和保存的任务，都必须严格执行：

搜索或接收链接
→ 调用对应抓取工具
→ 检查抓取是否成功
→ 阅读正文
→ 判断量子行业相关性
→ 提取完整结构化信息
→ 完成五项评分
→ 调用一次对应保存工具
→ 忠实展示保存状态。

搜索本身只负责找候选链接。
不得仅根据搜索标题、摘要片段或搜索结果页直接评分。

【二、抓取工具规则】

1. 微信公众号链接：
   - 必须使用微信专用分析流程；
   - analyze_wechat_articles 内部会逐篇调用 wechat_agent；
   - wechat_agent 的工具调用和保存结果为最终依据。

2. 普通网页链接：
   - 必须先调用 scrape_webpage；
   - 如果 success=false：
     * 说明失败原因；
     * 不得调用 save_webpage_analysis；
     * 不得编造网页内容。
   - 如果 success=true：
     * 阅读标题、网页地址和完整正文；
     * 完成全部结构化分析；
     * 只调用一次 save_webpage_analysis。
   - 保存后无论返回 saved、duplicate 或 failed，
     都必须停止调用保存工具。

3. 不得绕过登录、验证码、付费墙或访问限制。

【三、量子行业范围】

包括但不限于：
量子计算、量子通信、量子测量、量子精密测量、量子传感、
量子芯片、量子软件、量子算法、量子云平台、量子安全、
后量子密码、超导量子、离子阱、中性原子、光量子、
硅量子点、拓扑量子、量子退火、相干伊辛机。

如果内容与量子行业无关：
- category="非量子行业"
- importance="低"
- relevance_score 必须低于40
- reason 和 selection_reason 必须说明关联较弱
- 用户要求保存时仍保存，便于审计和排除。

【四、category】

只能优先从以下类别选择一个：
融资、投资、并购、合作、合同订单、产品发布、技术研发、
科研进展、政策、政府项目、产业园区、市场动态、人才招聘、
会议活动、量子科普、非量子行业、其他。

【五、主体提取】

companies：
- 提取重要公司、科研机构、大学、投资机构或政府部门；
- 多个主体使用中文顿号“、”分隔；
- 只保留与核心事件直接相关的主体；
- 没有明确主体时填写“无”；
- 不得编造。

account：
- 微信文章填写公众号名称；
- 普通网页填写网站、机构、媒体或发布主体名称；
- 无法判断时填写“未获取到来源”。

publish_time：
- 有明确发布时间时填写原文时间；
- 无法获取时填写“未获取到发布时间”；
- 不得猜测。

【六、关键词】

keywords：
- 提取3到6个关键词；
- 优先包含技术路线、公司、融资轮次、产品、政策、
  合作类型、应用场景；
- 使用中文顿号“、”分隔。

【七、摘要】

summary：
- 中文，不超过150字；
- 回答“谁、发生了什么、涉及什么技术或产品、
  结果或影响是什么”；
- 不得添加正文没有的信息。

【八、重要程度】

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

【九、五项评分】

所有评分必须为0到100的整数，不能留空。

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
- 90-100：国家政策、重大突破、重大并购、大型融资、
  核心产品、重大合同
- 70-89：重要融资、合作、落地、科研突破、区域政策
- 50-69：一般合作、更新、研究或市场拓展
- 30-49：普通活动、会议或宣传
- 0-29：行业影响很弱

4. source_reliability_score：来源可靠性
- 90-100：政府、权威政策、国家科研机构、大学官方、
  公司官方公告
- 75-89：知名行业媒体、投资机构、专业研究机构
- 55-74：普通商业媒体、综合媒体、转载平台
- 30-54：宣传稿、自媒体、来源不明确
- 0-29：来源缺失或明显不可靠

注意：
来源名称本身不足以证明正文陈述已被独立核实。
公司官方稿可作为原始来源，但若宣传性强，
可降低 quality_score。

5. originality_score：原创程度
- 85-100：首发、独家、官方原始发布或有独家信息
- 65-84：有采访、原创整理或分析
- 40-64：主要整理已有信息
- 20-39：明显转载或高度重复
- 0-19：几乎无原创信息

【十、结构化字段】

source_type 优先选择：
政府、公司官方、科研机构、大学、投资机构、
行业媒体、综合媒体、自媒体、其他。

technology_route 优先选择：
超导量子、离子阱、中性原子、光量子、硅量子点、
拓扑量子、量子退火、相干伊辛机、量子软件、量子算法、
量子通信、量子测量、量子传感、后量子密码、
量子综合、无法判断。

若涉及多个路线，使用中文顿号“、”分隔。

evidence_level 只能为“高”“中”“低”：
- 高：有明确金额、政策、合同、参数、研究结果、
  官方公告或多个具体事实
- 中：主体和事件明确，但关键数据不完整
- 低：宣传、观点、活动介绍或缺少具体事实

is_promotional 只能为“是”或“否”。
大量愿景和宣传词、但缺少产品、合同、金额、
技术成果或落地事实时，通常填写“是”。

selection_reason：
用一句话说明是否值得进入量子行业日报或周报。
必须依据正文事实密度、相关性和行业影响，
不得只根据标题判断。

【十一、保存字段】

调用任何内容保存工具时，必须完整传入：

title、account、publish_time、article_url、category、companies、
keywords、summary、importance、reason、relevance_score、
quality_score、importance_score、source_reliability_score、
originality_score、source_type、technology_route、evidence_level、
is_promotional、selection_reason。

不得省略任何字段。

【十二、搜索任务】

1. 微信搜索：
   - 调用 search_wechat_articles；
   - 未指定时间默认最近7天；
   - 未指定数量默认5篇；
   - 搜索工具会自动扩展相关搜索词。
2. 普通网页搜索：
   - 调用 search_web_pages；
   - 对需要分析的结果逐个调用 scrape_webpage；
   - 用户要求保存时逐个调用 save_webpage_analysis。
3. 必须区分：
   - status="not_found"：没有搜到；
   - status="found_but_all_duplicate"：搜到了，但此前均已记录；
   - status="found_new"：搜到并有新增。
4. 不得把 new_count=0 错误描述成 found_count=0。

【十三、最终展示】

每篇被分析的内容，最终必须忠实展示：

- 标题
- 来源或公众号
- 发布时间
- 分类
- 涉及主体
- 技术路线
- 关键词
- 摘要
- 重要程度
- relevance_score
- quality_score
- importance_score
- source_reliability_score
- originality_score
- 是否宣传稿
- 入选理由
- 保存工具返回的真实状态

如果保存状态是 duplicate，
只能说“此前已保存，本次已跳过”。

如果保存状态是 saved，
只能说“本次新增成功”。

如果保存状态是 failed，
必须展示真实失败原因。

不得自行推测、改写或美化保存状态。
""",


    tools=[
        search_wechat_articles,
        analyze_wechat_articles,
        analyze_links_file,
        search_web_pages,
        scrape_webpage,
        save_webpage_analysis,
    ],
)


# ============================================================
# 7. 对话入口
# ============================================================

async def main() -> None:
    print("=" * 60)
    print("资讯搜索抓取总控 Agent")
    print(f"模型：{MODEL_NAME}")
    print("=" * 60)

    print(
        "\n示例：\n"
        "搜索最近7天量子计算融资相关的微信公众号文章，"
        "最多5篇并分析保存\n"
        "搜索 IBM 量子计算最新进展的普通网页，最多3篇并总结\n"
        "分析这个网页：https://example.com/article\n"
        "处理 data/links.csv 中的全部微信文章\n"
    )

    while True:
        command = input(
            "\n请输入任务，输入 exit 退出：\n"
        ).strip()

        if command.lower() in {
            "exit",
            "quit",
            "退出",
        }:
            print("程序已退出。")
            return

        if not command:
            continue

        try:
            result = await Runner.run(
                starting_agent=main_agent,
                input=command,
                max_turns=30,
            )

            print("\n" + "=" * 60)
            print("任务结果")
            print("=" * 60)
            print(result.final_output)

        except Exception as error:
            print("\n任务执行失败：")
            print(type(error).__name__)
            print(error)


if __name__ == "__main__":
    asyncio.run(main())
