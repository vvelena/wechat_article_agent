import json
import os
from dataclasses import dataclass

from app.core.paths import (
    DATA_DIR,
)
from app.wechat.publish_time import (
    is_within_last_days,
    parse_wechat_publish_time,
)
from app.wechat.result_store import (
    article_already_saved,
    save_article_analysis,
)
from app.wechat.url_utils import (
    is_valid_wechat_url,
    normalize_wechat_url,
)

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

DATA_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
MAX_CONTENT_LENGTH = 15_000
ARTICLE_INTERVAL_SECONDS = 3

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
                    "链接格式不正确。目前只支持 "
                    "https://mp.weixin.qq.com/s/... 形式的文章。"
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
                    "source_kind": "wechat",
                    "error": "网页打开成功，但没有提取到正文。",
                    "article_url": normalized_url,
                },
                ensure_ascii=False,
            )

        content = content[:MAX_CONTENT_LENGTH]

        article_data = {
            "success": True,
            "source_kind": "wechat",
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
                "source_kind": "wechat",
                "error": str(error),
                "article_url": normalized_url,
            },
            ensure_ascii=False,
        )


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
2. 抓取结果的 source_kind 必须为 wechat；保存时必须原样使用
   article_url，不得替换、遗漏或改写为其他候选链接。
3. 如果 scrape_wechat_article 返回 success=false：
   - 说明失败原因；
   - 不得调用 save_article_analysis；
   - 不得编造文章信息。
4. 如果 success=true：
   - 阅读标题、公众号、发布时间和正文；
   - 进行分析；
   - 只调用一次 save_article_analysis。
5. 调用 save_article_analysis 后，无论返回 saved、duplicate 或 failed，
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


