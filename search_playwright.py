import csv
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import (
    parse_qsl,
    quote_plus,
    urlencode,
    urlparse,
    urlunparse,
)

import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright


# ============================================================
# 1. 基础配置
# ============================================================

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

LINKS_FILE = DATA_DIR / "links.csv"
SEARCH_LOG_FILE = DATA_DIR / "search_log.csv"

# 默认使用最近3天滚动窗口。
# main_agent.py 仍可在运行时动态修改 SEARCH_DAYS。
SEARCH_DAYS = 3

QUERY_INTERVAL_SECONDS = 5
RESULT_INTERVAL_MILLISECONDS = 1800

# 每个搜索词最多检查的候选数量。
MAX_CANDIDATES_PER_QUERY = 40

# 每个主题默认保留数量。
DEFAULT_RESULTS_PER_QUERY = 2


# ============================================================
# 2. 搜索关键词体系
# ============================================================

TECHNOLOGY_QUERIES = [
    "量子计算",
    "量子科技",
    "量子通信",
    "量子测量",
    "量子精密测量",
    "量子传感",
    "量子芯片",
    "量子软件",
    "量子算法",
    "量子云平台",
    "量子安全",
    "后量子密码",
    "超导量子",
    "离子阱量子计算",
    "中性原子量子计算",
    "光量子计算",
    "硅量子点",
    "拓扑量子",
    "量子退火",
    "相干伊辛机",
]

HIGH_PRIORITY_TECHNOLOGIES = [
    "量子计算",
    "量子通信",
    "量子芯片",
    "量子软件",
    "量子传感",
    "后量子密码",
    "超导量子",
    "中性原子量子计算",
    "光量子计算",
]

EVENT_QUERIES = [
    "融资",
    "投资",
    "并购",
    "合作",
    "签约",
    "合同",
    "订单",
    "中标",
    "产品发布",
    "技术突破",
    "科研进展",
    "政策",
    "政府项目",
    "商业化",
    "上市",
    "IPO",
]

HIGH_PRIORITY_EVENTS = [
    "融资",
    "合作",
    "产品发布",
    "技术突破",
    "科研进展",
    "政策",
    "商业化",
]

# 搜狗微信以中文资讯为主。
# 这些英文词用于发现英文名称出现在中文公众号中的文章。
ENGLISH_QUERIES = [
    "quantum computing",
    "quantum funding",
    "quantum startup",
    "quantum processor",
    "post-quantum cryptography",
    "superconducting quantum",
    "neutral atom quantum",
    "photonic quantum",
]

COMPANY_QUERIES = [
    "IBM Quantum",
    "Google Quantum AI",
    "Microsoft Quantum",
    "IonQ",
    "Quantinuum",
    "Rigetti",
    "D-Wave",
    "PsiQuantum",
    "QuEra",
    "Pasqal",
    "本源量子",
    "国盾量子",
    "中电信量子",
    "逻辑比特",
    "中科量枢",
    "玻色量子",
    "图灵量子",
]


def build_coverage_queries(
    include_english: bool = True,
    include_companies: bool = True,
) -> list[str]:
    """
    构建覆盖率优先的搜索词列表。

    顺序：
    1. 宽泛技术方向
    2. 技术方向 × 高频事件
    3. 英文主题
    4. 重点公司和机构
    """

    queries: list[str] = []

    queries.extend(
        TECHNOLOGY_QUERIES
    )

    for technology in HIGH_PRIORITY_TECHNOLOGIES:
        for event in HIGH_PRIORITY_EVENTS:
            queries.append(
                f"{technology} {event}"
            )

    if include_english:
        queries.extend(
            ENGLISH_QUERIES
        )

    if include_companies:
        for company in COMPANY_QUERIES:
            queries.extend(
                [
                    company,
                    f"{company} 融资",
                    f"{company} 产品",
                    f"{company} 合作",
                ]
            )

    # 保留原顺序去重。
    return list(
        dict.fromkeys(
            query.strip()
            for query in queries
            if query.strip()
        )
    )


def append_search_log(
    query: str,
    candidate_count: int,
    valid_time_count: int,
    duplicate_count: int,
    found_count: int,
    failure_reason: str = "",
) -> None:
    """
    保存搜索过程日志，便于分析覆盖率。
    """

    fieldnames = [
        "search_time",
        "query",
        "channel",
        "days",
        "candidate_count",
        "valid_time_count",
        "duplicate_count",
        "found_count",
        "failure_reason",
    ]

    row = {
        "search_time": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "query": query,
        "channel": "搜狗微信",
        "days": SEARCH_DAYS,
        "candidate_count": candidate_count,
        "valid_time_count": valid_time_count,
        "duplicate_count": duplicate_count,
        "found_count": found_count,
        "failure_reason": failure_reason,
    }

    file_exists = SEARCH_LOG_FILE.exists()

    with SEARCH_LOG_FILE.open(
        mode="a",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


# ============================================================
# 3. 日期处理
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
    days: int = SEARCH_DAYS,
) -> bool:
    """
    按自然日期判断文章是否在最近指定天数内。
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

    return earliest_date <= publish_date <= today


# ============================================================
# 4. 微信链接处理
# ============================================================

def normalize_wechat_url(url: str) -> str:
    """
    标准化微信公众号文章链接。

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


def is_wechat_article_url(url: str) -> bool:
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



def choose_valid_wechat_url(
    candidate_urls: list[str],
) -> str:
    """
    从多个候选地址中选择一个有效的微信公众号文章链接。
    """

    for candidate_url in candidate_urls:
        normalized_url = normalize_wechat_url(
            candidate_url
        )

        if is_wechat_article_url(
            normalized_url
        ):
            return normalized_url

    return ""


def extract_real_wechat_url(
    article_page,
) -> str:
    """
    从文章页面识别真实微信公众号文章链接。

    优先检查：
    1. 当前地址栏
    2. canonical
    3. og:url
    """

    candidate_urls = [
        article_page.url,
    ]

    html = article_page.content()
    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    canonical_tag = soup.select_one(
        'link[rel="canonical"]'
    )

    if canonical_tag:
        canonical_url = canonical_tag.get(
            "href"
        )

        if canonical_url:
            candidate_urls.append(
                canonical_url
            )

    og_url_tag = soup.select_one(
        'meta[property="og:url"]'
    )

    if og_url_tag:
        og_url = og_url_tag.get(
            "content"
        )

        if og_url:
            candidate_urls.append(
                og_url
            )

    for candidate_url in candidate_urls:
        normalized_url = normalize_wechat_url(
            candidate_url
        )

        if is_wechat_article_url(
            normalized_url
        ):
            return normalized_url

    return ""


# ============================================================
# 5. 发布时间提取
# ============================================================

def extract_publish_time_from_page(
    article_page,
) -> str:
    """
    从微信文章页面提取发布时间。

    依次尝试：
    1. 页面元素
    2. HTML 标签
    3. 页面脚本中的 Unix 时间戳
    """

    selectors = [
        "#publish_time",
        "em#publish_time",
        ".rich_media_meta_text",
    ]

    for selector in selectors:
        locator = article_page.locator(
            selector
        )

        if locator.count() <= 0:
            continue

        try:
            text = (
                locator
                .first
                .inner_text()
                .strip()
            )
        except Exception:
            continue

        if (
            text
            and parse_wechat_publish_time(text)
            is not None
        ):
            return text

    html = article_page.content()

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    publish_tag = (
        soup.select_one("#publish_time")
        or soup.select_one("em#publish_time")
    )

    if publish_tag:
        text = publish_tag.get_text(
            " ",
            strip=True,
        )

        if (
            text
            and parse_wechat_publish_time(text)
            is not None
        ):
            return text

    timestamp_patterns = [
        r'var\s+ct\s*=\s*"(\d{10})"',
        r"var\s+ct\s*=\s*'(\d{10})'",
        r'"createTime"\s*:\s*"(\d{10})"',
        r'"create_time"\s*:\s*"(\d{10})"',
        r'createTime\s*[:=]\s*"(\d{10})"',
        r'create_time\s*[:=]\s*"(\d{10})"',
    ]

    for pattern in timestamp_patterns:
        match = re.search(
            pattern,
            html,
        )

        if not match:
            continue

        timestamp = int(
            match.group(1)
        )

        publish_datetime = datetime.fromtimestamp(
            timestamp
        )

        return publish_datetime.strftime(
            "%Y-%m-%d %H:%M"
        )

    return ""


# ============================================================
# 6. links.csv 读写
# ============================================================

def load_existing_links() -> set[str]:
    """
    读取 data/links.csv 中已有链接。
    """

    if not LINKS_FILE.exists():
        return set()

    try:
        df = pd.read_csv(
            LINKS_FILE
        )
    except pd.errors.EmptyDataError:
        return set()

    if "article_url" not in df.columns:
        raise ValueError(
            "data/links.csv 中必须包含 article_url 列。"
        )

    existing_links: set[str] = set()

    for raw_url in df["article_url"].dropna():
        normalized = normalize_wechat_url(
            str(raw_url)
        )

        if is_wechat_article_url(
            normalized
        ):
            existing_links.add(
                normalized
            )

    return existing_links


def append_new_links(
    new_links: list[str],
) -> None:
    """
    将新链接追加写入 data/links.csv。
    """

    if not new_links:
        return

    file_exists = LINKS_FILE.exists()

    df = pd.DataFrame(
        {
            "article_url": new_links,
        }
    )

    df.to_csv(
        LINKS_FILE,
        mode="a",
        header=not file_exists,
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# 7. 构建搜索词
# ============================================================

def build_queries(
    max_queries: int,
) -> list[str]:
    """
    返回覆盖率优先的前 max_queries 个搜索词。
    """

    all_queries = build_coverage_queries()

    return all_queries[
        :max_queries
    ]


# ============================================================
# 8. 搜狗微信搜索
# ============================================================

def search_sogou_with_playwright(
    query: str,
    max_results: int,
) -> list[str]:
    """
    使用搜狗微信搜索公众号文章。

    搜狗通常返回中转链接。
    程序会打开结果、识别真实微信链接、
    提取发布时间，并仅保留最近 7 个自然日的文章。
    """

    found_links: list[str] = []
    valid_time_count = 0
    duplicate_count = 0
    result_count = 0
    failure_reason = ""

    search_url = (
        "https://weixin.sogou.com/weixin"
        f"?type=2&query={quote_plus(query)}"
    )

    print(f"\n正在搜索：{query}")
    print(f"搜索地址：{search_url}")

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

        try:
            page.goto(
                search_url,
                wait_until="domcontentloaded",
                timeout=60_000,
            )

            page.wait_for_timeout(
                3000
            )

            body_text = page.locator(
                "body"
            ).inner_text()

            if (
                "验证码" in body_text
                or "访问过于频繁" in body_text
                or "请输入验证码" in body_text
            ):
                print(
                    "[停止搜索] 搜狗要求验证码或限制访问。"
                )
                print(
                    "请手动完成验证后重新运行，"
                    "不要尝试自动绕过。"
                )

                failure_reason = "验证码或访问频率限制"

                append_search_log(
                    query=query,
                    candidate_count=result_count,
                    valid_time_count=valid_time_count,
                    duplicate_count=duplicate_count,
                    found_count=len(found_links),
                    failure_reason=failure_reason,
                )

                return []

            result_links = page.locator(
                "ul.news-list li h3 a"
            )

            result_count = result_links.count()

            print(
                f"[搜索页面结果数量] {result_count}"
            )

            if result_count == 0:
                screenshot_file = (
                    DATA_DIR
                    / "sogou_search_debug.png"
                )

                page.screenshot(
                    path=str(screenshot_file),
                    full_page=True,
                )

                print(
                    "[没有匹配结果] 可能是关键词无结果、"
                    "页面结构变化或网络限制。"
                )
                print(
                    f"[调试截图] {screenshot_file.resolve()}"
                )

                append_search_log(
                    query=query,
                    candidate_count=0,
                    valid_time_count=0,
                    duplicate_count=0,
                    found_count=0,
                    failure_reason="无搜索结果或页面结构变化",
                )

                return []

            check_count = min(
                result_count,
                max(
                    MAX_CANDIDATES_PER_QUERY,
                    max_results * 6,
                ),
            )

            for index in range(
                check_count
            ):
                if len(found_links) >= max_results:
                    break

                result_link = result_links.nth(
                    index
                )

                try:
                    title = (
                        result_link
                        .inner_text()
                        .strip()
                    )
                except Exception:
                    title = "未获取到标题"

                href = result_link.get_attribute(
                    "href"
                )

                if not href:
                    print(
                        "[忽略] 搜索结果没有链接。"
                    )
                    continue

                print(
                    f"\n[检查搜索结果] {title}"
                )

                article_page = None
                captured_wechat_urls: list[str] = []

                def capture_request(request) -> None:
                    request_url = request.url

                    if (
                        "mp.weixin.qq.com"
                        in request_url
                    ):
                        captured_wechat_urls.append(
                            request_url
                        )

                context.on(
                    "request",
                    capture_request,
                )

                try:
                    # 点击搜狗结果，并同时监听所有网络请求。
                    # 即使页面最终停在搜狗中转页，
                    # 只要浏览器请求过微信文章地址，也可以捕获。
                    pages_before_click = set(
                        context.pages
                    )

                    try:
                        result_link.click(
                            timeout=10_000
                        )
                    except Exception as click_error:
                        print(
                            f"[点击失败] {click_error}"
                        )
                        continue

                    page.wait_for_timeout(
                        8000
                    )

                    pages_after_click = context.pages
                    new_pages = [
                        opened_page
                        for opened_page in pages_after_click
                        if opened_page
                        not in pages_before_click
                    ]

                    if new_pages:
                        article_page = new_pages[-1]
                    else:
                        article_page = page

                    try:
                        article_page.wait_for_load_state(
                            "domcontentloaded",
                            timeout=30_000,
                        )
                    except Exception:
                        pass

                    article_page.wait_for_timeout(
                        4000
                    )

                    page_urls = [
                        opened_page.url
                        for opened_page in context.pages
                        if not opened_page.is_closed()
                    ]

                    print(
                        "[当前浏览器页面] "
                        + " | ".join(page_urls)
                    )

                    if captured_wechat_urls:
                        print(
                            "[捕获到微信请求] "
                            f"{len(captured_wechat_urls)} 条"
                        )
                    else:
                        print(
                            "[捕获到微信请求] 0 条"
                        )

                    candidate_urls = (
                        list(reversed(
                            captured_wechat_urls
                        ))
                        + list(reversed(
                            page_urls
                        ))
                    )

                    final_url = choose_valid_wechat_url(
                        candidate_urls
                    )

                    # 网络监听没有抓到时，再从页面 HTML 中寻找。
                    if (
                        not final_url
                        and article_page is not None
                        and not article_page.is_closed()
                    ):
                        final_url = extract_real_wechat_url(
                            article_page
                        )

                    raw_final_url = (
                        article_page.url
                        if (
                            article_page is not None
                            and not article_page.is_closed()
                        )
                        else ""
                    )

                    print(
                        f"[跳转后原始链接] {raw_final_url}"
                    )
                    print(
                        "[识别到文章链接] "
                        f"{final_url or '未识别到'}"
                    )

                    if not final_url:
                        print(
                            "[忽略] 页面虽然打开，"
                            "但没有识别到有效微信文章链接。"
                        )
                        continue

                    publish_time_text = (
                        extract_publish_time_from_page(
                            article_page
                        )
                    )

                    print(
                        "[文章发布时间] "
                        f"{publish_time_text or '未获取到'}"
                    )

                    if not publish_time_text:
                        print(
                            "[忽略] 未获取到发布时间，"
                            f"无法确认是否属于近 {SEARCH_DAYS} 天。"
                        )
                        continue

                    if not is_within_last_days(
                        publish_time_text,
                        days=SEARCH_DAYS,
                    ):
                        print(
                            "[忽略旧文章] 发布时间不在"
                            f"最近 {SEARCH_DAYS} 个自然日内。"
                        )
                        continue

                    valid_time_count += 1

                    if final_url in found_links:
                        duplicate_count += 1

                        print(
                            "[忽略重复] 本关键词中已发现该文章。"
                        )
                        continue

                    found_links.append(
                        final_url
                    )

                    print(
                        f"[发现近{SEARCH_DAYS}天文章] "
                        f"{final_url}"
                    )

                except Exception as error:
                    print(
                        f"[打开结果失败] {error}"
                    )

                finally:
                    try:
                        context.remove_listener(
                            "request",
                            capture_request,
                        )
                    except Exception:
                        pass

                    # 关闭点击后新开的页面。
                    for opened_page in list(
                        context.pages
                    ):
                        if (
                            opened_page != page
                            and not opened_page.is_closed()
                        ):
                            try:
                                opened_page.close()
                            except Exception:
                                pass

                    # 搜狗结果在当前页打开时，返回搜索页。
                    if (
                        not page.is_closed()
                        and "weixin.sogou.com/weixin"
                        not in page.url
                    ):
                        try:
                            page.goto(
                                search_url,
                                wait_until="domcontentloaded",
                                timeout=60_000,
                            )
                            page.wait_for_timeout(
                                3000
                            )
                        except Exception:
                            pass

                    result_links = page.locator(
                        "ul.news-list li h3 a"
                    )

                page.wait_for_timeout(
                    RESULT_INTERVAL_MILLISECONDS
                )

        except Exception as error:
            print(
                f"[搜索页面打开失败] {error}"
            )

        finally:
            context.close()
            browser.close()

    append_search_log(
        query=query,
        candidate_count=result_count,
        valid_time_count=valid_time_count,
        duplicate_count=duplicate_count,
        found_count=len(found_links),
        failure_reason=failure_reason,
    )

    return found_links


def search_multiple_topics(
    queries: list[str],
    results_per_query: int = DEFAULT_RESULTS_PER_QUERY,
) -> list[str]:
    """
    每个主题都单独搜索，再统一合并去重。

    不会因为前面的主题已经搜够文章，
    就跳过后续技术方向。
    """

    discovered_links: list[str] = []
    seen_links: set[str] = set()

    for index, query in enumerate(
        queries,
        start=1,
    ):
        print(
            f"\n[{index}/{len(queries)}] 搜索主题：{query}"
        )

        try:
            results = search_sogou_with_playwright(
                query=query,
                max_results=results_per_query,
            )
        except Exception as error:
            print(
                f"[搜索失败] {query}：{error}"
            )
            continue

        for url in results:
            if url in seen_links:
                continue

            seen_links.add(url)
            discovered_links.append(url)

        if index < len(queries):
            time.sleep(
                QUERY_INTERVAL_SECONDS
            )

    return discovered_links


# ============================================================
# 9. 主程序
# ============================================================

def main() -> None:
    print("=" * 60)
    print("量子行业微信公众号覆盖率优先搜索")
    print(
        f"滚动时间窗口：最近 {SEARCH_DAYS} 个自然日"
    )
    print("=" * 60)

    max_queries_text = input(
        "本次执行多少个搜索词？建议第一次输入10："
    ).strip()

    try:
        max_queries = int(
            max_queries_text
        )
    except ValueError:
        print(
            "搜索词数量必须是整数。"
        )
        return

    results_per_query_text = input(
        "每个搜索词最多保留几篇？建议输入2："
    ).strip()

    try:
        results_per_query = int(
            results_per_query_text
        )
    except ValueError:
        print(
            "结果数量必须是整数。"
        )
        return

    if max_queries <= 0:
        print(
            "搜索词数量必须大于0。"
        )
        return

    if results_per_query <= 0:
        print(
            "每个搜索词的结果数量必须大于0。"
        )
        return

    queries = build_queries(
        max_queries=max_queries
    )

    existing_links = load_existing_links()

    discovered_links = search_multiple_topics(
        queries=queries,
        results_per_query=results_per_query,
    )

    new_links = [
        url
        for url in discovered_links
        if url not in existing_links
    ]

    append_new_links(
        new_links
    )

    print("\n" + "=" * 60)
    print("搜索完成")
    print("=" * 60)
    print(
        f"执行搜索词：{len(queries)}"
    )
    print(
        f"原有链接：{len(existing_links)}"
    )
    print(
        f"合并后候选：{len(discovered_links)}"
    )
    print(
        f"本次新发现：{len(new_links)}"
    )
    print(
        f"链接文件：{LINKS_FILE.resolve()}"
    )
    print(
        f"搜索日志：{SEARCH_LOG_FILE.resolve()}"
    )


if __name__ == "__main__":
    main()