import re
from datetime import datetime, timedelta

from bs4 import BeautifulSoup


def parse_publish_datetime(
    publish_time_text: str,
) -> datetime | None:
    """
    将常见中文、英文和 ISO 发布时间转换为 datetime。
    """

    text = str(publish_time_text).strip()

    if (
        not text
        or text in {
            "未获取到发布时间",
            "未知",
            "无",
            "None",
        }
    ):
        return None

    text = text.replace("T", " ")
    text = re.sub(
        r"\s+(UTC|GMT)$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    iso_candidate = text.replace(
        "Z",
        "+00:00",
    )

    try:
        parsed = datetime.fromisoformat(
            iso_candidate
        )

        if parsed.tzinfo is not None:
            parsed = parsed.astimezone().replace(
                tzinfo=None
            )

        return parsed
    except ValueError:
        pass

    formats = [
        "%Y年%m月%d日 %H:%M:%S",
        "%Y年%m月%d日 %H:%M",
        "%Y年%m月%d日",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d",
        "%Y.%m.%d %H:%M",
        "%Y.%m.%d",
        "%b %d, %Y %H:%M",
        "%b %d, %Y",
        "%B %d, %Y %H:%M",
        "%B %d, %Y",
    ]

    for date_format in formats:
        try:
            return datetime.strptime(
                text,
                date_format,
            )
        except ValueError:
            continue

    date_patterns = [
        r"(20\d{2})年(\d{1,2})月(\d{1,2})日",
        r"(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})",
    ]

    for pattern in date_patterns:
        match = re.search(
            pattern,
            text,
        )

        if not match:
            continue

        try:
            return datetime(
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
            )
        except ValueError:
            continue

    return None


def is_within_requested_days(
    publish_time_text: str,
    days: int,
) -> bool:
    """
    判断发布时间是否位于最近 days 个自然日内。

    days <= 0 表示不启用时间限制。
    """

    if int(days) <= 0:
        return True

    publish_datetime = parse_publish_datetime(
        publish_time_text
    )

    if publish_datetime is None:
        return False

    today = datetime.now().date()
    earliest_date = today - timedelta(
        days=int(days) - 1
    )

    return (
        earliest_date
        <= publish_datetime.date()
        <= today
    )


def extract_web_publish_time(
    soup: BeautifulSoup,
    html: str,
) -> str:
    """
    从普通网页的 meta、time 标签、JSON-LD 和正文中提取发布时间。
    """

    meta_candidates = [
        ("property", "article:published_time"),
        ("property", "og:published_time"),
        ("name", "publishdate"),
        ("name", "publish_date"),
        ("name", "pubdate"),
        ("name", "date"),
        ("name", "timestamp"),
        ("itemprop", "datePublished"),
    ]

    for attribute, value in meta_candidates:
        tag = soup.find(
            "meta",
            attrs={
                attribute: value,
            },
        )

        if not tag:
            continue

        content = str(
            tag.get("content", "")
        ).strip()

        if parse_publish_datetime(content):
            return content

    time_tags = soup.find_all(
        "time"
    )

    for time_tag in time_tags:
        candidate = str(
            time_tag.get("datetime", "")
            or time_tag.get_text(
                " ",
                strip=True,
            )
        ).strip()

        if parse_publish_datetime(candidate):
            return candidate

    json_patterns = [
        r'"datePublished"\s*:\s*"([^"]+)"',
        r'"dateCreated"\s*:\s*"([^"]+)"',
        r'"pubDate"\s*:\s*"([^"]+)"',
        r'"publishTime"\s*:\s*"([^"]+)"',
        r'"publish_time"\s*:\s*"([^"]+)"',
    ]

    for pattern in json_patterns:
        match = re.search(
            pattern,
            html,
            flags=re.IGNORECASE,
        )

        if (
            match
            and parse_publish_datetime(
                match.group(1)
            )
        ):
            return match.group(1).strip()

    page_text = soup.get_text(
        " ",
        strip=True,
    )[:5000]

    text_patterns = [
        r"20\d{2}年\d{1,2}月\d{1,2}日(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?",
        r"20\d{2}[-/.]\d{1,2}[-/.]\d{1,2}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?",
    ]

    for pattern in text_patterns:
        match = re.search(
            pattern,
            page_text,
        )

        if (
            match
            and parse_publish_datetime(
                match.group(0)
            )
        ):
            return match.group(0)

    return "未获取到发布时间"
