import csv
from urllib.parse import urlparse

from agents import function_tool

from app.core.paths import WEB_RESULTS_FILE
from app.core.result_schema import (
    RESULT_FIELDS,
    current_china_time_text,
    ensure_csv_schema,
)
from app.web.publish_time import (
    is_within_requested_days,
    parse_publish_datetime,
)


WEB_RESULT_FIELDS = RESULT_FIELDS


def validate_webpage_url(url: str) -> str:
    """验证普通网页 URL，并拒绝微信公众号文章链接。"""

    cleaned_url = str(url).strip()
    parsed = urlparse(cleaned_url)
    hostname = (parsed.hostname or "").lower()

    if parsed.scheme not in {"http", "https"} or not hostname:
        raise ValueError("article_url 必须是有效的 HTTP(S) 网页链接。")

    if hostname == "mp.weixin.qq.com":
        raise ValueError(
            "检测到微信公众号链接，已拒绝写入 web_results.csv；"
            "请使用微信文章保存流程。"
        )

    return cleaned_url


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
    requested_days: int = 7,
) -> str:
    """
    将普通网页按照与微信公众号文章一致的结构保存。
    """

    ensure_csv_schema(
        WEB_RESULTS_FILE,
        WEB_RESULT_FIELDS,
    )
    article_url = validate_webpage_url(article_url)
    requested_days = max(
        0,
        min(int(requested_days), 3650),
    )

    if not is_within_requested_days(
        publish_time,
        requested_days,
    ):
        if parse_publish_datetime(publish_time) is None:
            return (
                "保存状态：filtered。"
                f"未获取到可验证的发布时间，无法确认属于最近 "
                f"{requested_days} 天，本次不保存。"
            )

        return (
            "保存状态：filtered。"
            f"发布时间 {publish_time} 不在最近 "
            f"{requested_days} 天内，本次不保存。"
        )

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
        "collected_at": current_china_time_text(),
    }

    file_exists = WEB_RESULTS_FILE.exists()

    with WEB_RESULTS_FILE.open(
        mode="a",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=WEB_RESULT_FIELDS,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

    return (
        "保存状态：saved。"
        f"本次新增成功，文件位置：{WEB_RESULTS_FILE.resolve()}"
    )
