import csv
import re
import unicodedata

from agents import function_tool

from app.core.paths import WECHAT_RESULTS_FILE
from app.core.result_schema import (
    RESULT_FIELDS,
    ensure_csv_schema,
)
from app.wechat.url_utils import (
    is_valid_wechat_url,
    normalize_wechat_url,
)


OUTPUT_FILE = WECHAT_RESULTS_FILE

CSV_FIELDS = RESULT_FIELDS


def validate_wechat_article_url(url: str) -> str:
    """标准化并验证微信公众号文章 URL。"""

    normalized_url = normalize_wechat_url(url)
    if not is_valid_wechat_url(normalized_url):
        raise ValueError(
            "article_url 不是有效的微信公众号文章链接，"
            "已拒绝写入 wechat_results.csv。"
        )

    return normalized_url


def normalize_dedup_text(text: str) -> str:
    """
    统一全半角、大小写，并删除空格和标点。
    """

    normalized = unicodedata.normalize(
        "NFKC",
        str(text or ""),
    ).lower()

    normalized = re.sub(
        r"[\W_]+",
        "",
        normalized,
        flags=re.UNICODE,
    )

    return normalized


def build_title_account_key(
    title: str,
    account: str,
) -> str:
    """
    构建“来源 + 标题”文章去重键。
    """

    clean_title = normalize_dedup_text(
        title
    )
    clean_account = normalize_dedup_text(
        account
    )

    if not clean_title:
        return ""

    return f"{clean_account}|{clean_title}"


def ensure_output_schema() -> None:
    """
    检查已有 CSV 表头是否与当前代码一致。
    """

    ensure_csv_schema(OUTPUT_FILE, CSV_FIELDS)


def find_existing_duplicate(
    article_url: str,
    title: str,
    account: str,
) -> dict[str, str] | None:
    """
    按完整链接、公众号和标题查找重复文章。
    """

    if not OUTPUT_FILE.exists():
        return None

    ensure_output_schema()

    normalized_url = normalize_wechat_url(
        article_url
    )
    current_title_key = build_title_account_key(
        title,
        account,
    )

    with OUTPUT_FILE.open(
        mode="r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            existing_url = normalize_wechat_url(
                row.get("article_url", "")
            )

            if (
                normalized_url
                and existing_url == normalized_url
            ):
                return {
                    "reason": "same_url",
                    "title": row.get("title", ""),
                    "account": row.get("account", ""),
                    "article_url": row.get(
                        "article_url",
                        "",
                    ),
                }

            existing_title_key = build_title_account_key(
                row.get("title", ""),
                row.get("account", ""),
            )

            if (
                current_title_key
                and existing_title_key
                and current_title_key
                == existing_title_key
            ):
                return {
                    "reason": "same_title_account",
                    "title": row.get("title", ""),
                    "account": row.get("account", ""),
                    "article_url": row.get(
                        "article_url",
                        "",
                    ),
                }

    return None


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
    在调用浏览器和模型前检查文章是否已处理。
    """

    normalized_url = normalize_wechat_url(url)
    return normalized_url in load_existing_urls()


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
    """

    print("\n[工具调用] 正在保存分析结果……")

    try:
        ensure_output_schema()

        normalized_url = validate_wechat_article_url(
            article_url
        )

        duplicate = find_existing_duplicate(
            article_url=normalized_url,
            title=title,
            account=account,
        )

        if duplicate is not None:
            if duplicate["reason"] == "same_title_account":
                duplicate_reason = (
                    "公众号/来源与标题相同，"
                    "虽然搜狗签名链接不同，仍判定为同一篇文章"
                )
            else:
                duplicate_reason = "标准化后的文章链接相同"

            message = (
                "保存状态：duplicate。"
                f"{duplicate_reason}；"
                "此前已保存，本次已跳过。"
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
