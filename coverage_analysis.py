import csv
import json
import re
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd


DATA_DIR = Path("data")
WECHAT_RESULTS_FILE = DATA_DIR / "wechat_results.csv"
WEB_RESULTS_FILE = DATA_DIR / "web_results.csv"
BENCHMARK_FILE = DATA_DIR / "benchmark_events.csv"
REPORT_FILE = DATA_DIR / "coverage_report.json"

TARGET_CATEGORIES = {
    "融资",
    "投资",
    "并购",
    "合作",
    "合同订单",
    "产品发布",
    "技术研发",
    "科研进展",
    "政策",
    "政府项目",
    "产业园区",
    "市场动态",
}

TARGET_TECHNOLOGY_ROUTES = {
    "超导量子",
    "离子阱",
    "中性原子",
    "光量子",
    "硅量子点",
    "拓扑量子",
    "量子退火",
    "相干伊辛机",
    "量子软件",
    "量子算法",
    "量子通信",
    "量子测量",
    "量子传感",
    "后量子密码",
}

TARGET_SOURCE_TYPES = {
    "政府",
    "公司官方",
    "科研机构",
    "大学",
    "投资机构",
    "行业媒体",
    "综合媒体",
    "自媒体",
}

REQUIRED_FIELDS = [
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


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize(
        "NFKC",
        str(text or ""),
    ).lower()

    return re.sub(
        r"[\W_]+",
        "",
        normalized,
        flags=re.UNICODE,
    )


def split_chinese_list(value: str) -> list[str]:
    return [
        item.strip()
        for item in re.split(
            r"[、,，;/；|]+",
            str(value or ""),
        )
        if item.strip()
    ]


def load_result_data() -> pd.DataFrame:
    frames = []

    for source_name, path in [
        ("微信公众号", WECHAT_RESULTS_FILE),
        ("普通网页", WEB_RESULTS_FILE),
    ]:
        if not path.exists() or path.stat().st_size == 0:
            continue

        try:
            frame = pd.read_csv(
                path,
                encoding="utf-8-sig",
            )
        except pd.errors.EmptyDataError:
            continue

        frame["_collection_source"] = source_name
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(
            "没有找到可分析的数据。请确认 "
            "data/wechat_results.csv 或 "
            "data/web_results.csv 至少存在一个。"
        )

    return pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )


def filter_by_days(
    df: pd.DataFrame,
    days: int,
) -> tuple[pd.DataFrame, int]:
    if days <= 0:
        return df.copy(), 0

    publish_datetime = pd.to_datetime(
        df.get("publish_time"),
        errors="coerce",
    )

    today = datetime.now().date()
    earliest_date = today - timedelta(
        days=days - 1
    )

    mask = publish_datetime.dt.date.between(
        earliest_date,
        today,
    )

    filtered = df.loc[mask].copy()
    invalid_date_count = int(
        publish_datetime.isna().sum()
    )

    return filtered, invalid_date_count


def field_is_complete(series: pd.Series) -> pd.Series:
    text = (
        series
        .fillna("")
        .astype(str)
        .str.strip()
    )

    invalid_values = {
        "",
        "无",
        "未知",
        "无法判断",
        "未获取到发布时间",
        "未获取到标题",
        "未获取到公众号",
        "none",
        "nan",
    }

    return ~text.str.lower().isin(
        invalid_values
    )


def calculate_internal_coverage(
    days: int = 0,
) -> dict:
    df = load_result_data()
    original_count = len(df)

    df, invalid_date_count = filter_by_days(
        df,
        days,
    )

    if df.empty:
        raise ValueError(
            "指定时间范围内没有可分析的数据。"
        )

    categories = set(
        df.get(
            "category",
            pd.Series(dtype=str),
        )
        .dropna()
        .astype(str)
        .str.strip()
    )

    routes = set()

    for value in df.get(
        "technology_route",
        pd.Series(dtype=str),
    ).fillna(""):
        routes.update(
            split_chinese_list(value)
        )

    routes -= {
        "",
        "量子综合",
        "无法判断",
    }

    source_types = set(
        df.get(
            "source_type",
            pd.Series(dtype=str),
        )
        .dropna()
        .astype(str)
        .str.strip()
    )

    covered_categories = (
        TARGET_CATEGORIES & categories
    )
    missing_categories = (
        TARGET_CATEGORIES - categories
    )

    covered_routes = (
        TARGET_TECHNOLOGY_ROUTES & routes
    )
    missing_routes = (
        TARGET_TECHNOLOGY_ROUTES - routes
    )

    covered_source_types = (
        TARGET_SOURCE_TYPES & source_types
    )
    missing_source_types = (
        TARGET_SOURCE_TYPES - source_types
    )

    category_coverage = (
        len(covered_categories)
        / len(TARGET_CATEGORIES)
    )

    technology_coverage = (
        len(covered_routes)
        / len(TARGET_TECHNOLOGY_ROUTES)
    )

    source_type_coverage = (
        len(covered_source_types)
        / len(TARGET_SOURCE_TYPES)
    )

    publish_datetime = pd.to_datetime(
        df.get("publish_time"),
        errors="coerce",
    )

    valid_dates = (
        publish_datetime
        .dropna()
        .dt.date
    )

    if days > 0:
        target_days = days
        covered_days = len(
            set(valid_dates)
        )
    elif valid_dates.empty:
        target_days = 0
        covered_days = 0
    else:
        start_date = min(valid_dates)
        end_date = max(valid_dates)
        target_days = (
            end_date - start_date
        ).days + 1
        covered_days = len(
            set(valid_dates)
        )

    time_coverage = (
        covered_days / target_days
        if target_days > 0
        else 0
    )

    field_rates = {}

    for field in REQUIRED_FIELDS:
        if field not in df.columns:
            field_rates[field] = 0.0
            continue

        field_rates[field] = float(
            field_is_complete(
                df[field]
            ).mean()
        )

    field_completeness = (
        sum(field_rates.values())
        / len(field_rates)
    )

    source_counts = (
        df.get(
            "account",
            pd.Series(dtype=str),
        )
        .fillna("未知来源")
        .astype(str)
        .str.strip()
        .value_counts()
    )

    top5_source_share = (
        float(
            source_counts.head(5).sum()
            / len(df)
        )
        if len(df) > 0
        else 0
    )

    score_columns = [
        "relevance_score",
        "quality_score",
        "importance_score",
    ]

    score_df = pd.DataFrame(
        index=df.index
    )

    for column in score_columns:
        score_df[column] = pd.to_numeric(
            df.get(
                column,
                pd.Series(index=df.index),
            ),
            errors="coerce",
        )

    high_value_mask = (
        score_df["relevance_score"].ge(80)
        & score_df["quality_score"].ge(70)
        & score_df["importance_score"].ge(60)
    )

    high_value_count = int(
        high_value_mask.sum()
    )

    high_value_rate = float(
        high_value_count / len(df)
    )

    overall_score = (
        technology_coverage * 0.30
        + category_coverage * 0.25
        + source_type_coverage * 0.20
        + time_coverage * 0.15
        + field_completeness * 0.10
    )

    return {
        "metric_type": "internal_coverage",
        "days": days,
        "original_record_count": original_count,
        "analyzed_record_count": len(df),
        "invalid_publish_time_count": invalid_date_count,
        "category_coverage": round(
            category_coverage,
            4,
        ),
        "covered_categories": sorted(
            covered_categories
        ),
        "missing_categories": sorted(
            missing_categories
        ),
        "technology_route_coverage": round(
            technology_coverage,
            4,
        ),
        "covered_technology_routes": sorted(
            covered_routes
        ),
        "missing_technology_routes": sorted(
            missing_routes
        ),
        "source_type_coverage": round(
            source_type_coverage,
            4,
        ),
        "covered_source_types": sorted(
            covered_source_types
        ),
        "missing_source_types": sorted(
            missing_source_types
        ),
        "time_coverage": round(
            time_coverage,
            4,
        ),
        "target_days": target_days,
        "covered_days": covered_days,
        "field_completeness": round(
            field_completeness,
            4,
        ),
        "field_completeness_detail": {
            key: round(value, 4)
            for key, value
            in field_rates.items()
        },
        "top5_source_share": round(
            top5_source_share,
            4,
        ),
        "high_value_count": high_value_count,
        "high_value_rate": round(
            high_value_rate,
            4,
        ),
        "overall_internal_coverage": round(
            overall_score,
            4,
        ),
        "important_note": (
            "内部覆盖率表示数据是否覆盖预设分类、"
            "技术路线、来源类型、时间与字段范围，"
            "不等于全网资讯覆盖率。"
        ),
    }


def create_benchmark_template() -> Path:
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if BENCHMARK_FILE.exists():
        return BENCHMARK_FILE

    fieldnames = [
        "event_id",
        "event_date",
        "event_title",
        "main_entities",
        "category",
        "technology_route",
        "benchmark_source",
        "source_url",
        "importance",
    ]

    example_rows = [
        {
            "event_id": "example_001",
            "event_date": "2026-07-20",
            "event_title": "示例：某量子公司完成A轮融资",
            "main_entities": "某量子公司、某投资机构",
            "category": "融资",
            "technology_route": "超导量子",
            "benchmark_source": "示例基准来源",
            "source_url": "https://example.com",
            "importance": "高",
        }
    ]

    with BENCHMARK_FILE.open(
        mode="w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(
            example_rows
        )

    return BENCHMARK_FILE


def build_event_text(row: pd.Series) -> str:
    return normalize_text(
        " ".join(
            [
                str(row.get("title", "")),
                str(row.get("companies", "")),
                str(row.get("category", "")),
                str(row.get("technology_route", "")),
                str(row.get("keywords", "")),
                str(row.get("summary", "")),
            ]
        )
    )


def benchmark_event_text(row: pd.Series) -> str:
    return normalize_text(
        " ".join(
            [
                str(row.get("event_title", "")),
                str(row.get("main_entities", "")),
                str(row.get("category", "")),
                str(row.get("technology_route", "")),
            ]
        )
    )


def title_similarity(
    left: str,
    right: str,
) -> float:
    return SequenceMatcher(
        None,
        normalize_text(left),
        normalize_text(right),
    ).ratio()


def entity_overlap(
    benchmark_entities: str,
    collected_companies: str,
) -> float:
    left = {
        normalize_text(item)
        for item in split_chinese_list(
            benchmark_entities
        )
        if normalize_text(item)
    }

    right = {
        normalize_text(item)
        for item in split_chinese_list(
            collected_companies
        )
        if normalize_text(item)
    }

    if not left:
        return 0.0

    matched = 0

    for benchmark_entity in left:
        if any(
            benchmark_entity in item
            or item in benchmark_entity
            for item in right
        ):
            matched += 1

    return matched / len(left)


def event_match_score(
    benchmark_row: pd.Series,
    collected_row: pd.Series,
) -> float:
    benchmark_title = str(
        benchmark_row.get(
            "event_title",
            "",
        )
    )

    collected_title = str(
        collected_row.get(
            "title",
            "",
        )
    )

    title_score = title_similarity(
        benchmark_title,
        collected_title,
    )

    entity_score = entity_overlap(
        str(
            benchmark_row.get(
                "main_entities",
                "",
            )
        ),
        str(
            collected_row.get(
                "companies",
                "",
            )
        ),
    )

    category_score = float(
        normalize_text(
            benchmark_row.get(
                "category",
                "",
            )
        )
        == normalize_text(
            collected_row.get(
                "category",
                "",
            )
        )
        and bool(
            normalize_text(
                benchmark_row.get(
                    "category",
                    "",
                )
            )
        )
    )

    route_score = float(
        bool(
            set(
                split_chinese_list(
                    benchmark_row.get(
                        "technology_route",
                        "",
                    )
                )
            )
            & set(
                split_chinese_list(
                    collected_row.get(
                        "technology_route",
                        "",
                    )
                )
            )
        )
    )

    benchmark_full = benchmark_event_text(
        benchmark_row
    )
    collected_full = build_event_text(
        collected_row
    )

    content_score = SequenceMatcher(
        None,
        benchmark_full,
        collected_full,
    ).ratio()

    return (
        title_score * 0.35
        + entity_score * 0.35
        + category_score * 0.10
        + route_score * 0.10
        + content_score * 0.10
    )


def calculate_relative_coverage(
    days: int = 0,
    match_threshold: float = 0.55,
    important_only: bool = True,
) -> dict:
    create_benchmark_template()

    benchmark_df = pd.read_csv(
        BENCHMARK_FILE,
        encoding="utf-8-sig",
    )

    if benchmark_df.empty:
        raise ValueError(
            "data/benchmark_events.csv 中没有基准事件。"
        )

    benchmark_df = benchmark_df[
        ~benchmark_df["event_id"]
        .astype(str)
        .str.startswith("example_")
    ].copy()

    if benchmark_df.empty:
        raise ValueError(
            "基准事件表目前只有示例行。请先删除示例行，"
            "填写真实基准事件后再计算相对覆盖率。"
        )

    collected_df = load_result_data()

    collected_df, _ = filter_by_days(
        collected_df,
        days,
    )

    if days > 0:
        benchmark_dates = pd.to_datetime(
            benchmark_df["event_date"],
            errors="coerce",
        )

        today = datetime.now().date()
        earliest_date = today - timedelta(
            days=days - 1
        )

        benchmark_df = benchmark_df.loc[
            benchmark_dates.dt.date.between(
                earliest_date,
                today,
            )
        ].copy()

    if important_only:
        importance = (
            benchmark_df["importance"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        benchmark_df = benchmark_df.loc[
            importance.isin(
                {"高", "中"}
            )
        ].copy()

    if benchmark_df.empty:
        raise ValueError(
            "当前筛选范围内没有可比较的基准事件。"
        )

    matched_events = []
    missed_events = []

    for _, benchmark_row in benchmark_df.iterrows():
        best_score = 0.0
        best_match = None

        for _, collected_row in collected_df.iterrows():
            score = event_match_score(
                benchmark_row,
                collected_row,
            )

            if score > best_score:
                best_score = score
                best_match = collected_row

        event_info = {
            "event_id": str(
                benchmark_row.get(
                    "event_id",
                    "",
                )
            ),
            "event_date": str(
                benchmark_row.get(
                    "event_date",
                    "",
                )
            ),
            "event_title": str(
                benchmark_row.get(
                    "event_title",
                    "",
                )
            ),
            "benchmark_source": str(
                benchmark_row.get(
                    "benchmark_source",
                    "",
                )
            ),
            "best_match_score": round(
                best_score,
                4,
            ),
        }

        if (
            best_match is not None
            and best_score >= match_threshold
        ):
            event_info.update(
                {
                    "matched_title": str(
                        best_match.get(
                            "title",
                            "",
                        )
                    ),
                    "matched_source": str(
                        best_match.get(
                            "account",
                            "",
                        )
                    ),
                    "matched_url": str(
                        best_match.get(
                            "article_url",
                            "",
                        )
                    ),
                }
            )
            matched_events.append(
                event_info
            )
        else:
            missed_events.append(
                event_info
            )

    total = len(benchmark_df)
    matched_count = len(
        matched_events
    )
    coverage_rate = (
        matched_count / total
        if total > 0
        else 0
    )

    result = {
        "metric_type": "relative_event_coverage",
        "days": days,
        "important_only": important_only,
        "match_threshold": match_threshold,
        "benchmark_event_count": total,
        "matched_event_count": matched_count,
        "missed_event_count": len(
            missed_events
        ),
        "relative_coverage_rate": round(
            coverage_rate,
            4,
        ),
        "miss_rate": round(
            1 - coverage_rate,
            4,
        ),
        "matched_events": matched_events,
        "missed_events": missed_events,
        "benchmark_file": str(
            BENCHMARK_FILE.resolve()
        ),
        "important_note": (
            "相对覆盖率取决于基准事件表和自动匹配阈值。"
            "边界结果建议人工复核。"
        ),
    }

    return result


def save_report(
    report: dict,
) -> Path:
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_FILE.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return REPORT_FILE


def format_percentage(value: float) -> str:
    return f"{value:.1%}"


def format_internal_report(
    result: dict,
) -> str:
    lines = [
        "内部覆盖率分析完成",
        f"分析记录数：{result['analyzed_record_count']}",
        (
            "综合内部覆盖率："
            + format_percentage(
                result[
                    "overall_internal_coverage"
                ]
            )
        ),
        (
            "事件分类覆盖率："
            + format_percentage(
                result[
                    "category_coverage"
                ]
            )
        ),
        (
            "技术路线覆盖率："
            + format_percentage(
                result[
                    "technology_route_coverage"
                ]
            )
        ),
        (
            "来源类型覆盖率："
            + format_percentage(
                result[
                    "source_type_coverage"
                ]
            )
        ),
        (
            "时间覆盖率："
            + format_percentage(
                result[
                    "time_coverage"
                ]
            )
        ),
        (
            "字段完整率："
            + format_percentage(
                result[
                    "field_completeness"
                ]
            )
        ),
        (
            "高价值文章比例："
            + format_percentage(
                result[
                    "high_value_rate"
                ]
            )
        ),
        (
            "前5来源占比："
            + format_percentage(
                result[
                    "top5_source_share"
                ]
            )
        ),
        (
            "缺失事件分类："
            + (
                "、".join(
                    result[
                        "missing_categories"
                    ]
                )
                or "无"
            )
        ),
        (
            "缺失技术路线："
            + (
                "、".join(
                    result[
                        "missing_technology_routes"
                    ]
                )
                or "无"
            )
        ),
        (
            "缺失来源类型："
            + (
                "、".join(
                    result[
                        "missing_source_types"
                    ]
                )
                or "无"
            )
        ),
    ]

    return "\n".join(lines)


def format_relative_report(
    result: dict,
) -> str:
    lines = [
        "相对事件覆盖率分析完成",
        (
            "基准事件数："
            f"{result['benchmark_event_count']}"
        ),
        (
            "已命中事件数："
            f"{result['matched_event_count']}"
        ),
        (
            "遗漏事件数："
            f"{result['missed_event_count']}"
        ),
        (
            "相对覆盖率："
            + format_percentage(
                result[
                    "relative_coverage_rate"
                ]
            )
        ),
        (
            "漏报率："
            + format_percentage(
                result[
                    "miss_rate"
                ]
            )
        ),
    ]

    if result["missed_events"]:
        lines.append("未命中的基准事件：")

        for item in result[
            "missed_events"
        ]:
            lines.append(
                "- "
                + item["event_title"]
                + "（最佳匹配分数："
                + f"{item['best_match_score']:.2f}"
                + "）"
            )

    return "\n".join(lines)


if __name__ == "__main__":
    template = create_benchmark_template()
    print(
        f"基准事件模板：{template.resolve()}"
    )

    internal = calculate_internal_coverage()
    save_report(internal)
    print(format_internal_report(internal))
