import csv
from pathlib import Path


RESULT_FIELDS = [
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


def ensure_csv_schema(
    output_file: Path,
    expected_fields: list[str] = RESULT_FIELDS,
) -> None:
    """拒绝向表头不兼容的结果文件继续追加数据。"""

    if not output_file.exists():
        return

    if output_file.stat().st_size == 0:
        output_file.unlink()
        return

    with output_file.open(
        mode="r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        existing_header = next(csv.reader(file), [])

    if existing_header == expected_fields:
        return

    missing_fields = [
        field
        for field in expected_fields
        if field not in existing_header
    ]
    extra_fields = [
        field
        for field in existing_header
        if field not in expected_fields
    ]
    details = []

    if missing_fields:
        details.append("缺少字段：" + "、".join(missing_fields))
    if extra_fields:
        details.append("多余字段：" + "、".join(extra_fields))
    if not details:
        details.append("字段顺序不一致")

    raise ValueError(
        f"现有 {output_file} 的表头与当前代码不一致。\n"
        + "\n".join(details)
        + "\n请先备份并迁移旧 CSV，再重新运行程序。"
    )
