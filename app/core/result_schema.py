import csv
import os
import tempfile
from datetime import datetime, timedelta, timezone
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
    "collected_at",
]


def current_china_time_text() -> str:
    """返回固定 UTC+8 的采集时间，避免依赖系统时区。"""

    china_timezone = timezone(timedelta(hours=8))
    return datetime.now(china_timezone).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def migrate_legacy_result_schema(
    output_file: Path,
    existing_header: list[str],
    expected_fields: list[str],
) -> bool:
    """为旧结果表补充 collected_at 空列，保留全部原始数据。"""

    legacy_fields = [
        field
        for field in expected_fields
        if field != "collected_at"
    ]
    if (
        "collected_at" not in existing_header
        and existing_header == legacy_fields
        and expected_fields == RESULT_FIELDS
    ):
        with output_file.open(
            mode="r",
            newline="",
            encoding="utf-8-sig",
        ) as source:
            rows = list(csv.DictReader(source))

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                newline="",
                encoding="utf-8-sig",
                dir=output_file.parent,
                prefix=f".{output_file.stem}_schema_",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                writer = csv.DictWriter(
                    temporary_file,
                    fieldnames=expected_fields,
                )
                writer.writeheader()
                for row in rows:
                    row["collected_at"] = ""
                    writer.writerow(row)
                temporary_path = Path(temporary_file.name)

            os.replace(temporary_path, output_file)
            return True
        except Exception:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise

    return False


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
    if migrate_legacy_result_schema(
        output_file,
        existing_header,
        expected_fields,
    ):
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
