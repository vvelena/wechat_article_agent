import csv
import os
import tempfile
from pathlib import Path

from app.core.paths import (
    LINKS_FILE,
    WECHAT_RESULTS_FILE,
    WEB_RESULTS_FILE,
)
from app.core.result_schema import RESULT_FIELDS


LINK_FIELDS = ["article_url"]

CSV_TARGETS: dict[str, tuple[Path, list[str]]] = {
    "links": (LINKS_FILE, LINK_FIELDS),
    "wechat": (WECHAT_RESULTS_FILE, RESULT_FIELDS),
    "web": (WEB_RESULTS_FILE, RESULT_FIELDS),
}

TARGET_ALIASES = {
    "links": "links",
    "links.csv": "links",
    "链接": "links",
    "链接表": "links",
    "wechat": "wechat",
    "wechat_results": "wechat",
    "wechat_results.csv": "wechat",
    "微信": "wechat",
    "微信结果": "wechat",
    "微信公众号": "wechat",
    "web": "web",
    "web_results": "web",
    "web_results.csv": "web",
    "网页": "web",
    "网页结果": "web",
    "all": "all",
    "全部": "all",
    "三个": "all",
}


def normalize_clear_target(target: str) -> str:
    """把允许的中英文目标名称转换为固定内部名称。"""

    normalized = str(target or "").strip().lower()
    resolved = TARGET_ALIASES.get(normalized)

    if resolved is None:
        raise ValueError(
            "无效的清空目标。只允许：links、wechat、web 或 all。"
        )

    return resolved


def count_csv_rows(csv_file: Path) -> int:
    """统计 CSV 数据行；文件不存在或为空时返回 0。"""

    if not csv_file.exists() or csv_file.stat().st_size == 0:
        return 0

    with csv_file.open(
        mode="r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        return sum(1 for _ in csv.DictReader(file))


def clear_csv_file(
    csv_file: Path,
    fields: list[str],
) -> int:
    """原子清空一个 CSV，返回清空前的数据行数。"""

    before_count = count_csv_rows(csv_file)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8-sig",
            dir=csv_file.parent,
            prefix=f".{csv_file.stem}_",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            writer = csv.writer(temporary_file)
            writer.writerow(fields)
            temporary_path = Path(temporary_file.name)

        os.replace(temporary_path, csv_file)
        return before_count
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def clear_csv_data(target: str) -> dict:
    """清空白名单中的一个或全部 CSV，同时保留标准表头。"""

    resolved_target = normalize_clear_target(target)
    target_names = (
        list(CSV_TARGETS)
        if resolved_target == "all"
        else [resolved_target]
    )
    cleared_files = []

    for target_name in target_names:
        csv_file, fields = CSV_TARGETS[target_name]
        before_count = clear_csv_file(csv_file, fields)
        after_count = count_csv_rows(csv_file)

        if after_count != 0:
            raise RuntimeError(
                f"{csv_file.name} 清空后仍有 {after_count} 条记录。"
            )

        cleared_files.append(
            {
                "target": target_name,
                "file": csv_file.name,
                "before_count": before_count,
                "after_count": after_count,
            }
        )

    return {
        "success": True,
        "target": resolved_target,
        "files": cleared_files,
    }
