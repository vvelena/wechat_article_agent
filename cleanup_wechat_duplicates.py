import re
import unicodedata
from pathlib import Path

import pandas as pd


DATA_FILE = Path(
    "data/wechat_results.csv"
)


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


def main() -> None:
    if not DATA_FILE.exists():
        print(
            f"没有找到文件：{DATA_FILE.resolve()}"
        )
        return

    df = pd.read_csv(
        DATA_FILE,
        encoding="utf-8-sig",
    )

    required = {
        "title",
        "account",
        "quality_score",
    }

    missing = required - set(df.columns)

    if missing:
        print(
            "CSV 缺少字段："
            + "、".join(sorted(missing))
        )
        return

    df["_dedup_key"] = (
        df["account"]
        .fillna("")
        .map(normalize_text)
        + "|"
        + df["title"]
        .fillna("")
        .map(normalize_text)
    )

    df["_quality"] = pd.to_numeric(
        df["quality_score"],
        errors="coerce",
    ).fillna(-1)

    # 同一公众号与标题重复时，
    # 优先保留 quality_score 最高的一条。
    sorted_df = df.sort_values(
        by="_quality",
        ascending=False,
        kind="stable",
    )

    cleaned_df = (
        sorted_df
        .drop_duplicates(
            subset=["_dedup_key"],
            keep="first",
        )
        .sort_index()
        .drop(
            columns=[
                "_dedup_key",
                "_quality",
            ]
        )
    )

    removed_count = (
        len(df)
        - len(cleaned_df)
    )

    backup_file = DATA_FILE.with_name(
        "wechat_results_before_dedup.csv"
    )

    df.drop(
        columns=[
            "_dedup_key",
            "_quality",
        ]
    ).to_csv(
        backup_file,
        index=False,
        encoding="utf-8-sig",
    )

    cleaned_df.to_csv(
        DATA_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 60)
    print("历史重复数据清理完成")
    print("=" * 60)
    print(f"原始记录：{len(df)}")
    print(f"删除重复：{removed_count}")
    print(f"保留记录：{len(cleaned_df)}")
    print(f"备份文件：{backup_file.resolve()}")
    print(f"结果文件：{DATA_FILE.resolve()}")


if __name__ == "__main__":
    main()
