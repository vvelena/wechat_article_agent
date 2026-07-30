import pandas as pd

from app.core.paths import LINKS_FILE
from app.wechat.url_utils import (
    is_valid_wechat_url,
    normalize_wechat_url,
)


def load_existing_links() -> set[str]:
    """
    读取 data/links.csv 中已有的有效微信链接。
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

        if is_valid_wechat_url(
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


def add_url_to_links_file(url: str) -> None:
    """
    将单篇链接追加到 data/links.csv，已有链接不重复写入。
    """

    normalized_url = normalize_wechat_url(url)

    if LINKS_FILE.exists():
        df = pd.read_csv(LINKS_FILE)

        if "article_url" not in df.columns:
            raise ValueError(
                "links.csv 中必须包含 article_url 列。"
            )

        existing_urls = {
            normalize_wechat_url(str(item))
            for item in df["article_url"].dropna()
        }

        if normalized_url in existing_urls:
            print("[链接记录] 该链接已存在于 links.csv。")
            return

        new_row = pd.DataFrame(
            {
                "article_url": [normalized_url]
            }
        )

        new_row.to_csv(
            LINKS_FILE,
            mode="a",
            header=False,
            index=False,
            encoding="utf-8-sig",
        )

    else:
        pd.DataFrame(
            {
                "article_url": [normalized_url]
            }
        ).to_csv(
            LINKS_FILE,
            index=False,
            encoding="utf-8-sig",
        )

    print("[链接记录] 已写入 data/links.csv。")


def load_article_urls() -> list[str]:
    """
    从 data/links.csv 读取、验证并去重微信文章链接。
    """

    if not LINKS_FILE.exists():
        raise FileNotFoundError(
            f"没有找到链接文件：{LINKS_FILE.resolve()}\n"
            "请创建 data/links.csv，并设置第一列列名为 article_url。"
        )

    df = pd.read_csv(LINKS_FILE)

    if "article_url" not in df.columns:
        raise ValueError(
            "links.csv 中必须包含 article_url 列。"
        )

    urls: list[str] = []
    seen: set[str] = set()

    for raw_url in df["article_url"].dropna():
        normalized_url = normalize_wechat_url(
            str(raw_url)
        )

        if not normalized_url:
            continue

        if not is_valid_wechat_url(normalized_url):
            print(
                f"[忽略无效链接] {raw_url}"
            )
            continue

        if normalized_url in seen:
            continue

        seen.add(normalized_url)
        urls.append(normalized_url)

    return urls
