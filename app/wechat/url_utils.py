from urllib.parse import (
    parse_qsl,
    urlencode,
    urlparse,
    urlunparse,
)


def normalize_wechat_url(url: str) -> str:
    """
    标准化微信公众号文章链接，同时保留访问文章所需参数。
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

        normalized = parsed._replace(
            scheme="https",
            netloc="mp.weixin.qq.com",
            query=urlencode(filtered_query),
            fragment="",
        )
        return urlunparse(normalized)

    return cleaned


def is_valid_wechat_url(url: str) -> bool:
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

    signed_keys = {
        "src",
        "timestamp",
        "ver",
        "signature",
    }

    return (
        standard_keys.issubset(
            query_params.keys()
        )
        or signed_keys.issubset(
            query_params.keys()
        )
    )


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

        if is_valid_wechat_url(
            normalized_url
        ):
            return normalized_url

    return ""
