from datetime import datetime, timedelta


def parse_wechat_publish_time(
    publish_time_text: str,
) -> datetime | None:
    """
    将微信公众号发布时间转换为 datetime。
    """

    text = str(publish_time_text).strip()

    if not text:
        return None

    formats = [
        "%Y年%m月%d日 %H:%M",
        "%Y年%m月%d日",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d",
    ]

    for date_format in formats:
        try:
            return datetime.strptime(
                text,
                date_format,
            )
        except ValueError:
            continue

    return None


def is_within_last_days(
    publish_time_text: str,
    days: int = 7,
) -> bool:
    """
    按自然日期判断文章是否在最近指定天数内。

    days <= 0 表示不启用时间限制。
    """

    if int(days) <= 0:
        return True

    publish_datetime = parse_wechat_publish_time(
        publish_time_text
    )

    if publish_datetime is None:
        return False

    today = datetime.now().date()
    earliest_date = today - timedelta(
        days=int(days) - 1
    )
    publish_date = publish_datetime.date()

    return earliest_date <= publish_date <= today
