import asyncio
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
)
from dotenv import load_dotenv
from openai import AsyncOpenAI


# ============================================================
# 1. 基础配置
# ============================================================

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not DEEPSEEK_API_KEY:
    raise ValueError(
        "没有找到 DEEPSEEK_API_KEY，请检查 .env 文件。"
    )

set_tracing_disabled(True)

DATA_FILE = Path("wechat_results.csv")
REPORTS_DIR = Path("reports")

REPORTS_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. 配置 DeepSeek
# ============================================================

deepseek_client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
)

deepseek_model = OpenAIChatCompletionsModel(
    model="deepseek-chat",
    openai_client=deepseek_client,
)


# ============================================================
# 3. 日报 Agent
# ============================================================

daily_report_agent = Agent(
    name="量子行业日报智能体",

    model=deepseek_model,

    instructions="""
你是量子行业资讯日报分析智能体。

你会收到当天收集到的量子行业文章数据。

你的任务不是逐篇重复摘要，
而是根据这些结构化文章数据生成一份量子行业日报。

请完成以下内容：

1. 统计今日资讯总数。
2. 总结今日主要资讯类别。
3. 选出最重要的3到5条新闻。
4. 识别今日高频出现的公司、科研机构或投资机构。
5. 总结今日主要技术路线。
6. 分析融资、合作、政策、技术研发和商业化动态。
7. 识别同一事件的重复报道。
8. 给出今日行业趋势判断。
9. 提出后续值得持续关注的事件。

要求：

- 所有内容必须基于输入数据。
- 不得编造文章中没有的信息。
- 明确区分“事实”和“分析判断”。
- 同一事件有多篇报道时，需要合并处理。
- 不要机械罗列所有文章。
- 使用中文。
- 结构清楚。
- 报告应适合行业研究和内部周报使用。

输出结构：

# 量子行业日报

## 一、今日概览

## 二、重点新闻

## 三、分类动态

## 四、重点公司与机构

## 五、技术路线观察

## 六、行业趋势判断

## 七、后续关注
""",
)


# ============================================================
# 4. 周报 Agent
# ============================================================

weekly_report_agent = Agent(
    name="量子行业周报智能体",

    model=deepseek_model,

    instructions="""
你是量子行业资讯周报分析智能体。

你会收到过去7天收集到的量子行业文章数据。

你的任务是识别这一周的行业变化、趋势、重要事件和持续事件，
而不是简单拼接每篇文章的摘要。

请完成以下内容：

1. 统计本周资讯总数。
2. 分析各类别新闻的数量和主要变化。
3. 选出本周最重要的5到10条新闻。
4. 总结本周融资与投资事件。
5. 总结本周合作、合同和商业订单。
6. 总结本周政策与政府项目。
7. 总结本周技术研发和科研进展。
8. 识别本周高频公司、机构和投资方。
9. 分析各量子技术路线的热度。
10. 分析商业化和应用落地情况。
11. 合并同一事件的重复报道。
12. 识别值得持续追踪的事件。
13. 给出下周值得关注的方向。

重点技术路线包括：

- 超导量子
- 离子阱
- 中性原子
- 光量子
- 硅量子点
- 拓扑量子
- 量子退火
- 相干伊辛机
- 量子软件
- 量子通信
- 量子测量
- 后量子密码

要求：

- 所有结论必须基于输入数据。
- 不得编造公司、金额、政策或技术进展。
- 明确区分事实和分析判断。
- 合并同一事件的重复报道。
- 重点分析趋势，而不是只罗列新闻。
- 使用中文。
- 结构清楚。
- 报告应适合行业研究和内部汇报使用。

输出结构：

# 量子行业周报

## 一、本周概览

## 二、核心事件

## 三、融资与投资

## 四、合作与商业订单

## 五、政策与政府项目

## 六、技术研发与科研进展

## 七、重点公司与机构

## 八、技术路线热度

## 九、行业趋势判断

## 十、下周关注
""",
)


# ============================================================
# 5. 读取 CSV
# ============================================================

def load_article_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "没有找到 wechat_results.csv，"
            "请先运行 wechat_agent.py 分析文章。"
        )

    df = pd.read_csv(DATA_FILE)

    if df.empty:
        raise ValueError("wechat_results.csv 中没有数据。")

    if "publish_time" not in df.columns:
        raise ValueError(
            "CSV 中缺少 publish_time 列。"
        )

    df["publish_time_parsed"] = pd.to_datetime(
        df["publish_time"],
        errors="coerce",
    )

    return df


# ============================================================
# 6. 把 DataFrame 转成给 Agent 阅读的文本
# ============================================================

def dataframe_to_text(df: pd.DataFrame) -> str:
    columns = [
        "title",
        "account",
        "publish_time",
        "category",
        "companies",
        "keywords",
        "summary",
        "importance",
        "reason",
        "article_url",
    ]

    available_columns = [
        column
        for column in columns
        if column in df.columns
    ]

    articles = []

    for index, row in df.iterrows():
        article_number = index + 1

        article_lines = [
            f"文章编号：{article_number}"
        ]

        for column in available_columns:
            value = row.get(column, "")

            if pd.isna(value):
                value = ""

            article_lines.append(
                f"{column}：{value}"
            )

        articles.append(
            "\n".join(article_lines)
        )

    return "\n\n--------------------\n\n".join(articles)


# ============================================================
# 7. 生成日报
# ============================================================

async def generate_daily_report(
    report_date: str,
) -> None:
    df = load_article_data()

    target_date = pd.to_datetime(
        report_date
    ).date()

    daily_df = df[
        df["publish_time_parsed"].dt.date
        == target_date
    ].copy()

    if daily_df.empty:
        print(
            f"{report_date} 没有找到文章数据。"
        )
        return

    article_text = dataframe_to_text(
        daily_df
    )

    task = f"""
请根据下面的数据，生成 {report_date} 的量子行业日报。

本次共有 {len(daily_df)} 篇文章。

文章数据如下：

{article_text}
"""

    result = await Runner.run(
        starting_agent=daily_report_agent,
        input=task,
        max_turns=4,
    )

    output_file = (
        REPORTS_DIR
        / f"daily_report_{report_date}.md"
    )

    output_file.write_text(
        result.final_output,
        encoding="utf-8",
    )

    print("\n日报生成成功：")
    print(output_file.resolve())


# ============================================================
# 8. 生成周报
# ============================================================

async def generate_weekly_report(
    end_date: str,
) -> None:
    df = load_article_data()

    week_end = pd.to_datetime(
        end_date
    ).date()

    week_start = week_end - timedelta(
        days=6
    )

    weekly_df = df[
        (
            df["publish_time_parsed"].dt.date
            >= week_start
        )
        &
        (
            df["publish_time_parsed"].dt.date
            <= week_end
        )
    ].copy()

    if weekly_df.empty:
        print(
            f"{week_start} 到 {week_end} "
            "没有找到文章数据。"
        )
        return

    article_text = dataframe_to_text(
        weekly_df
    )

    task = f"""
请根据下面的数据，生成量子行业周报。

统计时间：
{week_start} 至 {week_end}

本次共有 {len(weekly_df)} 篇文章。

文章数据如下：

{article_text}
"""

    result = await Runner.run(
        starting_agent=weekly_report_agent,
        input=task,
        max_turns=4,
    )

    output_file = (
        REPORTS_DIR
        / f"weekly_report_{week_start}_{week_end}.md"
    )

    output_file.write_text(
        result.final_output,
        encoding="utf-8",
    )

    print("\n周报生成成功：")
    print(output_file.resolve())


# ============================================================
# 9. 主程序
# ============================================================

async def main() -> None:
    print("=" * 60)
    print("量子行业日报 / 周报生成 Agent")
    print("=" * 60)

    print("\n请选择报告类型：")
    print("1. 日报")
    print("2. 周报")

    choice = input(
        "\n请输入 1 或 2："
    ).strip()

    if choice == "1":
        report_date = input(
            "请输入日报日期，例如 2026-07-21："
        ).strip()

        await generate_daily_report(
            report_date
        )

    elif choice == "2":
        end_date = input(
            "请输入周报结束日期，例如 2026-07-21："
        ).strip()

        await generate_weekly_report(
            end_date
        )

    else:
        print("输入错误，只能输入 1 或 2。")


if __name__ == "__main__":
    asyncio.run(main())