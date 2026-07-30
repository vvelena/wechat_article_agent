import csv
import tempfile
import unittest
import zipfile
from datetime import date
from pathlib import Path
from unittest.mock import patch

from app.core.result_schema import RESULT_FIELDS
from app.report import daily_word


def result_row(**overrides: str) -> dict[str, str]:
    row = {field: "" for field in RESULT_FIELDS}
    row.update(
        {
            "publish_time": "2026年7月30日 09:00",
            "title": "量子行业测试资讯",
            "account": "测试来源",
            "article_url": "https://example.com/article",
            "category": "技术研发",
            "summary": "这是一条用于验证日报生成的量子行业资讯摘要。",
            "relevance_score": "90",
            "quality_score": "80",
            "importance_score": "85",
            "source_reliability_score": "75",
            "is_promotional": "否",
            "collected_at": "2026-07-30 12:00:00",
        }
    )
    row.update(overrides)
    return row


def write_results(
    csv_file: Path,
    rows: list[dict[str, str]],
) -> None:
    with csv_file.open(
        mode="w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=RESULT_FIELDS,
        )
        writer.writeheader()
        writer.writerows(rows)


class DailyWordTests(unittest.TestCase):
    def test_selects_both_sources_and_deduplicates(self) -> None:
        rows = [
            result_row(source_kind="wechat"),
            result_row(
                source_kind="web",
                article_url="https://example.com/article",
                importance_score="60",
            ),
            result_row(
                source_kind="web",
                title="另一条量子资讯",
                article_url="https://example.com/second",
            ),
        ]

        selected, legacy_count = daily_word.select_daily_rows(
            rows,
            date(2026, 7, 30),
            max_items=4,
        )

        self.assertEqual(len(selected), 2)
        self.assertEqual(legacy_count, 0)
        self.assertEqual(
            {row["source_kind"] for row in selected},
            {"wechat", "web"},
        )

    def test_legacy_row_falls_back_to_publish_time(self) -> None:
        selected, legacy_count = daily_word.select_daily_rows(
            [result_row(collected_at="")],
            date(2026, 7, 30),
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(legacy_count, 1)

    def test_default_does_not_limit_item_count(self) -> None:
        rows = [
            result_row(
                title=f"量子资讯 {index}",
                article_url=f"https://example.com/{index}",
            )
            for index in range(6)
        ]

        selected, _ = daily_word.select_daily_rows(
            rows,
            date(2026, 7, 30),
        )

        self.assertEqual(len(selected), 6)

    def test_generates_readable_docx_with_hyperlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp_dir = Path(directory)
            wechat_file = temp_dir / "wechat_results.csv"
            web_file = temp_dir / "web_results.csv"
            reports_dir = temp_dir / "reports"
            write_results(wechat_file, [result_row()])
            write_results(web_file, [])

            with (
                patch.dict(
                    daily_word.RESULT_FILES,
                    {
                        "wechat": wechat_file,
                        "web": web_file,
                    },
                    clear=True,
                ),
                patch.object(
                    daily_word,
                    "REPORTS_DIR",
                    reports_dir,
                ),
            ):
                result = daily_word.generate_daily_word_report(
                    "2026-07-30"
                )

            self.assertTrue(result["success"])
            output_file = Path(result["output_file"])
            self.assertTrue(output_file.exists())

            with zipfile.ZipFile(output_file) as archive:
                document_xml = archive.read(
                    "word/document.xml"
                ).decode("utf-8")
                relationships_xml = archive.read(
                    "word/_rels/document.xml.rels"
                ).decode("utf-8")

            self.assertIn("量子科技行业资讯日报", document_xml)
            self.assertIn("量子行业测试资讯", document_xml)
            self.assertIn("2026年7月30日", document_xml)
            self.assertIn(
                "https://example.com/article",
                relationships_xml,
            )

    def test_no_data_does_not_create_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp_dir = Path(directory)
            wechat_file = temp_dir / "wechat_results.csv"
            web_file = temp_dir / "web_results.csv"
            reports_dir = temp_dir / "reports"
            write_results(wechat_file, [])
            write_results(web_file, [])

            with (
                patch.dict(
                    daily_word.RESULT_FILES,
                    {
                        "wechat": wechat_file,
                        "web": web_file,
                    },
                    clear=True,
                ),
                patch.object(
                    daily_word,
                    "REPORTS_DIR",
                    reports_dir,
                ),
            ):
                result = daily_word.generate_daily_word_report(
                    "2026-07-30"
                )

            self.assertEqual(result["status"], "no_data")
            self.assertFalse(reports_dir.exists())


if __name__ == "__main__":
    unittest.main()
