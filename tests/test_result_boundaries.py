import csv
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
import sys


agents_stub = ModuleType("agents")
agents_stub.function_tool = lambda function: function
sys.modules.setdefault("agents", agents_stub)

bs4_stub = ModuleType("bs4")
bs4_stub.BeautifulSoup = object
sys.modules.setdefault("bs4", bs4_stub)

from app.core.paths import DATA_DIR, PROJECT_ROOT
from app.core.result_schema import RESULT_FIELDS, ensure_csv_schema
from app.web.publish_time import is_within_requested_days
from app.web.result_store import validate_webpage_url
from app.wechat.result_store import validate_wechat_article_url


class ResultBoundaryTests(unittest.TestCase):
    def test_web_store_rejects_wechat_url(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "拒绝写入 web_results.csv",
        ):
            validate_webpage_url(
                "https://mp.weixin.qq.com/s/example"
            )

    def test_web_store_accepts_regular_https_url(self) -> None:
        url = "https://example.com/news/1"
        self.assertEqual(validate_webpage_url(url), url)

    def test_wechat_store_rejects_regular_web_url(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "拒绝写入 wechat_results.csv",
        ):
            validate_wechat_article_url(
                "https://example.com/news/1"
            )

    def test_wechat_store_normalizes_valid_url(self) -> None:
        self.assertEqual(
            validate_wechat_article_url(
                "http://mp.weixin.qq.com/s/example?from=test"
            ),
            "https://mp.weixin.qq.com/s/example",
        )

    def test_zero_days_disables_date_filter(self) -> None:
        self.assertTrue(
            is_within_requested_days(
                "未获取到发布时间",
                0,
            )
        )

    def test_result_schema_rejects_incompatible_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_file = Path(directory) / "results.csv"
            csv_file.write_text(
                "title,article_url\n",
                encoding="utf-8-sig",
            )

            with self.assertRaisesRegex(ValueError, "缺少字段"):
                ensure_csv_schema(csv_file, RESULT_FIELDS)

    def test_legacy_result_schema_adds_collected_at(self) -> None:
        legacy_fields = [
            field
            for field in RESULT_FIELDS
            if field != "collected_at"
        ]
        with tempfile.TemporaryDirectory() as directory:
            csv_file = Path(directory) / "results.csv"
            csv_file.write_text(
                ",".join(legacy_fields)
                + "\n"
                + ",".join(["value"] * len(legacy_fields))
                + "\n",
                encoding="utf-8-sig",
            )

            ensure_csv_schema(csv_file, RESULT_FIELDS)

            with csv_file.open(
                encoding="utf-8-sig",
                newline="",
            ) as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(
                list(rows[0]),
                RESULT_FIELDS,
            )
            self.assertEqual(rows[0]["collected_at"], "")

    def test_data_path_is_project_relative(self) -> None:
        self.assertEqual(DATA_DIR, PROJECT_ROOT / "data")
        self.assertTrue(DATA_DIR.is_absolute())


if __name__ == "__main__":
    unittest.main()
