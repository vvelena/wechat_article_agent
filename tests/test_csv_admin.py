import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.core import csv_admin
from app.core.result_schema import RESULT_FIELDS


def write_csv(
    csv_file: Path,
    fields: list[str],
    rows: list[list[str]],
) -> None:
    with csv_file.open(
        mode="w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(fields)
        writer.writerows(rows)


class CsvAdminTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        directory = Path(self.temporary_directory.name)
        self.links_file = directory / "links.csv"
        self.wechat_file = directory / "wechat_results.csv"
        self.web_file = directory / "web_results.csv"
        self.targets = {
            "links": (
                self.links_file,
                csv_admin.LINK_FIELDS,
            ),
            "wechat": (
                self.wechat_file,
                RESULT_FIELDS,
            ),
            "web": (
                self.web_file,
                RESULT_FIELDS,
            ),
        }

        write_csv(
            self.links_file,
            csv_admin.LINK_FIELDS,
            [["https://mp.weixin.qq.com/s/example"]],
        )
        result_row = ["value"] * len(RESULT_FIELDS)
        write_csv(
            self.wechat_file,
            RESULT_FIELDS,
            [result_row, result_row],
        )
        write_csv(
            self.web_file,
            RESULT_FIELDS,
            [result_row],
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_clear_one_csv_preserves_other_files(self) -> None:
        with patch.dict(
            csv_admin.CSV_TARGETS,
            self.targets,
            clear=True,
        ):
            result = csv_admin.clear_csv_data("微信")

        self.assertEqual(result["target"], "wechat")
        self.assertEqual(result["files"][0]["before_count"], 2)
        self.assertEqual(
            csv_admin.count_csv_rows(self.wechat_file),
            0,
        )
        self.assertEqual(
            csv_admin.count_csv_rows(self.links_file),
            1,
        )
        self.assertEqual(
            csv_admin.count_csv_rows(self.web_file),
            1,
        )

        with self.wechat_file.open(
            encoding="utf-8-sig",
            newline="",
        ) as file:
            self.assertEqual(
                next(csv.reader(file)),
                RESULT_FIELDS,
            )

    def test_clear_all_csv_files(self) -> None:
        with patch.dict(
            csv_admin.CSV_TARGETS,
            self.targets,
            clear=True,
        ):
            result = csv_admin.clear_csv_data("all")

        self.assertEqual(len(result["files"]), 3)
        for csv_file in (
            self.links_file,
            self.wechat_file,
            self.web_file,
        ):
            self.assertEqual(
                csv_admin.count_csv_rows(csv_file),
                0,
            )

    def test_invalid_target_does_not_change_files(self) -> None:
        with patch.dict(
            csv_admin.CSV_TARGETS,
            self.targets,
            clear=True,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "无效的清空目标",
            ):
                csv_admin.clear_csv_data(
                    "../../other.csv"
                )

        self.assertEqual(
            csv_admin.count_csv_rows(self.links_file),
            1,
        )
        self.assertEqual(
            csv_admin.count_csv_rows(self.wechat_file),
            2,
        )
        self.assertEqual(
            csv_admin.count_csv_rows(self.web_file),
            1,
        )


if __name__ == "__main__":
    unittest.main()
