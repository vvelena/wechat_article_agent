from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
LINKS_FILE = DATA_DIR / "links.csv"
WECHAT_RESULTS_FILE = DATA_DIR / "wechat_results.csv"
WEB_RESULTS_FILE = DATA_DIR / "web_results.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
