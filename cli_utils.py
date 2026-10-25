import argparse
from datetime import datetime


def parse_iso_date_arg(val: str) -> str:
    if not val:
        return ""
    try:
        dt = datetime.strptime(val, "%Y-%m-%d")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format '{val}'. Expected YYYY-MM-DD.")


def parse_history_arg(val: str) -> str:
    if not val or val.lower() == "all":
        return "all"
    try:
        dt = datetime.strptime(val, "%Y-%m-%d")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid history date '{val}'. Expected 'all' or 'YYYY-MM-DD'.")
