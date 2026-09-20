#!/usr/bin/env python3
"""Generate a pilot-jurisdiction-by-year policy panel from approval records."""

from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "ai_pilot_approvals.csv"
DEFAULT_OUTPUT = BASE_DIR / "ai_pilot_city_year_2010_2025.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2025)
    return parser.parse_args()


def load_approvals(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != 18:
        raise ValueError(f"Expected 18 confirmed pilot jurisdictions, found {len(rows)}")
    if len({row["pilot_id"] for row in rows}) != len(rows):
        raise ValueError("pilot_id must be unique")
    if len({row["scope_gb_code"] for row in rows}) != len(rows):
        raise ValueError("scope_gb_code must be unique")
    for row in rows:
        date.fromisoformat(row["approval_date"])
        if int(row["first_full_year"]) != int(row["approval_year"]) + 1:
            raise ValueError(f"Unexpected first_full_year for {row['pilot_id']}")
    return rows


def build_panel(
    approvals: list[dict[str, str]], start_year: int, end_year: int
) -> list[dict[str, object]]:
    if start_year > end_year:
        raise ValueError("start-year cannot exceed end-year")

    panel: list[dict[str, object]] = []
    for row in approvals:
        approval_year = int(row["approval_year"])
        first_full_year = int(row["first_full_year"])
        for year in range(start_year, end_year + 1):
            panel.append(
                {
                    "pilot_id": row["pilot_id"],
                    "cohort_id": row["cohort_id"],
                    "scope_level": row["scope_level"],
                    "scope_name": row["scope_name"],
                    "scope_gb_code": row["scope_gb_code"],
                    "province": row["province"],
                    "province_code": row["province_code"],
                    "prefecture": row["prefecture"],
                    "prefecture_code": row["prefecture_code"],
                    "county": row["county"],
                    "county_code": row["county_code"],
                    "year": year,
                    "approval_date": row["approval_date"],
                    "approval_year": approval_year,
                    "first_full_year": first_full_year,
                    "ever_treated": 1,
                    "post_approval_year": int(year >= approval_year),
                    "post_full_year": int(year >= first_full_year),
                    "event_time_approval_year": year - approval_year,
                    "event_time_full_year": year - first_full_year,
                }
            )
    return panel


def write_panel(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    approvals = load_approvals(args.input)
    panel = build_panel(approvals, args.start_year, args.end_year)
    write_panel(panel, args.output)
    print(
        f"Wrote {len(panel)} rows for {len(approvals)} pilot jurisdictions "
        f"({args.start_year}-{args.end_year}) to {args.output}"
    )


if __name__ == "__main__":
    main()
