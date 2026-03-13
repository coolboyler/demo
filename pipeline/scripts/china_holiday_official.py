from __future__ import annotations

import datetime as dt
from collections.abc import Iterable

import pandas as pd


HOLIDAY_NAME_MAP = {
    "new_year": "New Year's Day",
    "spring_festival": "Spring Festival",
    "tomb_sweeping": "Tomb-sweeping Day",
    "labour_day": "Labour Day",
    "dragon_boat": "Dragon Boat Festival",
    "mid_autumn": "Mid-autumn Festival",
    "national_day": "National Day",
}


# Official holiday schedules sourced from State Council holiday notices:
# 2024: 国办发明电〔2023〕7号
# 2025: 国办发明电〔2024〕12号
# 2026: 国办发明电〔2025〕7号
HOLIDAY_RULES = {
    2024: {
        "holidays": [
            ("new_year", "2024-01-01", "2024-01-01"),
            ("spring_festival", "2024-02-10", "2024-02-17"),
            ("tomb_sweeping", "2024-04-04", "2024-04-06"),
            ("labour_day", "2024-05-01", "2024-05-05"),
            ("dragon_boat", "2024-06-10", "2024-06-10"),
            ("mid_autumn", "2024-09-15", "2024-09-17"),
            ("national_day", "2024-10-01", "2024-10-07"),
        ],
        "makeup_workdays": {
            "2024-02-04": "spring_festival",
            "2024-02-18": "spring_festival",
            "2024-04-07": "tomb_sweeping",
            "2024-04-28": "labour_day",
            "2024-05-11": "labour_day",
            "2024-09-14": "mid_autumn",
            "2024-09-29": "national_day",
            "2024-10-12": "national_day",
        },
    },
    2025: {
        "holidays": [
            ("new_year", "2025-01-01", "2025-01-01"),
            ("spring_festival", "2025-01-28", "2025-02-04"),
            ("tomb_sweeping", "2025-04-04", "2025-04-06"),
            ("labour_day", "2025-05-01", "2025-05-05"),
            ("dragon_boat", "2025-05-31", "2025-06-02"),
            ("national_day", "2025-10-01", "2025-10-05"),
            ("mid_autumn", "2025-10-06", "2025-10-06"),
            ("national_day", "2025-10-07", "2025-10-08"),
        ],
        "makeup_workdays": {
            "2025-01-26": "spring_festival",
            "2025-02-08": "spring_festival",
            "2025-04-27": "labour_day",
            "2025-09-28": "national_day",
            "2025-10-11": "national_day",
        },
    },
    2026: {
        "holidays": [
            ("new_year", "2026-01-01", "2026-01-03"),
            ("spring_festival", "2026-02-15", "2026-02-23"),
            ("tomb_sweeping", "2026-04-04", "2026-04-06"),
            ("labour_day", "2026-05-01", "2026-05-05"),
            ("dragon_boat", "2026-06-19", "2026-06-21"),
            ("mid_autumn", "2026-09-25", "2026-09-27"),
            ("national_day", "2026-10-01", "2026-10-07"),
        ],
        "makeup_workdays": {
            "2026-01-04": "new_year",
            "2026-02-14": "spring_festival",
            "2026-02-28": "spring_festival",
            "2026-05-09": "labour_day",
            "2026-09-20": "national_day",
            "2026-10-10": "national_day",
        },
    },
}


def _to_ts(value: str | dt.date | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def official_holiday_maps() -> tuple[dict[pd.Timestamp, str], dict[pd.Timestamp, str]]:
    holiday_map: dict[pd.Timestamp, str] = {}
    makeup_map: dict[pd.Timestamp, str] = {}
    for year_rules in HOLIDAY_RULES.values():
        for holiday_key, start_date, end_date in year_rules["holidays"]:
            holiday_name = HOLIDAY_NAME_MAP[holiday_key]
            for day in pd.date_range(start_date, end_date, freq="D"):
                holiday_map[_to_ts(day)] = holiday_name
        for date_str, holiday_key in year_rules["makeup_workdays"].items():
            makeup_map[_to_ts(date_str)] = HOLIDAY_NAME_MAP[holiday_key]
    return holiday_map, makeup_map


OFFICIAL_HOLIDAY_MAP, OFFICIAL_MAKEUP_MAP = official_holiday_maps()


def official_get_holiday_detail(date_value: dt.date | pd.Timestamp) -> tuple[bool, str | None]:
    date_ts = _to_ts(date_value)
    holiday_name = OFFICIAL_HOLIDAY_MAP.get(date_ts)
    if holiday_name is not None:
        return True, holiday_name

    makeup_holiday_name = OFFICIAL_MAKEUP_MAP.get(date_ts)
    if makeup_holiday_name is not None:
        return False, makeup_holiday_name

    if date_ts.dayofweek >= 5:
        return True, None
    return False, None


def official_is_workday(date_value: dt.date | pd.Timestamp) -> bool:
    date_ts = _to_ts(date_value)
    if date_ts in OFFICIAL_HOLIDAY_MAP:
        return False
    if date_ts in OFFICIAL_MAKEUP_MAP:
        return True
    return date_ts.dayofweek < 5


def build_official_holiday_frame(date_values: Iterable[dt.date | pd.Timestamp]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date_value in sorted({_to_ts(value) for value in date_values}):
        holiday_flag, holiday_name = official_get_holiday_detail(date_value)
        workday_flag = official_is_workday(date_value)
        rows.append(
            {
                "date": date_value,
                "is_holiday_cn": int(bool(holiday_flag and holiday_name)),
                "holiday_name_cn": holiday_name if holiday_flag and holiday_name else "non_holiday",
                "is_workday_cn": int(workday_flag),
                "is_makeup_workday": int(date_value.dayofweek >= 5 and workday_flag and date_value not in OFFICIAL_HOLIDAY_MAP),
            }
        )
    return pd.DataFrame(rows)
