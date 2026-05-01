#!/usr/bin/env python
"""Generate synthetic credit-card transactions for TabFormer.

The output schema matches ``card_transaction.v1.csv`` as consumed by
``dataset/card.py``.
"""

import argparse
import bisect
import calendar
import csv
import datetime as dt
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple


COLUMNS = [
    "User",
    "Card",
    "Year",
    "Month",
    "Day",
    "Time",
    "Amount",
    "Use Chip",
    "Merchant Name",
    "Merchant City",
    "Merchant State",
    "Zip",
    "MCC",
    "Errors?",
    "Is Fraud?",
]


DEFAULT_FRAUD_RATE = 0.002
DEFAULT_AMOUNT_RESERVOIR_SIZE = 20_000


class WeightedTable:
    """Reusable weighted sampler for hot generation paths."""

    __slots__ = ("items", "cumulative_weights", "total_weight")

    def __init__(self, items: Sequence, weights: Sequence[float]):
        if len(items) != len(weights):
            raise ValueError("items and weights must have the same length")
        if not items:
            raise ValueError("weighted table cannot be empty")

        cumulative_weights: List[float] = []
        running_total = 0.0
        filtered_items = []
        for item, weight in zip(items, weights):
            if weight <= 0:
                continue
            running_total += float(weight)
            filtered_items.append(item)
            cumulative_weights.append(running_total)

        if not filtered_items:
            raise ValueError("weighted table must have at least one positive weight")

        self.items = tuple(filtered_items)
        self.cumulative_weights = tuple(cumulative_weights)
        self.total_weight = running_total

    @classmethod
    def from_counter(cls, counter: Counter) -> "WeightedTable":
        return cls(list(counter.keys()), list(counter.values()))

    def choice(self, rng: random.Random):
        threshold = rng.random() * self.total_weight
        index = bisect.bisect_left(self.cumulative_weights, threshold)
        if index >= len(self.items):
            index = len(self.items) - 1
        return self.items[index]


CITY_ZIP_BY_STATE: Dict[str, List[Tuple[str, str]]] = {
    "AZ": [("Phoenix", "85004"), ("Tucson", "85701"), ("Mesa", "85201")],
    "CA": [("Los Angeles", "90012"), ("San Diego", "92101"), ("San Jose", "95113"), ("Sacramento", "95814")],
    "CO": [("Denver", "80202"), ("Boulder", "80302"), ("Aurora", "80012")],
    "FL": [("Miami", "33130"), ("Orlando", "32801"), ("Tampa", "33602")],
    "GA": [("Atlanta", "30303"), ("Savannah", "31401"), ("Augusta", "30901")],
    "IL": [("Chicago", "60602"), ("Springfield", "62701"), ("Naperville", "60540")],
    "MA": [("Boston", "02108"), ("Cambridge", "02139"), ("Worcester", "01608")],
    "NC": [("Charlotte", "28202"), ("Raleigh", "27601"), ("Durham", "27701")],
    "NJ": [("Newark", "07102"), ("Jersey City", "07302"), ("Trenton", "08608")],
    "NY": [("New York", "10007"), ("Buffalo", "14202"), ("Albany", "12207")],
    "OH": [("Columbus", "43215"), ("Cleveland", "44114"), ("Cincinnati", "45202")],
    "PA": [("Philadelphia", "19107"), ("Pittsburgh", "15219"), ("Harrisburg", "17101")],
    "TX": [("Houston", "77002"), ("Austin", "78701"), ("Dallas", "75201"), ("San Antonio", "78205")],
    "WA": [("Seattle", "98101"), ("Spokane", "99201"), ("Tacoma", "98402")],
}


STATE_WEIGHTS = {
    "CA": 12,
    "TX": 10,
    "NY": 8,
    "FL": 7,
    "IL": 5,
    "PA": 5,
    "OH": 4,
    "GA": 4,
    "NC": 4,
    "NJ": 4,
    "WA": 3,
    "AZ": 3,
    "MA": 3,
    "CO": 3,
}


MCC_PROFILES = [
    {"mcc": "5411", "weight": 18, "fraud_weight": 8, "median": 42.0, "sigma": 0.55},
    {"mcc": "5812", "weight": 12, "fraud_weight": 7, "median": 34.0, "sigma": 0.60},
    {"mcc": "5541", "weight": 10, "fraud_weight": 5, "median": 48.0, "sigma": 0.50},
    {"mcc": "5311", "weight": 8, "fraud_weight": 11, "median": 61.0, "sigma": 0.75},
    {"mcc": "5912", "weight": 7, "fraud_weight": 5, "median": 28.0, "sigma": 0.45},
    {"mcc": "5999", "weight": 7, "fraud_weight": 13, "median": 76.0, "sigma": 0.80},
    {"mcc": "4121", "weight": 6, "fraud_weight": 8, "median": 23.0, "sigma": 0.70},
    {"mcc": "5651", "weight": 6, "fraud_weight": 12, "median": 85.0, "sigma": 0.70},
    {"mcc": "5732", "weight": 5, "fraud_weight": 15, "median": 140.0, "sigma": 0.85},
    {"mcc": "5300", "weight": 5, "fraud_weight": 9, "median": 72.0, "sigma": 0.65},
    {"mcc": "4511", "weight": 4, "fraud_weight": 14, "median": 260.0, "sigma": 0.85},
    {"mcc": "4722", "weight": 3, "fraud_weight": 10, "median": 210.0, "sigma": 0.90},
    {"mcc": "7011", "weight": 3, "fraud_weight": 13, "median": 180.0, "sigma": 0.75},
    {"mcc": "7399", "weight": 3, "fraud_weight": 8, "median": 95.0, "sigma": 0.90},
    {"mcc": "7995", "weight": 1, "fraud_weight": 18, "median": 120.0, "sigma": 0.95},
]


ERROR_VALUES = [
    "Bad PIN",
    "Bad CVV",
    "Bad Card Number",
    "Bad Expiration",
    "Insufficient Balance",
    "Technical Glitch",
]


@dataclass(frozen=True)
class UserProfile:
    user_id: int
    home_city: str
    home_state: str
    home_zip: str
    cards: Tuple[int, ...]
    spend_scale: float
    online_bias: float
    travel_rate: float
    year_table: Optional[WeightedTable] = None
    card_table: Optional[WeightedTable] = None


@dataclass(frozen=True)
class MerchantProfile:
    merchant_id: str
    city: str
    state: str
    zip_code: str
    mcc: str
    amount_median: float
    amount_sigma: float
    activity_weight: int = 1


@dataclass(frozen=True)
class SourceUserProfile:
    source_user_id: str
    transaction_count: int
    home_city: str
    home_state: str
    home_zip: str
    year_counts: Tuple[Tuple[int, int], ...]
    card_counts: Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class MerchantTemplate:
    city: str
    state: str
    zip_code: str
    mcc: str
    activity_weight: int


@dataclass(frozen=True)
class EmpiricalProfile:
    source_path: Path
    row_count: int
    fraud_rate: float
    user_profiles: Tuple[SourceUserProfile, ...]
    merchant_templates: Tuple[MerchantTemplate, ...]
    year_counts: Counter
    fraud_counts_by_year: Dict[int, Counter]
    month_counts_by_year: Dict[int, Counter]
    day_counts_by_year_month: Dict[Tuple[int, int], Counter]
    hour_counts_by_year: Dict[int, Counter]
    minute_counts_by_hour: Dict[int, Counter]
    mcc_counts_by_fraud: Dict[bool, Counter]
    use_chip_counts_by_year_fraud: Dict[Tuple[int, bool], Counter]
    error_counts_by_fraud_chip: Dict[Tuple[bool, str], Counter]
    amount_samples_by_mcc_fraud: Dict[Tuple[str, bool], Tuple[float, ...]]
    amount_samples_by_mcc: Dict[str, Tuple[float, ...]]
    amount_median_by_mcc: Dict[str, float]
    location_counts_by_mcc_chip: Dict[Tuple[str, str], Counter]
    location_counts_by_mcc: Dict[str, Counter]


@dataclass(frozen=True)
class MerchantIndexes:
    by_state_mcc: DefaultDict[Tuple[str, str], List[MerchantProfile]]
    by_mcc: DefaultDict[str, List[MerchantProfile]]
    all_merchants: List[MerchantProfile]
    weighted_by_state_mcc: Dict[Tuple[str, str], WeightedTable]
    weighted_by_mcc: Dict[str, WeightedTable]
    weighted_all: Optional[WeightedTable]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def probability(value: str) -> float:
    parsed = float(value)
    if parsed < 0 or parsed > 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def weighted_choice(rng: random.Random, items: Sequence, weights: Sequence[float]):
    return rng.choices(items, weights=weights, k=1)[0]


def weighted_counter_choice(rng: random.Random, counter: Counter):
    return WeightedTable.from_counter(counter).choice(rng)


def parse_amount(value: str) -> Optional[float]:
    cleaned = str(value).strip().replace("$", "").replace(",", "")
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_time(value: str) -> Optional[Tuple[int, int]]:
    match = re.match(r"^(\d{1,2}):(\d{2})$", str(value).strip())
    if not match:
        return None
    hour, minute = int(match.group(1)), int(match.group(2))
    if hour > 23 or minute > 59:
        return None
    return hour, minute


def add_reservoir_sample(
    rng: random.Random,
    samples_by_key: DefaultDict,
    seen_by_key: Counter,
    key,
    value: float,
    max_samples: int,
) -> None:
    if max_samples <= 0:
        return

    seen_by_key[key] += 1
    seen = seen_by_key[key]
    samples = samples_by_key[key]
    if len(samples) < max_samples:
        samples.append(value)
        return

    replacement_index = rng.randrange(seen)
    if replacement_index < max_samples:
        samples[replacement_index] = value


def median(values: Sequence[float]) -> float:
    if not values:
        return 30.0
    sorted_values = sorted(values)
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return float(sorted_values[midpoint])
    return float((sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2.0)


def weighted_state(rng: random.Random) -> str:
    states = list(STATE_WEIGHTS.keys())
    weights = [STATE_WEIGHTS[state] for state in states]
    return weighted_choice(rng, states, weights)


def random_city(rng: random.Random, state: Optional[str] = None) -> Tuple[str, str, str]:
    selected_state = state or weighted_state(rng)
    city, zip_code = rng.choice(CITY_ZIP_BY_STATE[selected_state])
    return city, selected_state, zip_code


def make_profiles(
    rng: random.Random,
    n_users: int,
    n_merchants: int,
    cards_per_user_min: int,
    cards_per_user_max: int,
) -> Tuple[List[UserProfile], List[MerchantProfile]]:
    users: List[UserProfile] = []
    for user_id in range(n_users):
        city, state, zip_code = random_city(rng)
        card_count = rng.randint(cards_per_user_min, cards_per_user_max)
        users.append(
            UserProfile(
                user_id=user_id,
                home_city=city,
                home_state=state,
                home_zip=zip_code,
                cards=tuple(range(card_count)),
                spend_scale=min(4.0, max(0.35, rng.lognormvariate(0.0, 0.45))),
                online_bias=min(0.70, max(0.05, rng.betavariate(2.0, 7.0))),
                travel_rate=min(0.25, max(0.01, rng.betavariate(1.5, 18.0))),
            )
        )

    mccs = [profile["mcc"] for profile in MCC_PROFILES]
    mcc_weights = [profile["weight"] for profile in MCC_PROFILES]
    mcc_by_code = {profile["mcc"]: profile for profile in MCC_PROFILES}

    merchants: List[MerchantProfile] = []
    seen_ids = set()
    while len(merchants) < n_merchants:
        mcc = weighted_choice(rng, mccs, mcc_weights)
        mcc_profile = mcc_by_code[mcc]
        city, state, zip_code = random_city(rng)
        merchant_id = str(rng.randint(-(2**63), 2**63 - 1))
        if merchant_id in seen_ids:
            continue
        seen_ids.add(merchant_id)
        merchants.append(
            MerchantProfile(
                merchant_id=merchant_id,
                city=city,
                state=state,
                zip_code=zip_code,
                mcc=mcc,
                amount_median=float(mcc_profile["median"]),
                amount_sigma=float(mcc_profile["sigma"]),
            )
        )

    return users, merchants


def load_empirical_profile_streaming(
    csv_path: Path,
    amount_reservoir_size: int,
    quiet: bool,
    progress_interval: int,
) -> EmpiricalProfile:
    """Build count tables from the reference TabFormer CSV in one streaming pass."""

    if not csv_path.exists():
        raise ValueError(f"profile CSV not found: {csv_path}")

    profile_rng = random.Random(0)
    row_count = 0
    fraud_count = 0

    year_counts: Counter = Counter()
    fraud_counts_by_year: DefaultDict[int, Counter] = defaultdict(Counter)
    month_counts_by_year: DefaultDict[int, Counter] = defaultdict(Counter)
    day_counts_by_year_month: DefaultDict[Tuple[int, int], Counter] = defaultdict(Counter)
    hour_counts_by_year: DefaultDict[int, Counter] = defaultdict(Counter)
    minute_counts_by_hour: DefaultDict[int, Counter] = defaultdict(Counter)

    user_year_counts: DefaultDict[str, Counter] = defaultdict(Counter)
    user_card_counts: DefaultDict[str, Counter] = defaultdict(Counter)
    user_location_counts: DefaultDict[str, Counter] = defaultdict(Counter)

    merchant_counts: Counter = Counter()
    mcc_counts_by_fraud: DefaultDict[bool, Counter] = defaultdict(Counter)
    use_chip_counts_by_year_fraud: DefaultDict[Tuple[int, bool], Counter] = defaultdict(Counter)
    error_counts_by_fraud_chip: DefaultDict[Tuple[bool, str], Counter] = defaultdict(Counter)
    location_counts_by_mcc_chip: DefaultDict[Tuple[str, str], Counter] = defaultdict(Counter)
    location_counts_by_mcc: DefaultDict[str, Counter] = defaultdict(Counter)

    amount_samples_by_mcc_fraud: DefaultDict[Tuple[str, bool], List[float]] = defaultdict(list)
    amount_seen_by_mcc_fraud: Counter = Counter()
    amount_samples_by_mcc: DefaultDict[str, List[float]] = defaultdict(list)
    amount_seen_by_mcc: Counter = Counter()

    if not quiet:
        print(f"building empirical profile from {csv_path}", file=sys.stderr)

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row_count, row in enumerate(reader, start=1):
            try:
                year = int(row["Year"])
                month = int(row["Month"])
                day = int(row["Day"])
            except (KeyError, TypeError, ValueError):
                continue

            time_parts = parse_time(row.get("Time", ""))
            if time_parts is None:
                hour, minute = 0, 0
            else:
                hour, minute = time_parts

            user_id = str(row.get("User", ""))
            card_value = row.get("Card", "0")
            try:
                card = int(card_value)
            except ValueError:
                card = 0

            fraud = row.get("Is Fraud?", "") == "Yes"
            if fraud:
                fraud_count += 1

            use_chip = row.get("Use Chip", "")
            errors = row.get("Errors?", "")
            city = row.get("Merchant City", "")
            state = row.get("Merchant State", "")
            zip_code = row.get("Zip", "")
            mcc = str(row.get("MCC", ""))
            merchant_id = str(row.get("Merchant Name", ""))
            location = (city, state, zip_code)

            year_counts[year] += 1
            fraud_counts_by_year[year][fraud] += 1
            month_counts_by_year[year][month] += 1
            day_counts_by_year_month[(year, month)][day] += 1
            hour_counts_by_year[year][hour] += 1
            minute_counts_by_hour[hour][minute] += 1

            user_year_counts[user_id][year] += 1
            user_card_counts[user_id][card] += 1
            if city or state or zip_code:
                user_location_counts[user_id][location] += 1

            merchant_counts[(merchant_id, city, state, zip_code, mcc)] += 1
            mcc_counts_by_fraud[fraud][mcc] += 1
            use_chip_counts_by_year_fraud[(year, fraud)][use_chip] += 1
            error_counts_by_fraud_chip[(fraud, use_chip)][errors] += 1
            location_counts_by_mcc_chip[(mcc, use_chip)][location] += 1
            location_counts_by_mcc[mcc][location] += 1

            amount = parse_amount(row.get("Amount", ""))
            if amount is not None:
                add_reservoir_sample(
                    profile_rng,
                    amount_samples_by_mcc_fraud,
                    amount_seen_by_mcc_fraud,
                    (mcc, fraud),
                    amount,
                    amount_reservoir_size,
                )
                add_reservoir_sample(
                    profile_rng,
                    amount_samples_by_mcc,
                    amount_seen_by_mcc,
                    mcc,
                    amount,
                    amount_reservoir_size,
                )

            if not quiet and progress_interval > 0 and row_count % progress_interval == 0:
                print(f"profiled {row_count:,} rows", file=sys.stderr)

    if row_count == 0:
        raise ValueError(f"profile CSV is empty: {csv_path}")

    user_profiles: List[SourceUserProfile] = []
    for source_user_id, per_year_counts in user_year_counts.items():
        if user_location_counts[source_user_id]:
            home_city, home_state, home_zip = user_location_counts[source_user_id].most_common(1)[0][0]
        else:
            home_city, home_state, home_zip = "", "", ""
        card_counts = tuple(sorted(user_card_counts[source_user_id].items()))
        year_count_items = tuple(sorted(per_year_counts.items()))
        user_profiles.append(
            SourceUserProfile(
                source_user_id=source_user_id,
                transaction_count=sum(per_year_counts.values()),
                home_city=home_city,
                home_state=home_state,
                home_zip=home_zip,
                year_counts=year_count_items,
                card_counts=card_counts,
            )
        )

    user_profiles.sort(
        key=lambda profile: (
            0,
            int(profile.source_user_id),
        )
        if profile.source_user_id.isdigit()
        else (1, profile.source_user_id)
    )

    merchant_templates = tuple(
        MerchantTemplate(
            city=city,
            state=state,
            zip_code=zip_code,
            mcc=mcc,
            activity_weight=count,
        )
        for (_merchant_id, city, state, zip_code, mcc), count in merchant_counts.items()
    )

    amount_samples_by_mcc_fraud_tuple = {
        key: tuple(values) for key, values in amount_samples_by_mcc_fraud.items()
    }
    amount_samples_by_mcc_tuple = {key: tuple(values) for key, values in amount_samples_by_mcc.items()}
    amount_median_by_mcc = {
        mcc: median(values) for mcc, values in amount_samples_by_mcc_tuple.items()
    }

    fraud_rate = fraud_count / row_count
    if not quiet:
        print(
            "empirical profile ready: "
            f"{row_count:,} rows, {len(user_profiles):,} users, "
            f"{len(merchant_templates):,} merchant templates, fraud {fraud_rate:.4%}",
            file=sys.stderr,
        )

    return EmpiricalProfile(
        source_path=csv_path,
        row_count=row_count,
        fraud_rate=fraud_rate,
        user_profiles=tuple(user_profiles),
        merchant_templates=merchant_templates,
        year_counts=year_counts,
        fraud_counts_by_year=dict(fraud_counts_by_year),
        month_counts_by_year=dict(month_counts_by_year),
        day_counts_by_year_month=dict(day_counts_by_year_month),
        hour_counts_by_year=dict(hour_counts_by_year),
        minute_counts_by_hour=dict(minute_counts_by_hour),
        mcc_counts_by_fraud=dict(mcc_counts_by_fraud),
        use_chip_counts_by_year_fraud=dict(use_chip_counts_by_year_fraud),
        error_counts_by_fraud_chip=dict(error_counts_by_fraud_chip),
        amount_samples_by_mcc_fraud=amount_samples_by_mcc_fraud_tuple,
        amount_samples_by_mcc=amount_samples_by_mcc_tuple,
        amount_median_by_mcc=amount_median_by_mcc,
        location_counts_by_mcc_chip=dict(location_counts_by_mcc_chip),
        location_counts_by_mcc=dict(location_counts_by_mcc),
    )


def scan_csv_polars(pl, csv_path: Path):
    schema_overrides = {
        "User": pl.Utf8,
        "Card": pl.Int64,
        "Year": pl.Int64,
        "Month": pl.Int64,
        "Day": pl.Int64,
        "Time": pl.Utf8,
        "Amount": pl.Utf8,
        "Use Chip": pl.Utf8,
        "Merchant Name": pl.Utf8,
        "Merchant City": pl.Utf8,
        "Merchant State": pl.Utf8,
        "Zip": pl.Utf8,
        "MCC": pl.Utf8,
        "Errors?": pl.Utf8,
        "Is Fraud?": pl.Utf8,
    }
    try:
        return pl.scan_csv(
            str(csv_path),
            infer_schema_length=1000,
            schema_overrides=schema_overrides,
        )
    except TypeError:
        return pl.scan_csv(
            str(csv_path),
            infer_schema_length=1000,
            dtypes=schema_overrides,
        )


def series_values(values) -> Tuple[float, ...]:
    if hasattr(values, "to_list"):
        values = values.to_list()
    return tuple(float(value) for value in values if value is not None)


def load_empirical_profile_polars(
    csv_path: Path,
    amount_reservoir_size: int,
    quiet: bool,
) -> EmpiricalProfile:
    import polars as pl

    if not quiet:
        print(f"building empirical profile from {csv_path} with polars", file=sys.stderr)

    lf = scan_csv_polars(pl, csv_path)
    base = lf.with_columns(
        [
            pl.col("User").cast(pl.Utf8).fill_null("").alias("_user"),
            pl.col("Card").cast(pl.Int64, strict=False).fill_null(0).alias("_card"),
            pl.col("Year").cast(pl.Int64, strict=False).alias("_year"),
            pl.col("Month").cast(pl.Int64, strict=False).alias("_month"),
            pl.col("Day").cast(pl.Int64, strict=False).alias("_day"),
            pl.col("Time").cast(pl.Utf8).fill_null("").str.slice(0, 2).cast(pl.Int64, strict=False).fill_null(0).alias("_hour"),
            pl.col("Time").cast(pl.Utf8).fill_null("").str.slice(3, 2).cast(pl.Int64, strict=False).fill_null(0).alias("_minute"),
            pl.col("Amount")
            .cast(pl.Utf8)
            .str.replace_all(r"[$,]", "")
            .cast(pl.Float64, strict=False)
            .alias("_amount"),
            (pl.col("Is Fraud?") == "Yes").fill_null(False).alias("_fraud"),
            pl.col("Use Chip").cast(pl.Utf8).fill_null("").alias("_use_chip"),
            pl.col("Errors?").cast(pl.Utf8).fill_null("").alias("_errors"),
            pl.col("Merchant Name").cast(pl.Utf8).fill_null("").alias("_merchant"),
            pl.col("Merchant City").cast(pl.Utf8).fill_null("").alias("_city"),
            pl.col("Merchant State").cast(pl.Utf8).fill_null("").alias("_state"),
            pl.col("Zip").cast(pl.Utf8).fill_null("").alias("_zip"),
            pl.col("MCC").cast(pl.Utf8).fill_null("").alias("_mcc"),
        ]
    ).select(
        [
            "_user",
            "_card",
            "_year",
            "_month",
            "_day",
            "_hour",
            "_minute",
            "_amount",
            "_fraud",
            "_use_chip",
            "_errors",
            "_merchant",
            "_city",
            "_state",
            "_zip",
            "_mcc",
        ]
    )

    summary = base.select(
        [
            pl.len().alias("_rows"),
            pl.col("_fraud").sum().alias("_fraud_count"),
        ]
    ).collect()
    row_count = int(summary["_rows"][0])
    fraud_count = int(summary["_fraud_count"][0])
    if row_count == 0:
        raise ValueError(f"profile CSV is empty: {csv_path}")

    year_counts: Counter = Counter()
    fraud_counts_by_year: DefaultDict[int, Counter] = defaultdict(Counter)
    year_fraud_df = base.group_by(["_year", "_fraud"]).agg(pl.len().alias("_count")).collect()
    for row in year_fraud_df.iter_rows(named=True):
        year = int(row["_year"])
        fraud = bool(row["_fraud"])
        count = int(row["_count"])
        year_counts[year] += count
        fraud_counts_by_year[year][fraud] = count

    month_counts_by_year: DefaultDict[int, Counter] = defaultdict(Counter)
    month_df = base.group_by(["_year", "_month"]).agg(pl.len().alias("_count")).collect()
    for row in month_df.iter_rows(named=True):
        month_counts_by_year[int(row["_year"])][int(row["_month"])] = int(row["_count"])

    day_counts_by_year_month: DefaultDict[Tuple[int, int], Counter] = defaultdict(Counter)
    day_df = base.group_by(["_year", "_month", "_day"]).agg(pl.len().alias("_count")).collect()
    for row in day_df.iter_rows(named=True):
        day_counts_by_year_month[(int(row["_year"]), int(row["_month"]))][int(row["_day"])] = int(row["_count"])

    hour_counts_by_year: DefaultDict[int, Counter] = defaultdict(Counter)
    hour_df = base.group_by(["_year", "_hour"]).agg(pl.len().alias("_count")).collect()
    for row in hour_df.iter_rows(named=True):
        hour_counts_by_year[int(row["_year"])][int(row["_hour"])] = int(row["_count"])

    minute_counts_by_hour: DefaultDict[int, Counter] = defaultdict(Counter)
    minute_df = base.group_by(["_hour", "_minute"]).agg(pl.len().alias("_count")).collect()
    for row in minute_df.iter_rows(named=True):
        minute_counts_by_hour[int(row["_hour"])][int(row["_minute"])] = int(row["_count"])

    user_year_counts: DefaultDict[str, Counter] = defaultdict(Counter)
    user_year_df = base.group_by(["_user", "_year"]).agg(pl.len().alias("_count")).collect()
    for row in user_year_df.iter_rows(named=True):
        user_year_counts[str(row["_user"])][int(row["_year"])] = int(row["_count"])

    user_card_counts: DefaultDict[str, Counter] = defaultdict(Counter)
    user_card_df = base.group_by(["_user", "_card"]).agg(pl.len().alias("_count")).collect()
    for row in user_card_df.iter_rows(named=True):
        user_card_counts[str(row["_user"])][int(row["_card"])] = int(row["_count"])

    user_home: Dict[str, Tuple[str, str, str]] = {}
    user_location_counts = base.filter(
        (pl.col("_city") != "") | (pl.col("_state") != "") | (pl.col("_zip") != "")
    ).group_by(["_user", "_city", "_state", "_zip"]).agg(pl.len().alias("_count"))
    user_location_df = user_location_counts.sort(
        ["_user", "_count"],
        descending=[False, True],
    ).unique(subset=["_user"], keep="first").collect()
    for row in user_location_df.iter_rows(named=True):
        user_home[str(row["_user"])] = (str(row["_city"]), str(row["_state"]), str(row["_zip"]))

    user_profiles: List[SourceUserProfile] = []
    for source_user_id, per_year_counts in user_year_counts.items():
        home_city, home_state, home_zip = user_home.get(source_user_id, ("", "", ""))
        user_profiles.append(
            SourceUserProfile(
                source_user_id=source_user_id,
                transaction_count=sum(per_year_counts.values()),
                home_city=home_city,
                home_state=home_state,
                home_zip=home_zip,
                year_counts=tuple(sorted(per_year_counts.items())),
                card_counts=tuple(sorted(user_card_counts[source_user_id].items())),
            )
        )
    user_profiles.sort(
        key=lambda profile: (
            0,
            int(profile.source_user_id),
        )
        if profile.source_user_id.isdigit()
        else (1, profile.source_user_id)
    )

    merchant_df = base.group_by(["_merchant", "_city", "_state", "_zip", "_mcc"]).agg(
        pl.len().alias("_count")
    ).collect()
    merchant_templates = tuple(
        MerchantTemplate(
            city=str(row["_city"]),
            state=str(row["_state"]),
            zip_code=str(row["_zip"]),
            mcc=str(row["_mcc"]),
            activity_weight=int(row["_count"]),
        )
        for row in merchant_df.iter_rows(named=True)
    )

    mcc_counts_by_fraud: DefaultDict[bool, Counter] = defaultdict(Counter)
    mcc_df = base.group_by(["_mcc", "_fraud"]).agg(pl.len().alias("_count")).collect()
    for row in mcc_df.iter_rows(named=True):
        mcc_counts_by_fraud[bool(row["_fraud"])][str(row["_mcc"])] = int(row["_count"])

    use_chip_counts_by_year_fraud: DefaultDict[Tuple[int, bool], Counter] = defaultdict(Counter)
    chip_df = base.group_by(["_year", "_fraud", "_use_chip"]).agg(pl.len().alias("_count")).collect()
    for row in chip_df.iter_rows(named=True):
        use_chip_counts_by_year_fraud[(int(row["_year"]), bool(row["_fraud"]))][str(row["_use_chip"])] = int(row["_count"])

    error_counts_by_fraud_chip: DefaultDict[Tuple[bool, str], Counter] = defaultdict(Counter)
    error_df = base.group_by(["_fraud", "_use_chip", "_errors"]).agg(pl.len().alias("_count")).collect()
    for row in error_df.iter_rows(named=True):
        error_counts_by_fraud_chip[(bool(row["_fraud"]), str(row["_use_chip"]))][str(row["_errors"])] = int(row["_count"])

    location_counts_by_mcc_chip: DefaultDict[Tuple[str, str], Counter] = defaultdict(Counter)
    location_chip_df = base.group_by(["_mcc", "_use_chip", "_city", "_state", "_zip"]).agg(
        pl.len().alias("_count")
    ).collect()
    for row in location_chip_df.iter_rows(named=True):
        location_counts_by_mcc_chip[(str(row["_mcc"]), str(row["_use_chip"]))][
            (str(row["_city"]), str(row["_state"]), str(row["_zip"]))
        ] = int(row["_count"])

    location_counts_by_mcc: DefaultDict[str, Counter] = defaultdict(Counter)
    location_df = base.group_by(["_mcc", "_city", "_state", "_zip"]).agg(pl.len().alias("_count")).collect()
    for row in location_df.iter_rows(named=True):
        location_counts_by_mcc[str(row["_mcc"])][
            (str(row["_city"]), str(row["_state"]), str(row["_zip"]))
        ] = int(row["_count"])

    amount_lf = base.filter(pl.col("_amount").is_not_null())
    amount_samples_by_mcc_fraud: Dict[Tuple[str, bool], Tuple[float, ...]] = {}
    amount_fraud_df = amount_lf.group_by(["_mcc", "_fraud"]).agg(
        pl.col("_amount")
        .sample(n=amount_reservoir_size, with_replacement=True, seed=0)
        .alias("_samples")
    ).collect()
    for row in amount_fraud_df.iter_rows(named=True):
        amount_samples_by_mcc_fraud[(str(row["_mcc"]), bool(row["_fraud"]))] = series_values(row["_samples"])

    amount_samples_by_mcc: Dict[str, Tuple[float, ...]] = {}
    amount_mcc_df = amount_lf.group_by(["_mcc"]).agg(
        pl.col("_amount")
        .sample(n=amount_reservoir_size, with_replacement=True, seed=1)
        .alias("_samples")
    ).collect()
    for row in amount_mcc_df.iter_rows(named=True):
        amount_samples_by_mcc[str(row["_mcc"])] = series_values(row["_samples"])

    amount_median_by_mcc: Dict[str, float] = {}
    amount_median_df = amount_lf.group_by(["_mcc"]).agg(pl.col("_amount").median().alias("_median")).collect()
    for row in amount_median_df.iter_rows(named=True):
        if row["_median"] is not None:
            amount_median_by_mcc[str(row["_mcc"])] = float(row["_median"])

    fraud_rate = fraud_count / row_count
    if not quiet:
        print(
            "empirical profile ready: "
            f"{row_count:,} rows, {len(user_profiles):,} users, "
            f"{len(merchant_templates):,} merchant templates, fraud {fraud_rate:.4%}",
            file=sys.stderr,
        )

    return EmpiricalProfile(
        source_path=csv_path,
        row_count=row_count,
        fraud_rate=fraud_rate,
        user_profiles=tuple(user_profiles),
        merchant_templates=merchant_templates,
        year_counts=year_counts,
        fraud_counts_by_year=dict(fraud_counts_by_year),
        month_counts_by_year=dict(month_counts_by_year),
        day_counts_by_year_month=dict(day_counts_by_year_month),
        hour_counts_by_year=dict(hour_counts_by_year),
        minute_counts_by_hour=dict(minute_counts_by_hour),
        mcc_counts_by_fraud=dict(mcc_counts_by_fraud),
        use_chip_counts_by_year_fraud=dict(use_chip_counts_by_year_fraud),
        error_counts_by_fraud_chip=dict(error_counts_by_fraud_chip),
        amount_samples_by_mcc_fraud=amount_samples_by_mcc_fraud,
        amount_samples_by_mcc=amount_samples_by_mcc,
        amount_median_by_mcc=amount_median_by_mcc,
        location_counts_by_mcc_chip=dict(location_counts_by_mcc_chip),
        location_counts_by_mcc=dict(location_counts_by_mcc),
    )


def load_empirical_profile(
    csv_path: Path,
    amount_reservoir_size: int,
    quiet: bool,
    progress_interval: int,
) -> EmpiricalProfile:
    try:
        import polars  # noqa: F401
    except ImportError:
        return load_empirical_profile_streaming(
            csv_path,
            amount_reservoir_size,
            quiet,
            progress_interval,
        )

    try:
        return load_empirical_profile_polars(csv_path, amount_reservoir_size, quiet)
    except Exception as exc:
        if not quiet:
            print(
                f"polars profile failed ({exc}); falling back to streaming CSV profiling",
                file=sys.stderr,
            )
        return load_empirical_profile_streaming(
            csv_path,
            amount_reservoir_size,
            quiet,
            progress_interval,
        )


class EmpiricalProfileSampler:
    def __init__(self, profile: EmpiricalProfile):
        self.profile = profile
        self.year_table = WeightedTable.from_counter(Counter(profile.year_counts))
        self.fraud_by_year = {
            year: WeightedTable.from_counter(counter)
            for year, counter in profile.fraud_counts_by_year.items()
            if counter
        }
        self.month_by_year = {
            year: WeightedTable.from_counter(counter)
            for year, counter in profile.month_counts_by_year.items()
            if counter
        }
        self.day_by_year_month = {
            key: WeightedTable.from_counter(counter)
            for key, counter in profile.day_counts_by_year_month.items()
            if counter
        }
        self.hour_by_year = {
            year: WeightedTable.from_counter(counter)
            for year, counter in profile.hour_counts_by_year.items()
            if counter
        }
        self.minute_by_hour = {
            hour: WeightedTable.from_counter(counter)
            for hour, counter in profile.minute_counts_by_hour.items()
            if counter
        }
        self.mcc_by_fraud = {
            fraud: WeightedTable.from_counter(counter)
            for fraud, counter in profile.mcc_counts_by_fraud.items()
            if counter
        }
        self.use_chip_by_year_fraud = {
            key: WeightedTable.from_counter(counter)
            for key, counter in profile.use_chip_counts_by_year_fraud.items()
            if counter
        }
        self.error_by_fraud_chip = {
            key: WeightedTable.from_counter(counter)
            for key, counter in profile.error_counts_by_fraud_chip.items()
            if counter
        }
        self.location_by_mcc_chip = {
            key: WeightedTable.from_counter(counter)
            for key, counter in profile.location_counts_by_mcc_chip.items()
            if counter
        }
        self.location_by_mcc = {
            key: WeightedTable.from_counter(counter)
            for key, counter in profile.location_counts_by_mcc.items()
            if counter
        }

    def timestamp(
        self,
        rng: random.Random,
        start: dt.datetime,
        end: dt.datetime,
        user: UserProfile,
    ) -> dt.datetime:
        for _ in range(25):
            if user.year_table is not None:
                year = user.year_table.choice(rng)
                if year < start.year or year > end.year:
                    year = self.year_table.choice(rng)
            else:
                year = self.year_table.choice(rng)

            if year < start.year or year > end.year:
                continue

            month_table = self.month_by_year.get(year)
            month = month_table.choice(rng) if month_table else rng.randint(1, 12)
            max_day = calendar.monthrange(year, month)[1]
            day_table = self.day_by_year_month.get((year, month))
            day = day_table.choice(rng) if day_table else rng.randint(1, max_day)
            day = min(max(1, int(day)), max_day)

            hour_table = self.hour_by_year.get(year)
            hour = hour_table.choice(rng) if hour_table else rng.randint(0, 23)
            minute_table = self.minute_by_hour.get(hour)
            minute = minute_table.choice(rng) if minute_table else rng.randint(0, 59)
            timestamp = dt.datetime(year, month, day, int(hour), int(minute))
            if start <= timestamp <= end:
                return timestamp

        return random_timestamp_uniform(rng, start, end)

    def fraud(self, rng: random.Random, year: int, fraud_rate_override: Optional[float]) -> bool:
        counts = self.profile.fraud_counts_by_year.get(year)
        if not counts:
            base_rate = self.profile.fraud_rate if fraud_rate_override is None else fraud_rate_override
            return rng.random() < base_rate

        year_total = counts[False] + counts[True]
        if year_total <= 0:
            base_rate = self.profile.fraud_rate if fraud_rate_override is None else fraud_rate_override
            return rng.random() < base_rate

        empirical_year_rate = counts[True] / year_total
        if fraud_rate_override is None:
            return rng.random() < empirical_year_rate

        if self.profile.fraud_rate <= 0:
            return rng.random() < fraud_rate_override
        adjusted_rate = fraud_rate_override * (empirical_year_rate / self.profile.fraud_rate)
        return rng.random() < min(1.0, max(0.0, adjusted_rate))

    def mcc(self, rng: random.Random, fraud: bool) -> Optional[str]:
        table = self.mcc_by_fraud.get(fraud)
        if table is None:
            table = self.mcc_by_fraud.get(False) or self.mcc_by_fraud.get(True)
        return table.choice(rng) if table is not None else None

    def use_chip(self, rng: random.Random, year: int, fraud: bool) -> Optional[str]:
        table = self.use_chip_by_year_fraud.get((year, fraud))
        if table is None:
            table = self.use_chip_by_year_fraud.get((year, False))
        return table.choice(rng) if table is not None else None

    def error(self, rng: random.Random, fraud: bool, use_chip: str) -> Optional[str]:
        table = self.error_by_fraud_chip.get((fraud, use_chip))
        if table is None:
            table = self.error_by_fraud_chip.get((fraud, ""))
        return table.choice(rng) if table is not None else None

    def location(
        self,
        rng: random.Random,
        mcc: str,
        use_chip: str,
        default_location: Tuple[str, str, str],
    ) -> Tuple[str, str, str]:
        table = self.location_by_mcc_chip.get((mcc, use_chip))
        if table is None:
            table = self.location_by_mcc.get(mcc)
        return table.choice(rng) if table is not None else default_location

    def amount(self, rng: random.Random, mcc: str, fraud: bool) -> Optional[float]:
        samples = self.profile.amount_samples_by_mcc_fraud.get((mcc, fraud))
        if not samples:
            samples = self.profile.amount_samples_by_mcc.get(mcc)
        if not samples:
            return None
        return samples[rng.randrange(len(samples))]


def make_empirical_profiles(
    rng: random.Random,
    profile: EmpiricalProfile,
    n_users: int,
    n_merchants: int,
) -> Tuple[List[UserProfile], List[int], List[MerchantProfile]]:
    if not profile.user_profiles:
        raise ValueError("empirical profile does not contain user profiles")
    if not profile.merchant_templates:
        raise ValueError("empirical profile does not contain merchant templates")

    source_users = list(profile.user_profiles)
    if n_users <= len(source_users):
        selected_users = rng.sample(source_users, n_users)
    else:
        selected_users = source_users[:]
        while len(selected_users) < n_users:
            selected_users.append(rng.choice(source_users))

    users: List[UserProfile] = []
    for user_id, source_user in enumerate(selected_users):
        cards = tuple(card for card, _count in source_user.card_counts) or (0,)
        card_weights = tuple(count for _card, count in source_user.card_counts) or (1,)
        year_items = tuple(year for year, _count in source_user.year_counts)
        year_weights = tuple(count for _year, count in source_user.year_counts)
        users.append(
            UserProfile(
                user_id=user_id,
                home_city=source_user.home_city,
                home_state=source_user.home_state,
                home_zip=source_user.home_zip,
                cards=cards,
                spend_scale=1.0,
                online_bias=0.0,
                travel_rate=min(0.30, max(0.01, rng.betavariate(1.5, 18.0))),
                year_table=WeightedTable(year_items, year_weights) if year_items else None,
                card_table=WeightedTable(cards, card_weights),
            )
        )

    template_table = WeightedTable(
        profile.merchant_templates,
        [max(1.0, math.sqrt(template.activity_weight)) for template in profile.merchant_templates],
    )
    merchants: List[MerchantProfile] = []
    seen_ids = set()

    def add_merchant_from_template(template: MerchantTemplate) -> None:
        while True:
            merchant_id = str(rng.randint(-(2**63), 2**63 - 1))
            if merchant_id not in seen_ids:
                break
        seen_ids.add(merchant_id)
        merchants.append(
            MerchantProfile(
                merchant_id=merchant_id,
                city=template.city,
                state=template.state,
                zip_code=template.zip_code,
                mcc=template.mcc,
                amount_median=profile.amount_median_by_mcc.get(template.mcc, 30.0),
                amount_sigma=0.75,
                activity_weight=max(1, template.activity_weight),
            )
        )

    templates_by_mcc: DefaultDict[str, List[MerchantTemplate]] = defaultdict(list)
    for template in profile.merchant_templates:
        templates_by_mcc[template.mcc].append(template)

    if n_merchants >= len(templates_by_mcc):
        for templates in templates_by_mcc.values():
            table = WeightedTable(
                templates,
                [max(1.0, math.sqrt(template.activity_weight)) for template in templates],
            )
            add_merchant_from_template(table.choice(rng))

    while len(merchants) < n_merchants:
        template = template_table.choice(rng)
        add_merchant_from_template(template)

    source_counts = [source_user.transaction_count for source_user in selected_users]
    return users, source_counts, merchants


def allocate_counts(
    rng: random.Random,
    n_transactions: int,
    n_users: int,
    min_transactions_per_user: int,
) -> List[int]:
    required = n_users * min_transactions_per_user
    if required > n_transactions:
        raise ValueError(
            "n-transactions must be at least n-users * min-transactions-per-user "
            f"({required:,} required for the current settings)"
        )

    remaining = n_transactions - required
    weights = [rng.lognormvariate(0.0, 0.95) for _ in range(n_users)]
    total_weight = sum(weights)
    raw_allocations = [remaining * weight / total_weight for weight in weights]
    allocations = [int(math.floor(value)) for value in raw_allocations]
    leftover = remaining - sum(allocations)

    ranked_remainders = sorted(
        range(n_users),
        key=lambda idx: raw_allocations[idx] - allocations[idx],
        reverse=True,
    )
    for idx in ranked_remainders[:leftover]:
        allocations[idx] += 1

    return [min_transactions_per_user + allocation for allocation in allocations]


def scale_counts_to_total(
    source_counts: Sequence[int],
    n_transactions: int,
    min_transactions_per_user: int,
) -> List[int]:
    if not source_counts:
        raise ValueError("source_counts cannot be empty")

    required = len(source_counts) * min_transactions_per_user
    if required > n_transactions:
        raise ValueError(
            "n-transactions must be at least n-users * min-transactions-per-user "
            f"({required:,} required for the current settings)"
        )

    if sum(source_counts) == n_transactions and all(count >= min_transactions_per_user for count in source_counts):
        return list(source_counts)

    source_total = sum(source_counts)
    if source_total <= 0:
        return [min_transactions_per_user for _ in source_counts]

    remaining = n_transactions - required
    raw_allocations = [remaining * count / source_total for count in source_counts]
    allocations = [int(math.floor(value)) for value in raw_allocations]
    leftover = remaining - sum(allocations)

    ranked_remainders = sorted(
        range(len(source_counts)),
        key=lambda idx: raw_allocations[idx] - allocations[idx],
        reverse=True,
    )
    for idx in ranked_remainders[:leftover]:
        allocations[idx] += 1

    return [min_transactions_per_user + allocation for allocation in allocations]


def build_merchant_indexes(
    merchants: Iterable[MerchantProfile],
) -> MerchantIndexes:
    by_state_mcc: DefaultDict[Tuple[str, str], List[MerchantProfile]] = defaultdict(list)
    by_mcc: DefaultDict[str, List[MerchantProfile]] = defaultdict(list)
    all_merchants = list(merchants)

    for merchant in all_merchants:
        by_state_mcc[(merchant.state, merchant.mcc)].append(merchant)
        by_mcc[merchant.mcc].append(merchant)

    weighted_by_state_mcc = {
        key: WeightedTable(value, [merchant.activity_weight for merchant in value])
        for key, value in by_state_mcc.items()
        if value
    }
    weighted_by_mcc = {
        key: WeightedTable(value, [merchant.activity_weight for merchant in value])
        for key, value in by_mcc.items()
        if value
    }
    weighted_all = (
        WeightedTable(all_merchants, [merchant.activity_weight for merchant in all_merchants])
        if all_merchants
        else None
    )

    return MerchantIndexes(
        by_state_mcc=by_state_mcc,
        by_mcc=by_mcc,
        all_merchants=all_merchants,
        weighted_by_state_mcc=weighted_by_state_mcc,
        weighted_by_mcc=weighted_by_mcc,
        weighted_all=weighted_all,
    )


def random_timestamp_uniform(rng: random.Random, start: dt.datetime, end: dt.datetime) -> dt.datetime:
    total_seconds = int((end - start).total_seconds())
    return start + dt.timedelta(seconds=rng.randint(0, total_seconds))


def choose_merchant(
    rng: random.Random,
    user: UserProfile,
    fraud: bool,
    indexes: MerchantIndexes,
    profile_sampler: Optional[EmpiricalProfileSampler],
) -> MerchantProfile:
    if profile_sampler is not None:
        mcc = profile_sampler.mcc(rng, fraud)
    else:
        mcc_weights_key = "fraud_weight" if fraud else "weight"
        mccs = [profile["mcc"] for profile in MCC_PROFILES]
        mcc_weights = [profile[mcc_weights_key] for profile in MCC_PROFILES]
        mcc = weighted_choice(rng, mccs, mcc_weights)

    stay_local = (not fraud and rng.random() > user.travel_rate) or (fraud and rng.random() < 0.15)
    if stay_local:
        local_matches = indexes.by_state_mcc.get((user.home_state, mcc), [])
        if local_matches:
            table = indexes.weighted_by_state_mcc.get((user.home_state, mcc))
            return table.choice(rng) if table is not None else rng.choice(local_matches)

    mcc_matches = indexes.by_mcc.get(mcc, [])
    if mcc_matches:
        table = indexes.weighted_by_mcc.get(mcc)
        return table.choice(rng) if table is not None else rng.choice(mcc_matches)

    if indexes.weighted_all is not None:
        return indexes.weighted_all.choice(rng)
    return rng.choice(indexes.all_merchants)


def choose_use_chip(
    rng: random.Random,
    fraud: bool,
    user: UserProfile,
    year: int,
    profile_sampler: Optional[EmpiricalProfileSampler],
) -> str:
    if profile_sampler is not None:
        use_chip = profile_sampler.use_chip(rng, year, fraud)
        if use_chip is not None:
            return use_chip

    if rng.random() < 0.003:
        return ""

    if fraud:
        methods = ["Online Transaction", "Swipe Transaction", "Chip Transaction"]
        weights = [0.58, 0.32, 0.10]
    else:
        online_weight = min(0.45, max(0.08, user.online_bias))
        methods = ["Chip Transaction", "Swipe Transaction", "Online Transaction"]
        weights = [0.48, 0.52 - online_weight, online_weight]

    return weighted_choice(rng, methods, weights)


def choose_error(
    rng: random.Random,
    fraud: bool,
    use_chip: str,
    profile_sampler: Optional[EmpiricalProfileSampler],
) -> str:
    if profile_sampler is not None:
        error = profile_sampler.error(rng, fraud, use_chip)
        if error is not None:
            return error

    error_rate = 0.018 if fraud else 0.004
    if rng.random() > error_rate:
        return ""
    return rng.choice(ERROR_VALUES)


def transaction_amount(
    rng: random.Random,
    merchant: MerchantProfile,
    user: UserProfile,
    fraud: bool,
    profile_sampler: Optional[EmpiricalProfileSampler],
) -> str:
    if profile_sampler is not None:
        amount = profile_sampler.amount(rng, merchant.mcc, fraud)
        if amount is not None:
            return f"${amount:.2f}"

    median = merchant.amount_median * user.spend_scale
    amount = rng.lognormvariate(math.log(max(1.0, median)), merchant.amount_sigma)
    if fraud:
        amount *= rng.uniform(1.4, 5.5)
        if rng.random() < 0.20:
            amount += rng.uniform(250.0, 1800.0)

    amount = min(9999.99, max(1.00, amount))
    return f"${amount:.2f}"


def make_row(
    rng: random.Random,
    user: UserProfile,
    timestamp: dt.datetime,
    fraud_rate: Optional[float],
    indexes: MerchantIndexes,
    profile_sampler: Optional[EmpiricalProfileSampler],
) -> Dict[str, object]:
    if profile_sampler is not None:
        fraud = profile_sampler.fraud(rng, timestamp.year, fraud_rate)
    else:
        fraud = rng.random() < (fraud_rate if fraud_rate is not None else DEFAULT_FRAUD_RATE)

    use_chip = choose_use_chip(rng, fraud, user, timestamp.year, profile_sampler)
    merchant = choose_merchant(rng, user, fraud, indexes, profile_sampler)

    merchant_city, merchant_state, merchant_zip = merchant.city, merchant.state, merchant.zip_code
    if profile_sampler is not None:
        merchant_city, merchant_state, merchant_zip = profile_sampler.location(
            rng,
            merchant.mcc,
            use_chip,
            (merchant.city, merchant.state, merchant.zip_code),
        )
    else:
        if use_chip == "Online Transaction" and rng.random() < 0.12:
            merchant_state = ""
        if use_chip == "Online Transaction" and rng.random() < 0.18:
            merchant_zip = ""

    card = user.card_table.choice(rng) if user.card_table is not None else rng.choice(user.cards)

    return {
        "User": user.user_id,
        "Card": card,
        "Year": timestamp.year,
        "Month": timestamp.month,
        "Day": timestamp.day,
        "Time": f"{timestamp.hour:02d}:{timestamp.minute:02d}",
        "Amount": transaction_amount(rng, merchant, user, fraud, profile_sampler),
        "Use Chip": use_chip,
        "Merchant Name": merchant.merchant_id,
        "Merchant City": merchant_city,
        "Merchant State": merchant_state,
        "Zip": merchant_zip,
        "MCC": merchant.mcc,
        "Errors?": choose_error(rng, fraud, use_chip, profile_sampler),
        "Is Fraud?": "Yes" if fraud else "No",
    }


def same_resolved_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return False


def default_profile_candidates(output_path: Path) -> List[Path]:
    cwd = Path.cwd()
    return [
        cwd / "data" / "credit_card" / "card_transaction.v1.csv",
        cwd / "data" / "TabFormer" / "raw" / "card_transaction.v1.csv",
        cwd.parent / "transaction-foundation-model" / "data" / "TabFormer" / "raw" / "card_transaction.v1.csv",
        Path.home() / "devpub" / "transaction-foundation-model" / "data" / "TabFormer" / "raw" / "card_transaction.v1.csv",
        output_path.parent.parent / "card_transaction.v1.csv",
    ]


def resolve_profile_csv(args: argparse.Namespace, output_path: Path) -> Optional[Path]:
    if args.no_empirical_profile:
        return None

    if args.profile_csv:
        profile_path = Path(args.profile_csv)
        if same_resolved_path(profile_path, output_path):
            raise ValueError("--profile-csv cannot point to the same file as --output")
        return profile_path

    for candidate in default_profile_candidates(output_path):
        if not candidate.exists() or not candidate.is_file():
            continue
        if same_resolved_path(candidate, output_path):
            continue
        return candidate

    return None


def generate_csv(args: argparse.Namespace) -> None:
    if args.cards_per_user_min > args.cards_per_user_max:
        raise ValueError("cards-per-user-min cannot be greater than cards-per-user-max")
    if args.start_year > args.end_year:
        raise ValueError("start-year cannot be greater than end-year")

    rng = random.Random(args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    profile_path = resolve_profile_csv(args, output_path)
    profile_sampler: Optional[EmpiricalProfileSampler] = None
    if profile_path is not None:
        empirical_profile = load_empirical_profile(
            profile_path,
            args.amount_reservoir_size,
            args.quiet,
            args.progress_interval,
        )
        profile_sampler = EmpiricalProfileSampler(empirical_profile)
        users, source_counts, merchants = make_empirical_profiles(
            rng,
            empirical_profile,
            args.n_users,
            args.n_merchants,
        )
        counts = scale_counts_to_total(
            source_counts,
            args.n_transactions,
            args.min_transactions_per_user,
        )
    else:
        counts = allocate_counts(
            rng,
            args.n_transactions,
            args.n_users,
            args.min_transactions_per_user,
        )
        users, merchants = make_profiles(
            rng,
            args.n_users,
            args.n_merchants,
            args.cards_per_user_min,
            args.cards_per_user_max,
        )

    merchant_indexes = build_merchant_indexes(merchants)
    start = dt.datetime(args.start_year, 1, 1, 0, 0)
    end = dt.datetime(args.end_year, 12, 31, 23, 59)

    rows_written = 0
    last_progress_report = 0
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        pending_rows: List[Dict[str, object]] = []

        def flush_pending_rows() -> None:
            nonlocal rows_written, last_progress_report
            if not pending_rows:
                return
            writer.writerows(pending_rows)
            rows_written += len(pending_rows)
            pending_rows.clear()
            if not args.quiet:
                while rows_written - last_progress_report >= args.progress_interval:
                    last_progress_report += args.progress_interval
                    print(f"wrote {last_progress_report:,} rows", file=sys.stderr)

        for user, count in zip(users, counts):
            if profile_sampler is not None:
                timestamps = sorted(profile_sampler.timestamp(rng, start, end, user) for _ in range(count))
            else:
                timestamps = sorted(random_timestamp_uniform(rng, start, end) for _ in range(count))
            for timestamp in timestamps:
                pending_rows.append(
                    make_row(
                        rng,
                        user,
                        timestamp,
                        args.fraud_rate,
                        merchant_indexes,
                        profile_sampler,
                    )
                )
                if len(pending_rows) >= args.chunk_size:
                    flush_pending_rows()

        flush_pending_rows()

    if not args.quiet:
        print(f"wrote {rows_written:,} rows to {output_path}", file=sys.stderr)


def validate_header(header: Optional[List[str]], errors: List[str]) -> None:
    if header != COLUMNS:
        errors.append(f"header mismatch: expected {COLUMNS}, found {header}")


def validate_row(row: Dict[str, str], row_number: int, errors: List[str]) -> Optional[int]:
    try:
        user_id = int(row["User"])
        int(row["Card"])
        year = int(row["Year"])
        month = int(row["Month"])
        day = int(row["Day"])
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"row {row_number}: invalid integer/date field: {exc}")
        return None

    try:
        max_day = calendar.monthrange(year, month)[1]
        if day < 1 or day > max_day:
            raise ValueError(f"day {day} is outside valid range for {year}-{month}")
    except ValueError as exc:
        errors.append(f"row {row_number}: invalid date: {exc}")

    time_value = row.get("Time", "")
    match = re.match(r"^(\d{2}):(\d{2})$", time_value)
    if not match:
        errors.append(f"row {row_number}: invalid Time value {time_value!r}")
    else:
        hour, minute = int(match.group(1)), int(match.group(2))
        if hour > 23 or minute > 59:
            errors.append(f"row {row_number}: invalid Time value {time_value!r}")

    amount = row.get("Amount", "")
    if not re.match(r"^\$-?\d+(\.\d{2})$", amount):
        errors.append(f"row {row_number}: invalid Amount value {amount!r}")

    is_fraud = row.get("Is Fraud?", "")
    if is_fraud not in {"Yes", "No"}:
        errors.append(f"row {row_number}: invalid Is Fraud? value {is_fraud!r}")

    return user_id


def validate_csv(
    csv_path: str,
    expected_rows: Optional[int],
    min_transactions_per_user: int,
    max_errors: int = 25,
) -> int:
    path = Path(csv_path)
    errors: List[str] = []
    user_counts: DefaultDict[int, int] = defaultdict(int)
    row_count = 0

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        validate_header(reader.fieldnames, errors)
        for row_count, row in enumerate(reader, start=1):
            if reader.fieldnames and len(row) != len(reader.fieldnames):
                errors.append(f"row {row_count}: incorrect column count")
            user_id = validate_row(row, row_count, errors)
            if user_id is not None:
                user_counts[user_id] += 1
            if len(errors) >= max_errors:
                break

    if expected_rows is not None and row_count != expected_rows:
        errors.append(f"row count mismatch: expected {expected_rows:,}, found {row_count:,}")

    short_users = [user_id for user_id, count in user_counts.items() if count < min_transactions_per_user]
    if short_users:
        errors.append(
            f"{len(short_users):,} users have fewer than {min_transactions_per_user} transactions"
        )

    if errors:
        print("validation failed:", file=sys.stderr)
        for error in errors[:max_errors]:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(
        f"validation passed: {row_count:,} rows, {len(user_counts):,} users, "
        f"minimum {min_transactions_per_user} rows/user",
        file=sys.stderr,
    )
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or validate synthetic TabFormer credit-card transactions."
    )
    parser.add_argument("--output", help="CSV path to write when generating data")
    parser.add_argument("--validate", metavar="CSV", help="validate an existing CSV and exit")
    parser.add_argument("--expected-rows", type=positive_int, help="expected row count for validation")
    parser.add_argument("--n-transactions", type=positive_int, default=100_000)
    parser.add_argument("--n-users", type=positive_int, default=1_000)
    parser.add_argument("--n-merchants", type=positive_int, default=10_000)
    parser.add_argument("--start-year", type=positive_int, default=2002)
    parser.add_argument("--end-year", type=positive_int, default=2020)
    parser.add_argument(
        "--fraud-rate",
        type=probability,
        default=None,
        help=(
            "target fraud rate. Defaults to the empirical profile rate when a "
            f"profile is used, otherwise {DEFAULT_FRAUD_RATE}"
        ),
    )
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--cards-per-user-min", type=positive_int, default=1)
    parser.add_argument("--cards-per-user-max", type=positive_int, default=5)
    parser.add_argument("--min-transactions-per-user", type=positive_int, default=10)
    parser.add_argument(
        "--profile-csv",
        help=(
            "reference card_transaction CSV used to fit empirical marginal and "
            "conditional distributions. If omitted, common local TabFormer paths "
            "are auto-detected."
        ),
    )
    parser.add_argument(
        "--no-empirical-profile",
        action="store_true",
        help="disable empirical profiling and use the built-in fallback generator",
    )
    parser.add_argument(
        "--amount-reservoir-size",
        type=positive_int,
        default=DEFAULT_AMOUNT_RESERVOIR_SIZE,
        help="maximum sampled amount values retained per MCC/fraud bucket while profiling",
    )
    parser.add_argument(
        "--progress-interval",
        type=positive_int,
        default=1_000_000,
        help="print progress every N generated rows",
    )
    parser.add_argument(
        "--chunk-size",
        type=positive_int,
        default=50_000,
        help="number of rows to buffer before writing to the CSV",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress generation progress")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.validate:
            return validate_csv(
                args.validate,
                args.expected_rows,
                args.min_transactions_per_user,
            )

        if not args.output:
            raise ValueError("--output is required unless --validate is used")

        generate_csv(args)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
