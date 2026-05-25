from __future__ import annotations

import json
import re
from collections import defaultdict
from contextlib import contextmanager
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Iterator, TextIO
from urllib.request import Request, urlopen

from .eq_to_magnitude_native import _validate_band_gain_db, _validate_filter_slope
from .preamp import validate_preamp_db
from .validation import ensure_numeric_scalar

VALID_EQ_TYPES = {"parametric_eq"}
VALID_BAND_TYPES = {"peak_dip", "low_shelf", "high_shelf", "low_pass"}
VALID_SUBTYPES = {"earbuds", "in_ear", "on_ear", "over_the_ear"}
DEFAULT_OPRA_DB_URL = "https://opra.roonlabs.net/database_v1.jsonl"
DEFAULT_OPRA_USER_AGENT = "fir-dsp-opra/1.1 (+https://opra.roonlabs.net/)"
DEFAULT_OPRA_TIMEOUT_SECONDS = 20
MEASUREMENT_STOP_WORDS = {
    "and",
    "for",
    "from",
    "in",
    "on",
    "rig",
    "target",
    "using",
    "with",
}
DEFAULT_TARGET_PREFERENCE_ORDER = (
    "oratory1990_harman_target",
    "autoeq_oratory1990",
    "autoeq_crinacle",
    "autoeq_rtings",
    "autoeq_innerfidelity",
    "autoeq_headphonecom_legacy",
    "autoeq_super_review",
    "autoeq_tonedeafmonk",
    "autoeq_jaytiss",
    "autoeq_hi_end_portable",
    "autoeq_fahryst",
    "autoeq_rikudougoku",
    "autoeq_kazi",
    "oratory1990_usound_target",
    "oratory1990_oratory1990_target",
)
_DEFAULT_TARGET_PREFERENCE_RANK = {
    target: index for index, target in enumerate(DEFAULT_TARGET_PREFERENCE_ORDER)
}


def _is_url(path: str | Path) -> bool:
    return isinstance(path, str) and path.startswith(("http://", "https://"))


@contextmanager
def _open_opra_source(path: str | Path) -> Iterator[TextIO]:
    if _is_url(path):
        request = Request(str(path), headers={"User-Agent": DEFAULT_OPRA_USER_AGENT})
        with urlopen(request, timeout=DEFAULT_OPRA_TIMEOUT_SECONDS) as response:
            yield TextIOWrapper(response, encoding="utf-8-sig")
        return

    with Path(path).open("r", encoding="utf-8-sig") as handle:
        yield handle


def load_opra_jsonl(path: str | Path) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    products: dict[str, dict[str, Any]] = {}
    eqs_by_product: dict[str, list[dict[str, Any]]] = defaultdict(list)
    vendors: dict[str, dict[str, Any]] = {}

    with _open_opra_source(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            obj = json.loads(line)
            obj_type = obj["type"]

            if obj_type == "product":
                products[obj["id"]] = obj
            elif obj_type == "eq":
                eqs_by_product[obj["data"]["product_id"]].append(obj)
            elif obj_type == "vendor":
                vendors[obj["id"]] = obj

    return products, eqs_by_product, vendors


def normalize_query(value: str) -> str:
    value = value.lower()
    value = value.replace("wh1000xm5", "wh_1000xm5")
    value = value.replace("lcd2c", "lcd_2_classic")
    value = re.sub(r"[^a-z0-9_]+", " ", value)
    return " ".join(value.split())


def find_product(products: dict[str, dict[str, Any]], query: str) -> dict[str, Any]:
    normalized_query = normalize_query(query)
    matches = []

    for product in products.values():
        name = normalize_query(product["data"].get("name", ""))
        product_id = normalize_query(product["id"])
        if normalized_query in name or normalized_query in product_id:
            matches.append(product)

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous product: {[match['id'] for match in matches]}")
    raise ValueError(f"No product match for '{query}'")


def parse_eq_id(eq_id: str) -> tuple[str, str, str]:
    left, target = eq_id.split("::", 1)
    vendor, product = left.split(":", 1)
    return vendor, f"{vendor}::{product}", target


def _normalize_measurement(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def extract_measurement(eq: dict[str, Any]) -> str:
    details = eq["data"].get("details", "").lower()
    match = re.search(r"\bmeasured by\b(?P<tail>.+)", details)
    if match is None:
        return "unknown"

    tokens = re.findall(r"[a-z0-9]+", match.group("tail"))
    measurement_tokens: list[str] = []
    for token in tokens:
        if token in MEASUREMENT_STOP_WORDS:
            break
        measurement_tokens.append(token)

    if not measurement_tokens:
        return "unknown"
    return _normalize_measurement(" ".join(measurement_tokens))


def _default_target_preference_rank(eq: dict[str, Any]) -> int | None:
    return _DEFAULT_TARGET_PREFERENCE_RANK.get(parse_eq_id(eq["id"])[2])


def validate_product(product: dict[str, Any]) -> None:
    subtype = product["data"].get("subtype")
    if subtype not in VALID_SUBTYPES:
        raise ValueError(f"{product['id']}: invalid subtype '{subtype}'")


def validate_eq(eq: dict[str, Any], product: dict[str, Any], vendors: dict[str, dict[str, Any]]) -> None:
    data = eq["data"]
    if data.get("type") not in VALID_EQ_TYPES:
        raise ValueError(f"{eq['id']}: unsupported EQ type")
    if data["product_id"] != product["id"]:
        raise ValueError(f"{eq['id']}: product_id mismatch")

    vendor, _, _ = parse_eq_id(eq["id"])
    if vendor != product["data"].get("vendor_id"):
        raise ValueError(f"{eq['id']}: vendor mismatch")
    if vendor not in vendors:
        raise ValueError(f"{eq['id']}: unknown vendor")

    extract_measurement(eq)

    params = data.get("parameters", {})
    validate_preamp_db(0.0 if params.get("gain_db", 0.0) is None else params.get("gain_db", 0.0))
    bands = params.get("bands", [])
    if not isinstance(bands, list) or not bands:
        raise ValueError(f"{eq['id']}: no bands")

    for band in bands:
        if not isinstance(band, dict):
            raise ValueError(f"{eq['id']}: each band must be a dictionary")
        band_type = band["type"]
        if band_type not in VALID_BAND_TYPES:
            raise ValueError(f"{eq['id']}: invalid band type {band_type}")
        if "frequency" not in band:
            raise ValueError(f"{eq['id']}: missing frequency")
        frequency = ensure_numeric_scalar("band frequency", band["frequency"])
        if frequency <= 0.0:
            raise ValueError("band frequency must be a positive finite value")
        if band_type in {"peak_dip", "low_shelf", "high_shelf"}:
            _validate_band_gain_db(band.get("gain_db", 0.0))
            q = ensure_numeric_scalar("band q", band.get("q", 0.7))
            if q <= 0.0:
                raise ValueError("band q must be a positive finite value")
        elif band_type == "low_pass":
            _validate_filter_slope(band.get("slope", 12.0))


def select_eq(
    product: dict[str, Any],
    eqs_by_product: dict[str, list[dict[str, Any]]],
    vendors: dict[str, dict[str, Any]],
    *,
    target: str | None = None,
    measurement: str | None = None,
) -> dict[str, Any]:
    eqs = eqs_by_product.get(product["id"], [])
    if not eqs:
        raise ValueError(f"No EQ for {product['id']}")

    validate_product(product)

    candidates = list(eqs)
    if target:
        candidates = [eq for eq in candidates if parse_eq_id(eq["id"])[2].startswith(target)]
    if measurement:
        normalized_measurement = _normalize_measurement(measurement)
        candidates = [eq for eq in candidates if extract_measurement(eq) == normalized_measurement]

    valid: list[dict[str, Any]] = []
    validation_errors: list[ValueError] = []
    for eq in candidates:
        try:
            validate_eq(eq, product, vendors)
        except ValueError as exc:
            validation_errors.append(exc)
            continue
        valid.append(eq)

    if not valid:
        if validation_errors:
            raise validation_errors[0]
        raise ValueError("No EQ matches constraints")
    if len(valid) == 1:
        return valid[0]

    preferred_candidates = [
        (_default_target_preference_rank(eq), eq)
        for eq in valid
        if _default_target_preference_rank(eq) is not None
    ]
    if preferred_candidates:
        best_rank = min(rank for rank, _ in preferred_candidates if rank is not None)
        winners = [eq for rank, eq in preferred_candidates if rank == best_rank]
        if len(winners) == 1:
            return winners[0]

    identities = {(extract_measurement(eq), parse_eq_id(eq["id"])[2]) for eq in valid}
    if len(identities) > 1:
        raise ValueError(f"Ambiguous EQ set: {identities}. Use --target or --measurement.")

    return valid[0]


def band_to_txt(index: int, band: dict[str, Any]) -> str:
    band_type = band["type"]
    frequency = ensure_numeric_scalar("band frequency", band["frequency"])
    if frequency <= 0.0:
        raise ValueError("band frequency must be a positive finite value")

    if band_type == "peak_dip":
        gain_db = _validate_band_gain_db(band.get("gain_db", 0))
        q = ensure_numeric_scalar("band q", band.get("q", 0.7))
        if q <= 0.0:
            raise ValueError("band q must be a positive finite value")
        return f"Filter {index}: ON PK Fc {frequency} Hz Gain {gain_db} dB Q {q}"
    if band_type == "low_shelf":
        gain_db = _validate_band_gain_db(band.get("gain_db", 0))
        q = ensure_numeric_scalar("band q", band.get("q", 0.7))
        if q <= 0.0:
            raise ValueError("band q must be a positive finite value")
        return f"Filter {index}: ON LS Fc {frequency} Hz Gain {gain_db} dB Q {q}"
    if band_type == "high_shelf":
        gain_db = _validate_band_gain_db(band.get("gain_db", 0))
        q = ensure_numeric_scalar("band q", band.get("q", 0.7))
        if q <= 0.0:
            raise ValueError("band q must be a positive finite value")
        return f"Filter {index}: ON HS Fc {frequency} Hz Gain {gain_db} dB Q {q}"
    if band_type == "low_pass":
        slope = _validate_filter_slope(band.get("slope", 12))
        return f"Filter {index}: ON LP Fc {frequency} Hz Slope {slope} dB/oct"
    raise ValueError(f"Invalid band type: {band_type}")


def eq_to_txt(eq: dict[str, Any]) -> str:
    params = eq["data"]["parameters"]
    raw_preamp = params.get("gain_db", 0.0)
    preamp_db = validate_preamp_db(0.0 if raw_preamp is None else raw_preamp)
    lines = [f"Preamp: {preamp_db} dB"]
    for index, band in enumerate(params["bands"], start=1):
        lines.append(band_to_txt(index, band))
    return "\n".join(lines)


def build_ui_attribution(product: dict[str, Any], eq: dict[str, Any], vendors: dict[str, dict[str, Any]]) -> dict[str, str]:
    vendor_id, _, target = parse_eq_id(eq["id"])
    vendor_name = vendors.get(vendor_id, {}).get("data", {}).get("name", vendor_id)
    product_name = product["data"].get("name", product["id"])
    measurement = extract_measurement(eq)
    measurement_display = measurement.replace("_", " ").title() if measurement != "unknown" else "Unknown"
    target_display = target.replace("_", " ")

    title = product_name
    subtitle = f"{vendor_name} | {target_display}"
    short_credit = f"Preset from OPRA for {product_name}"
    long_credit = (
        f"Product metadata and EQ preset from OPRA. "
        f"Vendor: {vendor_name}. Product: {product_name}. "
        f"Target: {target_display}. Measurement: {measurement_display}."
    )
    legal_credit = "OPRA dataset attribution required. See NOTICE_OPRA.txt for license details."

    return {
        "title": title,
        "subtitle": subtitle,
        "target": target_display,
        "measurement": measurement_display,
        "short_credit": short_credit,
        "long_credit": long_credit,
        "legal_credit": legal_credit,
    }
