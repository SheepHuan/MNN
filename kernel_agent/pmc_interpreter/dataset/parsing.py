"""Shared parsing helpers for PMC dataset sources."""

import hashlib
import json
import math


def parse_number(raw, allow_negative=True):
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    if not allow_negative and value < 0:
        return None
    return value


def parse_optional_int(raw):
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def parse_optional_bool(raw):
    if raw is None or raw == "":
        return None
    return str(raw).strip().lower() in {"1", "true", "yes", "y"}


def semantic_equivalence_key(op_type, dtype, validator, shapes, params):
    payload = {
        "op_type": op_type,
        "dtype": dtype,
        "validator": validator,
        "shapes": shapes,
        "params": params,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
