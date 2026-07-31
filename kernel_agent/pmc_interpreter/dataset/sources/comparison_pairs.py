"""读取并追加平台无关 baseline/candidate implementation 配对。"""

from __future__ import annotations

import csv
from pathlib import Path

from ..model import ComparisonPair, PmcDataset
from ..parsing import parse_optional_bool


def _resolve_condition(raw_id, conditions, condition_by_case):
    if raw_id in conditions:
        return raw_id
    return condition_by_case.get(raw_id, "")


def read_comparison_pairs_csv(path, conditions, existing_pair_ids=()):
    comparisons = []
    issues = []
    known_pair_ids = set(existing_pair_ids)
    condition_by_case = {
        condition.case_id: condition_id
        for condition_id, condition in conditions.items()
    }
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        legacy_fields = {"baseline_case", "variant_case"}
        normalized_fields = {
            "baseline_condition_id",
            "candidate_condition_id",
        }
        if not legacy_fields.issubset(fields) and not normalized_fields.issubset(
            fields
        ):
            raise ValueError(
                "pair CSV must contain baseline_case,variant_case or "
                "baseline_condition_id,candidate_condition_id"
            )
        for index, row in enumerate(reader, 1):
            baseline_raw = (
                row.get("baseline_condition_id", "").strip()
                or row.get("baseline_case", "").strip()
            )
            candidate_raw = (
                row.get("candidate_condition_id", "").strip()
                or row.get("variant_case", "").strip()
            )
            baseline = _resolve_condition(
                baseline_raw, conditions, condition_by_case
            )
            candidate = _resolve_condition(
                candidate_raw, conditions, condition_by_case
            )
            if not baseline or not candidate:
                issues.append(
                    "pair row {} references a missing condition".format(index)
                )
                continue
            if baseline == candidate:
                issues.append(
                    "pair row {} compares a condition with itself".format(index)
                )
                continue
            if (
                conditions[baseline].implementation_id
                == conditions[candidate].implementation_id
            ):
                issues.append(
                    "pair row {} does not change implementation_id".format(index)
                )
                continue
            if (
                conditions[baseline].semantic_equivalence_key
                != conditions[candidate].semantic_equivalence_key
            ):
                issues.append(
                    "pair row {} semantic equivalence mismatch".format(index)
                )
                continue
            pair_id = row.get("pair_id", "").strip() or "manifest_{:04d}".format(
                index
            )
            if pair_id in known_pair_ids:
                issues.append("duplicate comparison pair ID {}".format(pair_id))
                continue
            known_pair_ids.add(pair_id)
            matched_raw = row.get("matched_fields", "").strip()
            matched_fields = (
                tuple(
                    item.strip()
                    for item in matched_raw.split(",")
                    if item.strip()
                )
                if matched_raw
                else ("op_type", "dtype", "validator", "shapes", "params")
            )
            comparisons.append(
                ComparisonPair(
                    pair_id=pair_id,
                    baseline_condition_id=baseline,
                    candidate_condition_id=candidate,
                    intervention_id=row.get("intervention_id", "").strip(),
                    controlled_mechanism=row.get(
                        "controlled_mechanism", ""
                    ).strip(),
                    design=row.get("design", "").strip()
                    or "paired_observational",
                    matched_fields=matched_fields,
                    environment_matched=parse_optional_bool(
                        row.get("environment_matched")
                    ),
                    randomized_order=parse_optional_bool(
                        row.get("randomized_order")
                    ),
                    notes=row.get("notes", "").strip(),
                )
            )
    return tuple(comparisons), tuple(issues)


def append_comparison_pairs_csv(dataset, path):
    comparisons, issues = read_comparison_pairs_csv(
        path,
        dataset.conditions,
    )
    retained = list(dataset.comparisons)
    retained_by_endpoint = {
        tuple(sorted((pair.baseline_condition_id, pair.candidate_condition_id))): pair
        for pair in retained
    }
    retained_ids = {pair.pair_id for pair in retained}
    accepted = []
    append_issues = list(issues)
    for pair in comparisons:
        endpoint = tuple(
            sorted((pair.baseline_condition_id, pair.candidate_condition_id))
        )
        existing = retained_by_endpoint.get(endpoint)
        if existing is not None:
            if existing.design == "auto_semantic_match":
                retained.remove(existing)
                retained_ids.discard(existing.pair_id)
                append_issues.append(
                    "explicit pair {} replaced auto pair {}".format(
                        pair.pair_id, existing.pair_id
                    )
                )
            else:
                append_issues.append(
                    "pair {} duplicates existing comparison endpoints".format(
                        pair.pair_id
                    )
                )
                continue
        if pair.pair_id in retained_ids:
            append_issues.append(
                "duplicate comparison pair ID {}".format(pair.pair_id)
            )
            continue
        retained_by_endpoint[endpoint] = pair
        retained_ids.add(pair.pair_id)
        accepted.append(pair)
    payload = dataset.model_dump(mode="python")
    payload.update(
        {
            "comparisons": tuple(retained + accepted),
            "issues": tuple(dataset.issues) + tuple(append_issues),
        }
    )
    return PmcDataset.model_validate(payload)
