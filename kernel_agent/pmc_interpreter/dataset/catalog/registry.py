"""Map native counter names and kernel metadata into a shared ontology."""

import json
from pathlib import Path

from ..model import MetricDescriptor


DEFAULT_TAXONOMY = Path(__file__).with_name("default_kernel_taxonomy.json")
ROLLUPS = ("avg", "sum", "max", "min")
NORMALIZED_SUFFIXES = {
    "per_second": "second",
    "per_cycle_active": "active_cycle",
    "per_cycle_elapsed": "elapsed_cycle",
    "per_warp_active": "active_warp",
    "pct_of_peak_sustained_active": "active_peak",
    "pct_of_peak_sustained_elapsed": "elapsed_peak",
    "ratio": "ratio",
    "pct": "ratio",
}


class KernelTaxonomy:
    def __init__(self, payload):
        self.payload = payload
        self.op_to_family = {}
        for family, op_types in payload.get("semantic_families", {}).items():
            for op_type in op_types:
                if op_type in self.op_to_family:
                    raise ValueError("op_type {} appears in multiple families".format(op_type))
                self.op_to_family[op_type] = family

    @classmethod
    def load(cls, path=DEFAULT_TAXONOMY):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def classify(self, op_type, variant):
        family = self.op_to_family.get(op_type, "unclassified")
        expected_mechanisms = self.payload.get("expected_mechanisms", {}).get(
            family, []
        )
        exact_roles = self.payload.get("exact_variant_roles", {})
        if variant in exact_roles:
            role = exact_roles[variant]
            role_source = "exact_variant"
            role_confidence = 1.0
        else:
            lower = variant.lower()
            role = "core_compute"
            role_source = "default"
            role_confidence = 0.45
            for rule in self.payload.get("ordered_role_rules", []):
                if any(token in lower for token in rule.get("tokens", [])):
                    role = rule["role"]
                    role_source = "variant_token"
                    role_confidence = 0.75
                    break
        return {
            "semantic_family": family,
            "execution_role": role,
            "semantic_family_source": "op_type_taxonomy" if family != "unclassified" else "unmapped",
            "execution_role_source": role_source,
            "semantic_family_confidence": 0.8 if family != "unclassified" else 0.0,
            "execution_role_confidence": role_confidence,
            "expected_mechanisms": list(expected_mechanisms),
        }


class MetricRegistry:
    """Generic name-based fallback with optional exact descriptor overrides.

    Analysis layers consume the resulting descriptor and never branch on the
    platform.  Vendor-specific maps can override any inferred field.
    """

    def __init__(self, overrides=None):
        self.overrides = dict(overrides or {})

    @classmethod
    def from_json(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload.get("metrics", payload))

    def describe(self, native_name, namespace="generic", metric_id=None):
        inferred = self._infer(native_name, namespace, metric_id)
        override = self.overrides.get("{}::{}".format(namespace, native_name), self.overrides.get(native_name))
        if not override:
            return inferred
        values = inferred.model_dump(mode="python")
        values.update(override)
        if values.get("bounded_range") is not None:
            values["bounded_range"] = tuple(values["bounded_range"])
        values["dependency_metric_ids"] = tuple(values.get("dependency_metric_ids", ()))
        values["mapping_source"] = override.get("mapping_source", "registry_override")
        values["mapping_level"] = override.get("mapping_level", "exact_quantity")
        values["mapping_confidence"] = float(override.get("mapping_confidence", 1.0))
        return MetricDescriptor(**values)

    def _infer(self, native_name, namespace, metric_id):
        basename, aggregation, suffix = self._split_name(native_name)
        lower = basename.lower()
        native_domain = self._native_domain(basename)
        normalizer = NORMALIZED_SUFFIXES.get(suffix, "none")
        duration_coupled = any(token in native_name.lower() for token in (
            "time_duration", "cycles_elapsed", "per_second", "per_cycle_elapsed",
            "pct_of_peak_sustained_elapsed",
        ))
        target_equivalent = any(token in lower for token in (
            "time_duration", "cycles_elapsed", "cycles_in_frame", "cycles_in_region"
        ))
        mechanism, hardware_scope = self._mechanism(lower, native_domain, target_equivalent)
        quantity_kind, unit = self._quantity(lower, suffix, target_equivalent)
        phenomenon_role = self._phenomenon_role(
            lower, normalizer, target_equivalent, mechanism, quantity_kind
        )
        analysis_scope = self._analysis_scope(lower, native_domain, target_equivalent)
        concept_id = self._concept_id(lower, mechanism, hardware_scope, unit, target_equivalent)
        bounded_range = (0.0, 100.0) if suffix.startswith("pct") else (
            (0.0, 1.0) if suffix == "ratio" else None
        )
        return MetricDescriptor(
            metric_id=metric_id or "{}::{}".format(namespace, native_name),
            namespace=namespace,
            native_name=native_name,
            native_domain=native_domain,
            basename=basename,
            concept_id=concept_id,
            hardware_scope=hardware_scope,
            mechanism=mechanism,
            phenomenon_role=phenomenon_role,
            quantity_kind=quantity_kind,
            unit=unit,
            aggregation=aggregation,
            normalizer=normalizer,
            bounded_range=bounded_range,
            duration_coupled=duration_coupled,
            target_equivalent=target_equivalent,
            dependency_metric_ids=(),
            collection_cost=None,
            analysis_scope=analysis_scope,
            mapping_level="mechanism_proxy",
            mapping_confidence=0.65,
            mapping_source="generic_name_inference",
        )

    @staticmethod
    def _split_name(native_name):
        segments = native_name.split(".")
        rollup_index = next((index for index, value in enumerate(segments[1:], 1) if value in ROLLUPS), None)
        if rollup_index is not None:
            return (
                ".".join(segments[:rollup_index]),
                segments[rollup_index],
                ".".join(segments[rollup_index + 1:]),
            )
        if len(segments) > 1 and segments[-1] in NORMALIZED_SUFFIXES:
            return ".".join(segments[:-1]), "none", segments[-1]
        return native_name, "none", ""

    @staticmethod
    def _native_domain(basename):
        if "__" in basename:
            return basename.split("__", 1)[0]
        lower = basename.lower()
        known_prefixes = (
            "dram", "fbpa", "idc", "lts", "l2", "l1tex", "smsp", "sm", "tpc",
            "gpc", "gr", "sp", "tp", "uche", "rbbm", "vbif", "compute", "gpu", "cp",
            "pcie", "sys", "fe", "gcc",
        )
        return next((prefix for prefix in known_prefixes if lower.startswith(prefix + "_")), "unknown")

    @staticmethod
    def _mechanism(lower, domain, target_equivalent):
        if target_equivalent:
            return "target_timing", "global"
        if "atom" in lower or "atomic" in lower:
            return "atomic_contention", "memory_system"
        if any(token in lower for token in ("bank_conflict", "barrier", "membar")):
            return "synchronization", "execution_core"
        if any(token in lower for token in ("diverg", "branch", "pred_off")):
            return "control_divergence", "execution_core"
        if any(token in lower for token in ("stalled", "pending", "scoreboard", "throttle")):
            return "scheduler_wait", "scheduler"
        if domain in {"lts", "l2", "uche", "gcc"} or "l2_" in lower:
            return "cache_locality", "l2"
        if domain in {"l1tex", "tp"} or "l1_" in lower:
            return "cache_locality", "l1_or_shared"
        if domain in {"dram", "fbpa", "idc", "vbif", "rbbm"} or "ext_read" in lower or "ext_write" in lower:
            return "dram_memory", "external_memory"
        if any(token in lower for token in (
            "warps_active", "ctas_active", "working_eu", "eligible", "issue_active", "tasks"
        )):
            return "parallelism", "scheduler"
        if domain in {"sm", "smsp", "tpc", "gpc", "gr", "sp", "compute"}:
            return "compute_pipeline", "execution_core"
        if domain in {"pcie", "sys"}:
            return "interconnect", "host_link"
        if domain in {"gpu", "cp", "fe"}:
            return "global_activity", "global"
        return "unmapped", "unknown"

    @staticmethod
    def _quantity(lower, suffix, target_equivalent):
        if target_equivalent:
            return "timing", "cycle_or_time"
        if "throughput" in lower or "pct_of_peak" in suffix:
            return "throughput", "ratio"
        if any(token in lower for token in ("ratio", "rate", "efficiency", "occupancy", "utilization")):
            return "ratio", "ratio"
        if "bytes" in lower:
            return "counter", "byte"
        if "inst" in lower or "instruction" in lower:
            return "counter", "instruction"
        if "sector" in lower:
            return "counter", "sector"
        if "request" in lower or "lookup" in lower:
            return "counter", "request"
        if "cycle" in lower or "busy" in lower or "active" in lower:
            return "counter", "cycle"
        if "warp" in lower or "eu" in lower:
            return "counter", "thread_group"
        return "counter", "event"

    @staticmethod
    def _phenomenon_role(lower, normalizer, target_equivalent, mechanism, quantity_kind):
        if target_equivalent:
            return "timing"
        if mechanism in {"scheduler_wait", "control_divergence", "synchronization", "atomic_contention"}:
            return "symptom"
        if normalizer in {"active_cycle", "active_peak"}:
            return "active_efficiency"
        if normalizer in {"elapsed_cycle", "elapsed_peak", "second"} or quantity_kind == "throughput":
            return "utilization"
        if quantity_kind == "ratio":
            return "ratio"
        if any(token in lower for token in ("register", "shared_memory", "occupancy")):
            return "capacity"
        return "work"

    @staticmethod
    def _analysis_scope(lower, domain, target_equivalent):
        if target_equivalent:
            return "diagnostic_only"
        if domain in {"fe", "gcc", "pcie", "sys"}:
            return "out_of_kernel_core"
        if "aperture_sysmem" in lower or "aperture_peer" in lower:
            return "conditional_external_memory"
        if domain in {"gpu", "cp"}:
            return "diagnostic_only"
        return "kernel_core"

    @staticmethod
    def _concept_id(lower, mechanism, scope, unit, target_equivalent):
        if target_equivalent:
            return "timing.elapsed"
        if mechanism == "dram_memory":
            return "memory.external.{}".format("bytes" if unit == "byte" else "activity")
        if mechanism == "cache_locality":
            level = "l2" if scope == "l2" else "l1_or_shared"
            result = "miss" if "miss" in lower else ("hit" if "hit" in lower else "traffic")
            return "cache.{}.{}".format(level, result)
        if mechanism == "scheduler_wait":
            return "scheduler.wait"
        if mechanism == "parallelism":
            return "parallelism.active"
        if mechanism == "compute_pipeline":
            return "compute.instructions" if unit == "instruction" else "compute.activity"
        if mechanism == "synchronization":
            return "synchronization.wait_or_conflict"
        if mechanism == "atomic_contention":
            return "synchronization.atomic"
        if mechanism == "control_divergence":
            return "control.divergence"
        return "{}.{}".format(mechanism, unit)
