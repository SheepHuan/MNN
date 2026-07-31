"""当前数据快照中的观测冗余与同 concept 保守去重层。"""

from __future__ import annotations

from collections import defaultdict

from ..dataset.reports import (
    QualityOutput,
    RedundancyCluster,
    RedundancyDecision,
    RedundancyOutput,
    RedundancyReport,
)
from .base import (
    AnalysisLayer,
    append_layer_record,
    descriptor_for_feature,
    feature_condition_medians,
)


def _fingerprint(output, feature_id):
    medians = feature_condition_medians(output, feature_id)
    return tuple(
        medians.get(condition_id)
        for condition_id in sorted(output.context.active_condition_ids)
    )


def _representative(output, feature_ids):
    return max(
        feature_ids,
        key=lambda feature_id: (
            descriptor_for_feature(output, feature_id).mapping_confidence,
            descriptor_for_feature(output, feature_id).aggregation == "sum",
            -len(descriptor_for_feature(output, feature_id).native_name),
            feature_id,
        ),
    )


class RedundancyLayer(AnalysisLayer):
    name = "redundancy"
    input_type = QualityOutput

    def run(self, layer_input):
        self.require_input(layer_input)
        observational_clusters = defaultdict(list)
        for feature_id in layer_input.active_feature_ids:
            observational_clusters[_fingerprint(layer_input, feature_id)].append(
                feature_id
            )

        forced_groups = defaultdict(list)
        for fingerprint, feature_ids in observational_clusters.items():
            for feature_id in feature_ids:
                descriptor = descriptor_for_feature(layer_input, feature_id)
                forced_groups[(fingerprint, descriptor.concept_id)].append(feature_id)

        representative_for = {}
        for feature_ids in forced_groups.values():
            representative = _representative(layer_input, feature_ids)
            for feature_id in feature_ids:
                representative_for[feature_id] = representative

        active = []
        decisions = {}
        cross_concept_clusters = []
        for cluster_index, feature_ids in enumerate(
            observational_clusters.values(), 1
        ):
            concepts = tuple(
                sorted(
                    {
                        descriptor_for_feature(layer_input, item).concept_id
                        for item in feature_ids
                    }
                )
            )
            cluster_id = "observational_cluster_{:04d}".format(cluster_index)
            if len(feature_ids) > 1 and len(concepts) > 1:
                cross_concept_clusters.append(
                    RedundancyCluster(
                        cluster_id=cluster_id,
                        features=tuple(sorted(feature_ids)),
                        concepts=concepts,
                        disposition="annotated_only_not_forced",
                    )
                )
            for feature_id in feature_ids:
                representative = representative_for[feature_id]
                included = feature_id == representative
                if included:
                    active.append(feature_id)
                decisions[feature_id] = RedundancyDecision(
                    feature_id=feature_id,
                    observational_cluster_id=cluster_id,
                    same_concept_representative=representative,
                    included_after_same_concept_dedup=included,
                )

        active_ids = tuple(sorted(active))
        policy = (
            "equal vectors are forcibly deduplicated only within the same canonical concept; "
            "cross-concept equality is retained as mechanism-level evidence"
        )
        report = RedundancyReport(
            input_feature_count=len(layer_input.active_feature_ids),
            active_feature_count=len(active_ids),
            observational_cluster_count=len(observational_clusters),
            cross_concept_equal_vector_clusters=tuple(cross_concept_clusters),
            metrics=decisions,
            policy=policy,
            descriptive_canonical_features=active_ids,
        )
        return RedundancyOutput(
            context=layer_input.context,
            active_feature_ids=active_ids,
            projection_report=layer_input.projection_report,
            layer_history=append_layer_record(
                layer_input,
                self.name,
                active_ids,
                (policy,),
            ),
            quality_report=layer_input.quality_report,
            redundancy_report=report,
        )
