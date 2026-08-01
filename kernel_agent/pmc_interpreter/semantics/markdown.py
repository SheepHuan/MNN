"""将结构化 PMC 分析报告渲染为固定章节的中文 Markdown。"""

from __future__ import annotations

import re
from collections import Counter

from ..dataset.document import MarkdownDocument, MarkdownSection
from ..dataset.reports import PmcAnalysisReport


CODE_LABELS = {
    "diagnostic_only": "仅诊断",
    "workload_indicator": "工作量指标",
    "latency_informative_candidate": "延迟信息候选",
    "descriptive_association": "描述性关联",
    "descriptive_only": "仅表征",
    "raw_monotonic_descriptive": "原始单调描述",
    "raw_nonlinear_descriptive_candidate": "原始非线性描述候选",
    "raw_context_descriptive_candidate": "原始情境关联候选",
    "residual_monotonic": "残差单调关联",
    "residual_nonlinear_candidate": "残差非线性候选",
    "moderate_context_candidate": "中等情境关联候选",
    "weak_or_none": "弱关联或无稳定关联",
    "insufficient_evidence": "证据不足",
    "sparse_activation": "稀疏触发",
    "target_coupled": "目标耦合",
    "target_equivalent": "目标等价",
    "increase_associated_with_higher_latency": "PMC 增大伴随延迟升高",
    "increase_associated_with_lower_latency": "PMC 增大伴随延迟降低",
    "paired_observational": "配对观察证据",
    "controlled_intervention": "受控干预证据",
    "atomic_contention": "原子竞争",
    "atomic_operation": "原子操作工作量",
    "cache_locality": "缓存局部性",
    "compute_pipeline": "计算管线",
    "control_divergence": "控制流分歧",
    "dram_memory": "外部显存",
    "global_activity": "全局活动",
    "interconnect": "互连",
    "parallelism": "并行度",
    "scheduler_wait": "调度等待",
    "synchronization": "同步/冲突",
    "target_timing": "目标时长",
    "unmapped": "未映射",
    "active_efficiency": "活跃期间效率",
    "capacity": "容量/配置",
    "ratio": "比例",
    "symptom": "症状",
    "timing": "时长",
    "utilization": "整体利用率",
    "work": "累计工作量",
    "counter": "计数",
    "throughput": "吞吐/利用率",
    "small_n_descriptive_signature": "小样本描述性面板",
    "exploratory_signature": "探索性面板",
    "insufficient_samples": "样本不足",
    "fdr_supported_group_association": "类别内 FDR 支持",
    "descriptive_group_association": "类别内描述性候选",
    "small_n_descriptive": "小样本描述性候选",
    "exploratory": "探索性可用",
    "exploratory_incomplete_sources": "数据源不完整，仅可探索",
    "family_level_exploratory": "仅 family 层探索性可用",
    "family_level_exploratory_incomplete_sources": "数据源不完整，仅 family 层探索",
    "insufficient_group_samples": "类别样本不足",
    "insufficient_paired_variants": "实现配对不足",
    "not_estimable_without_latency": "缺少 latency，无法估计",
    "not_estimable_without_pmc_and_latency": "缺少 PMC 或 latency，无法估计",
    "paired_analysis_available": "配对分析可用",
    "insufficient_paired_variant_evidence": "实现配对证据不足",
}


EXACT_ZH = {
    "integer counters contain many ties": "整数 counter 含有大量并列值，显著性近似需谨慎。",
    "association does not establish optimization causality": "相关性不能建立优化因果关系。",
    "missing workload or environment controls lower the evidence level": "工作量或环境控制缺失会降低证据等级。",
    "distance correlation has no permutation p-value in the current implementation": "当前 distance correlation 尚无置换检验 p 值。",
    "zero-delta pairs are retained": "分析保留 ΔPMC 为零的配对。",
    "univariate relations cannot separate simultaneously changing non-collinear mechanisms": "单变量关系不能拆分同时变化的多个非共线机制。",
    "duration-coupled and diagnostic-only metrics cannot enter optimization effect rules": "与时长公式耦合或仅用于诊断的指标不能进入优化效应规则。",
    "causal claims require controlled intervention, environment matching, randomization, and independent repeats": "因果表述要求受控干预、环境匹配、随机顺序和独立重复。",
    "the current slope is a single-variable transformed linear approximation": "当前斜率只是单变量变换空间中的线性近似。",
    "rules are valid only for their listed conditions, interventions, and observed value ranges": "规则只适用于其中列出的 condition、干预和观测取值范围，不得外推。",
    "implementation pairs are required for delta analysis": "ΔPMC–ΔLatency 分析必须有实现配对。",
    "latency associations are uncontrolled cross-kernel descriptions": "当前 latency 关联只是未充分控制的跨 kernel 描述。",
    "no delta effect rule is generated from cross-kernel correlation": "不会把跨 kernel 相关性转换成 delta 优化规则。",
    "the current dataset does not support a stable delta-PMC/delta-latency rule": "当前数据集不支持稳定的 ΔPMC–ΔLatency 规则。",
    "diagnostic-only metrics are isolated from latency explanation and optimization effect sets": "仅诊断指标已与延迟解释集合和优化效应集合隔离。",
    "no independent repeat_id; repeatability cannot be estimated": "缺少独立 repeat_id，无法估计重复性。",
    "legacy rows have no collection_session_id; per-case PMC run grouping is synthetic and must not be used to assume metrics were co-collected": "当前 rows 没有 collection_session_id；按 case 合成的 PMC run 不能证明不同 metric 同时采集。",
    "the latency association set contains only unnormalized work counters; it does not measure execution efficiency": "当前 latency association set 全部是未归一化累计工作量 counter，不能据此判断执行效率。",
    "some selected metric semantics are generic mechanism proxies rather than vendor-verified exact mappings": "部分入选 PMC 只来自通用名称推断的机制代理，并非厂商公式核验后的精确语义映射。",
}


REQUIRED_DESIGN_ZH = {
    "same device and semantic_equivalence_key": "baseline 与 candidate 必须属于同一设备和同一 semantic_equivalence_key。",
    "explicit baseline/candidate implementation IDs": "必须显式记录 baseline/candidate 的不同 implementation_id。",
    "independent latency and PMC repeat_id values inside shared paired-run groups": "共享配套运行组内，latency 与 PMC 都必须有独立 repeat_id。",
    "at least two shared paired_run_group_id values linking valid PMC and latency": "每个实现配对至少需要两个共享 paired_run_group_id，并同时连接有效 PMC 与 latency。",
    "measured clock/temperature compatibility inside every shared paired-run group": "每个共享配套运行组都要实测并核验 clock/temperature 匹配。",
    "controlled intervention with randomized order, intervention ID, and one non-empty matching mechanism": "受控实验必须随机运行顺序，并记录 intervention_id 和唯一、非空且匹配的 controlled_mechanism。",
}


def _zh_code(value):
    return CODE_LABELS.get(value, value)


def _zh_warning(value):
    if value in EXACT_ZH:
        return EXACT_ZH[value]
    if value.startswith("missing high-value controls: "):
        return "缺失高价值控制字段：{}。".format(
            value.split(": ", 1)[1]
        )
    if value.startswith("latency/manifest mismatch: "):
        match = re.search(
            r"(\d+) missing, (\d+) extra, (\d+) invalid", value
        )
        if match:
            return (
                "Latency 与 manifest 不一致：缺少 {} 个 case，多出 {} 个 case，"
                "另有 {} 个无效值。"
            ).format(*match.groups())
    if value.startswith("PMC rows/manifest representative mismatch: "):
        match = re.search(
            r"(\d+) expected op-type representatives, (\d+) observed", value
        )
        if match:
            return (
                "PMC rows 与 manifest 的代表 case 不一致：应有 {} 个 op_type "
                "代表 case，实际只有 {} 个。"
            ).format(*match.groups())
    if value.startswith("PMC rows contain "):
        return "PMC rows 存在重复记录：{}。".format(value)
    if value.startswith("PMC rows are not a complete "):
        return "PMC rows 的 case × metric 网格不完整：{}。".format(value)
    return value


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers, rows):
    if not rows:
        return "_无可展示记录。_"
    lines = [
        "| " + " | ".join(_cell(item) for item in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_cell(item) for item in row) + " |"
        for row in rows
    )
    return "\n".join(lines)


def _short_metric(feature_id):
    return feature_id.split("::", 1)[-1]


def _interpretation_boundary(descriptor):
    if descriptor.target_equivalent:
        return "这是目标时长的直接等价表达，不能作为独立解释变量。"
    if descriptor.duration_coupled:
        return "公式含 elapsed time/cycle；数值变化可能由 latency 分母机械产生。"
    if descriptor.phenomenon_role == "work":
        if descriptor.unit == "cycle":
            return "高值表示累计活动周期更多，混合了工作量和活动时长，不等于效率更差。"
        return "高值表示累计事件、流量或指令更多；未按语义工作量归一化时不能判断效率。"
    if descriptor.phenomenon_role == "symptom":
        return "高值表示该症状事件更多；仍需在工作量固定时结合绝对周期和占比解释。"
    if descriptor.phenomenon_role == "utilization":
        return "高值既可能表示利用充分，也可能表示资源已饱和；不能设成统一单调目标。"
    if descriptor.phenomenon_role == "active_efficiency":
        return "只描述活跃期间效率；总 latency 还取决于活跃时长和未活跃等待。"
    if descriptor.phenomenon_role == "capacity":
        return "容量与资源占用通常存在 trade-off，不能解释为越高或越低越好。"
    if descriptor.phenomenon_role == "ratio":
        return "比例受分子、分母和其他组成项共同约束，需要配套绝对计数。"
    return "当前仅能描述观测现象，不能给出统一的性能好坏方向。"


def _selection_evidence(report, knowledge):
    association = knowledge.association
    if association is None:
        return "无关联统计"
    threshold = report.provenance.analysis_config.fdr_threshold
    if association.fdr_qvalue <= threshold:
        return "FDR 描述性支持"
    return "mRMR 多样性候选"


def _association_direction(report, knowledge):
    association = knowledge.association
    if association is None:
        return "无可用 latency 关联。"
    if association.statistical_relation_class in {
        "weak_or_none",
        "insufficient_evidence",
    }:
        return "当前未观察到稳定的 latency 单调关系。"
    rho = association.residual_log_latency_spearman
    direction = "较高 latency" if rho > 0 else "较低 latency"
    if report.layer_reports.correlation.evidence_level == "descriptive_cross_kernel":
        return "跨 kernel 中数值较高伴随{}；这是描述性关联，不是优化方向。".format(
            direction
        )
    return "控制当前可用字段后，数值较高伴随{}；仍不是因果结论。".format(
        direction
    )


def _families_for_metric(knowledge):
    values = [
        link.value
        for link in knowledge.kernel_class_links
        if link.axis == "semantic_family"
    ]
    return ", ".join(dict.fromkeys(values)) or "未形成 family signature"


def _zh_selection_basis(value):
    if value.startswith(
        "constrained mRMR over uncontrolled cross-kernel descriptive associations"
    ):
        return (
            "基于未充分控制的跨 kernel 描述性关联执行约束 mRMR；"
            "当前 workload/shape residualization 未生效"
        )
    if value.startswith("constrained mRMR over correlation evidence residualized"):
        return "基于 latency baseline 所列控制项残差化后的关联证据执行约束 mRMR"
    return value


def _core_questions_section(report):
    association_features = report.canonical_metric_sets.latency_association.features
    selected_knowledge = [
        report.metric_knowledge[feature_id]
        for feature_id in association_features
        if feature_id in report.metric_knowledge
    ]
    fdr_supported = sum(
        knowledge.association is not None
        and knowledge.association.fdr_qvalue
        <= report.provenance.analysis_config.fdr_threshold
        for knowledge in selected_knowledge
    )
    unnormalized_work = sum(
        knowledge.descriptor.phenomenon_role == "work"
        and knowledge.descriptor.normalizer == "none"
        for knowledge in selected_knowledge
    )

    family_signatures = report.kernel_signatures.signature_index.get(
        "semantic_family", {}
    )
    direct_families = sum(
        bool(signature.resolved_features or signature.diagnostic_features)
        for signature in family_signatures.values()
    )
    op_signatures = report.kernel_signatures.signature_index.get("op_type", {})
    direct_op_types = sum(bool(item.resolved_features) for item in op_signatures.values())
    inherited_op_types = sum(
        not item.resolved_features and bool(item.inherited_family_features)
        for item in op_signatures.values()
    )
    unavailable_op_types = len(op_signatures) - direct_op_types - inherited_op_types

    question_one = (
        "{} 个 association PMC，其中 {} 个达到当前 FDR 门槛；{} 个是未归一化工作量 counter。"
        "可用于描述硬件活动量，不能据此回答执行效率或优化方向。"
    ).format(
        len(selected_knowledge),
        fdr_supported,
        unnormalized_work,
    )
    question_two = (
        "{} 个 semantic family 中 {} 个形成直接探索面板；op_type 直接实证 {} 个、"
        "继承 family 先验 {} 个、不可用 {} 个。"
    ).format(
        len(family_signatures),
        direct_families,
        direct_op_types,
        inherited_op_types,
        unavailable_op_types,
    )
    delta = report.layer_reports.delta_relations
    question_three = (
        "合法 implementation pair {} 个、独立语义组 {} 个、effect rule {} 条；"
        "当前不能从跨 kernel 相关性推导 ΔPMC–ΔLatency 结论。"
    ).format(
        delta.valid_pair_count,
        delta.independent_pair_group_count,
        len(report.delta_rulebook),
    )
    return _table(
        ("核心问题", "当前可以回答到的程度"),
        (
            ("1. PMC 相关性与分类", question_one),
            ("2. Kernel type × PMC set", question_two),
            ("3. ΔPMC × ΔLatency", question_three),
        ),
    )


def _dataset_section(report):
    readiness = report.data_readiness
    rows = (
        ("数据集", report.scope.dataset_id),
        ("平台 / 后端", "{} / {}".format(report.scope.platform, report.scope.backend)),
        ("设备数", readiness.device_count),
        ("Kernel condition 数", readiness.condition_count),
        ("原生 metric 数", readiness.metric_count),
        ("有效 latency condition 数", readiness.latency_condition_count),
        (
            "合法实现配对数 / 独立语义组数",
            "{} / {}".format(
                readiness.valid_comparison_pair_count,
                readiness.valid_comparison_group_count,
            ),
        ),
        (
            "问题一：相关性与分类",
            _zh_code(readiness.question_readiness.correlation_and_classification),
        ),
        (
            "问题二：Kernel type × PMC set",
            _zh_code(readiness.question_readiness.kernel_type_metric_sets),
        ),
        (
            "问题三：ΔPMC × ΔLatency",
            _zh_code(readiness.question_readiness.delta_pmc_delta_latency),
        ),
    )
    missing = (
        "、".join(readiness.missing_high_value_fields)
        if readiness.missing_high_value_fields
        else "无"
    )
    return "{}\n\n缺失或覆盖不足的高价值控制字段：{}。".format(
        _table(("项目", "结果"), rows), missing
    )


def _quality_section(report):
    quality = report.layer_reports.quality
    rows = [
        (reason, count)
        for reason, count in sorted(
            quality.exclusion_counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    summary = (
        "输入 {} 个特征，质量过滤后保留 {} 个；形成 {} 个 basename/表达分组。"
    ).format(
        quality.input_feature_count,
        quality.active_feature_count,
        quality.native_basename_expression_groups,
    )
    return (
        summary
        + " 排除原因可以重叠，因此下表数量之和不等于输入特征数。\n\n"
        + _table(("排除原因", "数量"), rows)
    )


def _correlation_section(report):
    correlation = report.layer_reports.correlation
    class_counts = Counter(
        item.statistical_relation_class
        for item in correlation.associations.values()
    )
    class_table = _table(
        ("相关性类别", "Metric 数"),
        [
            (_zh_code(name), count)
            for name, count in sorted(
                class_counts.items(), key=lambda item: (-item[1], item[0])
            )
        ],
    )
    association_rows = []
    for feature_id in report.canonical_metric_sets.latency_association.features:
        knowledge = report.metric_knowledge.get(feature_id)
        if knowledge is None or knowledge.association is None:
            continue
        descriptor = knowledge.descriptor
        association = knowledge.association
        association_rows.append(
            (
                descriptor.native_name,
                "{} / {}".format(
                    _zh_code(descriptor.mechanism),
                    _zh_code(descriptor.phenomenon_role),
                ),
                "{} · {} · {}".format(
                    _zh_code(descriptor.quantity_kind),
                    descriptor.unit,
                    descriptor.normalizer,
                ),
                "{:.3f} / {:.3f}".format(
                    association.raw_log_latency_spearman,
                    association.residual_log_latency_spearman,
                ),
                "{:.3f}".format(
                    association.residual_log_latency_distance_correlation
                ),
                "{:.3g}".format(association.fdr_qvalue),
                _selection_evidence(report, knowledge),
                "{} / {:.2f}".format(
                    descriptor.mapping_level,
                    descriptor.mapping_confidence,
                ),
                _families_for_metric(knowledge),
                "{} {}".format(
                    _association_direction(report, knowledge),
                    _interpretation_boundary(descriptor),
                ),
            )
        )
    residualized_count = sum(
        association.feature_residualized
        for association in correlation.associations.values()
    )
    controls = ", ".join(correlation.baseline.controls) or "无有效控制项"
    return (
        "Latency baseline 证据等级：`{}`，交叉验证 R²：`{:.4f}`；控制项：{}。"
        "实际完成 feature residualization 的 PMC 为 {}/{} 个。\n\n"
        "Latency association set 的选择依据：{}。\n\n{}\n\n{}"
    ).format(
        correlation.evidence_level,
        correlation.baseline.cross_validated_r_squared,
        controls,
        residualized_count,
        len(correlation.associations),
        _zh_selection_basis(
            report.canonical_metric_sets.latency_association.selection_basis
        ),
        class_table,
        _table(
            (
                "PMC",
                "机制 / 角色",
                "类型 · 单位 · normalizer",
                "raw / residual ρ",
                "dCor",
                "FDR q",
                "入选依据",
                "语义映射 / 置信度",
                "适用 family",
                "解释边界",
            ),
            association_rows,
        ),
    )


def _redundancy_section(report):
    redundancy = report.layer_reports.redundancy
    canonical = report.canonical_metric_sets.descriptive_canonical.features
    mechanism_counts = Counter()
    role_counts = Counter()
    for feature_id in canonical:
        knowledge = report.metric_knowledge.get(feature_id)
        if knowledge is None:
            continue
        mechanism_counts[knowledge.descriptor.mechanism] += 1
        role_counts[knowledge.descriptor.phenomenon_role] += 1
    mechanism_table = _table(
        ("机制", "Canonical PMC 数"),
        [
            (_zh_code(name), count)
            for name, count in sorted(
                mechanism_counts.items(), key=lambda item: (-item[1], item[0])
            )
        ],
    )
    role_table = _table(
        ("物理角色", "Canonical PMC 数"),
        [
            (_zh_code(name), count)
            for name, count in sorted(
                role_counts.items(), key=lambda item: (-item[1], item[0])
            )
        ],
    )
    preview = ", ".join(_short_metric(item) for item in canonical[:12])
    if not preview:
        preview = "无"
    return (
        "发现 {} 个观测向量簇；其中 {} 个跨 concept 相等簇只做注释、不强制删除。"
        "最终描述性 canonical set 共 {} 个。它只表示稳定、非重复且可描述 kernel，"
        "不使用 latency 做筛选。完整列表见结构化 JSON。\n\n{}\n\n{}\n\n"
        "示例：{}。"
    ).format(
        redundancy.observational_cluster_count,
        len(redundancy.cross_concept_equal_vector_clusters),
        len(canonical),
        mechanism_table,
        role_table,
        preview,
    )


def _signature_section(report):
    index = report.kernel_signatures.signature_index
    family_rows = []
    family_metric_rows = []
    for family, signature in sorted(index.get("semantic_family", {}).items()):
        selected = signature.resolved_features + signature.diagnostic_features
        mechanisms = tuple(dict.fromkeys(item.mechanism for item in selected))
        matched = sum(item.semantic_mechanism_match for item in selected)
        family_rows.append(
            (
                family,
                signature.sample_count,
                _zh_code(signature.status),
                len(signature.resolved_features),
                len(signature.diagnostic_features),
                ", ".join(_zh_code(item) for item in mechanisms) or "无",
                "{}/{}".format(matched, len(selected)) if selected else "0/0",
            )
        )
        for role, metrics in (
            ("解释候选", signature.resolved_features[:8]),
            ("仅诊断", signature.diagnostic_features[:4]),
        ):
            for rank, item in enumerate(metrics, 1):
                family_metric_rows.append(
                    (
                        family,
                        role,
                        rank,
                        item.native_name,
                        item.concept_id,
                        _zh_code(item.mechanism),
                        "{:.3f}".format(
                            item.residual_latency_spearman or 0.0
                        ),
                        "{:.3g}".format(item.fdr_qvalue or 1.0),
                        "是" if item.semantic_mechanism_match else "否",
                        _zh_code(item.evidence),
                    )
                )

    op_rows = []
    for op_type, signature in sorted(index.get("op_type", {}).items()):
        if signature.resolved_features:
            source = "op_type 直接实证"
            features = tuple(
                item.feature_id for item in signature.resolved_features
            )
        elif signature.inherited_family_features:
            source = "继承 family 先验"
            features = signature.inherited_family_features
        else:
            source = "无可用面板"
            features = ()
        preview = ", ".join(_short_metric(item) for item in features[:5]) or "无"
        if len(features) > 5:
            preview += "，另 {} 个".format(len(features) - 5)
        op_rows.append(
            (
                op_type,
                signature.sample_count,
                _zh_code(signature.status),
                source,
                preview,
            )
        )

    axis_rows = [
        (axis, len(groups))
        for axis, groups in (
            ("semantic_family", index.get("semantic_family", {})),
            ("execution_role", index.get("execution_role", {})),
            ("semantic_family × execution_role", index.get("semantic_role", {})),
            ("op_type", index.get("op_type", {})),
            ("performance_regime", index.get("performance_regime", {})),
        )
    ]
    return (
        "Signature 是类别诊断面板，不是瓶颈因果集合。只有 `resolved_features` 是该类别"
        "直接计算得到的探索结果；`inherited_family_features` 只是样本不足时的 family 先验，"
        "两者不会混写。\n\n"
        "### 分类轴覆盖\n\n{}\n\n"
        "### Semantic family 直接面板\n\n{}\n\n"
        "### Family 入选 PMC 证据\n\n{}\n\n"
        "### Op type 直接证据与继承关系\n\n{}"
    ).format(
        _table(("分类轴", "类别数"), axis_rows),
        _table(
            (
                "Semantic family",
                "样本",
                "状态",
                "解释候选数",
                "诊断指标数",
                "已覆盖机制",
                "匹配预期机制",
            ),
            family_rows,
        ),
        _table(
            (
                "Family",
                "角色",
                "Rank",
                "PMC",
                "Concept",
                "机制",
                "类别内 ρ",
                "FDR q",
                "匹配预期",
                "证据",
            ),
            family_metric_rows,
        ),
        _table(
            ("Op type", "样本", "状态", "证据来源", "PMC set"),
            op_rows,
        ),
    )


def _delta_section(report):
    delta = report.layer_reports.delta_relations
    if not report.delta_rulebook:
        required = "\n".join(
            "- {}".format(REQUIRED_DESIGN_ZH.get(item, item))
            for item in delta.required_design
        )
        return (
            "当前状态：`{}`；合法 implementation pair 为 {}，独立语义组为 {}，要求至少 {}。\n\n"
            "因此当前数据不能回答“某个 PMC 变大会显著导致 latency 如何变化”。"
            "跨 kernel 相关性不会被转换成优化规则。\n\n"
            "要生成第一版 rulebook，实验输入至少必须满足：\n\n{}"
        ).format(
            _zh_code(delta.status),
            delta.valid_pair_count,
            delta.independent_pair_group_count,
            delta.minimum_pairs,
            required,
        )
    rows = []
    for rule in report.delta_rulebook[:40]:
        statistic = rule.statistics
        op_types = tuple(
            sorted({item.op_type for item in rule.scope_selector.conditions})
        )
        raw_delta = rule.observed_support.raw_feature_delta
        latency_ratio = rule.observed_support.latency_ratio
        rows.append(
            (
                rule.scope_selector.semantic_families or ("全局",),
                op_types or ("未指定",),
                rule.native_name,
                _zh_code(statistic.relation_class),
                "{:.3f}".format(statistic.delta_spearman or 0.0),
                "{:.3g}".format(statistic.fdr_qvalue or 1.0),
                "{:.4g}".format(statistic.transformed_slope or 0.0),
                "[{:.4g}, {:.4g}]".format(raw_delta.minimum, raw_delta.maximum),
                "[{:.4g}, {:.4g}]".format(
                    latency_ratio.minimum, latency_ratio.maximum
                ),
                _zh_code(rule.evidence_level),
                "是" if rule.causal_claim_allowed else "否",
                rule.interpretation,
            )
        )
    return _table(
        (
            "适用 family",
            "适用 op_type",
            "PMC",
            "方向",
            "ΔSpearman",
            "FDR q",
            "变换后斜率",
            "原始 ΔPMC 范围",
            "Latency 比值范围",
            "证据",
            "允许因果表述",
            "规则解释",
        ),
        rows,
    )


def _limitations_section(report):
    items = list(report.warnings)
    items.extend(report.layer_reports.correlation.limitations)
    items.extend(report.layer_reports.delta_relations.limitations)
    unique = tuple(dict.fromkeys(items))
    return "\n".join("- {}".format(_zh_warning(item)) for item in unique) or "- 无额外警告。"


def _next_steps_section(report):
    readiness = report.data_readiness
    items = []
    baseline_fields = tuple(
        field
        for field in readiness.missing_high_value_fields
        if field not in {"pmc_repeat_id", "latency_repeats"}
    )
    if baseline_fields:
        items.append(
            "P0：补齐 {}，让 latency baseline 能区分工作量与执行效率。".format(
                "、".join(baseline_fields)
            )
        )
    if readiness.op_types_with_multiple_conditions == 0:
        items.append(
            "P0：目标 op_type 至少采集 5 个不同 shape/dtype/workload condition；"
            "当前没有 op_type 具备多个 condition，无法形成 op-type 直接实证 signature。"
        )
    if (
        readiness.pmc_conditions_with_independent_repeats == 0
        or readiness.latency_conditions_with_independent_repeats == 0
    ):
        items.append(
            "P1：为 PMC 和 latency 增加独立 repeat_id，并记录同一配套运行组中的 clock、"
            "temperature、运行顺序和 cache policy。"
        )
    if readiness.valid_comparison_group_count < report.layer_reports.delta_relations.minimum_pairs:
        items.append(
            "P1：构造同语义 baseline/candidate implementation pairs，满足 Delta 章节列出的"
            "配套实验契约。"
        )
    selected = [
        report.metric_knowledge[feature_id]
        for feature_id in report.canonical_metric_sets.latency_association.features
        if feature_id in report.metric_knowledge
    ]
    if any(
        item.descriptor.mapping_level != "exact_quantity"
        or item.descriptor.mapping_confidence < 0.8
        for item in selected
    ):
        items.append(
            "P1：为最终入选的 native PMC 增加厂商公式、dependency 和架构版本 override，"
            "把通用 mechanism proxy 升级为可审计的精确语义映射。"
        )
    items.append(
        "P2：在 workload 字段完备后，投影 PMC/output_elements、PMC/FLOPs、"
        "PMC/algorithmic_bytes 等单位工作量特征。"
    )
    items.append(
        "P2：若要输出“原因候选 → 资源结果 → scheduler 症状”链，需要在 catalog 中定义"
        "平台无关 mechanism ontology，并由 typed 分析层输出机制一致性证据；Markdown 只负责渲染。"
    )
    return "\n".join("- {}".format(item) for item in items)


def render_markdown_document(report):
    if type(report) is not PmcAnalysisReport:
        raise TypeError(
            "render_markdown_document expects PmcAnalysisReport, got {}".format(
                type(report).__name__
            )
        )
    sections = (
        MarkdownSection(
            title="数据集与证据完整性", content=_dataset_section(report)
        ),
        MarkdownSection(
            title="三个核心问题结论", content=_core_questions_section(report)
        ),
        MarkdownSection(
            title="PMC 质量和过滤结果", content=_quality_section(report)
        ),
        MarkdownSection(
            title="PMC 相关性分类", content=_correlation_section(report)
        ),
        MarkdownSection(
            title="重复指标与 Canonical Set",
            content=_redundancy_section(report),
        ),
        MarkdownSection(
            title="Kernel Type × PMC Signature",
            content=_signature_section(report),
        ),
        MarkdownSection(
            title="ΔPMC 与 ΔLatency 规则", content=_delta_section(report)
        ),
        MarkdownSection(
            title="下一步完成路径", content=_next_steps_section(report)
        ),
        MarkdownSection(
            title="证据限制和不可回答问题",
            content=_limitations_section(report),
        ),
    )
    title = "PMC 语义分析报告"
    markdown = "# {}\n\n{}\n".format(
        title,
        "\n\n".join(
            "## {}\n\n{}".format(section.title, section.content)
            for section in sections
        ),
    )
    return MarkdownDocument(
        title=title,
        sections=sections,
        warnings=report.warnings,
        markdown=markdown,
    )
