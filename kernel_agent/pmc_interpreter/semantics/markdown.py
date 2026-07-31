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
    "increase_associated_with_higher_latency": "PMC 增大伴随延迟升高",
    "increase_associated_with_lower_latency": "PMC 增大伴随延迟降低",
    "paired_observational": "配对观察证据",
    "controlled_intervention": "受控干预证据",
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
            readiness.question_readiness.correlation_and_classification,
        ),
        (
            "问题二：Kernel type × PMC set",
            readiness.question_readiness.kernel_type_metric_sets,
        ),
        (
            "问题三：ΔPMC × ΔLatency",
            readiness.question_readiness.delta_pmc_delta_latency,
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
    return summary + "\n\n" + _table(("排除原因", "数量"), rows)


def _correlation_section(report):
    correlation = report.layer_reports.correlation
    class_counts = Counter(
        item.statistical_relation_class
        for item in correlation.associations.values()
    )
    class_table = _table(
        ("相关性类别", "Metric 数"),
        sorted(class_counts.items(), key=lambda item: (-item[1], item[0])),
    )
    top_rows = []
    for item in correlation.top_associations[:15]:
        top_rows.append(
            (
                item.feature_id,
                "{:.3f}".format(item.raw_log_latency_spearman),
                "{:.3f}".format(item.residual_log_latency_spearman),
                "{:.3g}".format(item.fdr_qvalue),
                _zh_code(item.statistical_relation_class),
                _zh_code(item.usage_class),
            )
        )
    return (
        "Latency baseline 证据等级：`{}`，交叉验证 R²：`{:.4f}`。\n\n{}\n\n{}"
    ).format(
        correlation.evidence_level,
        correlation.baseline.cross_validated_r_squared,
        class_table,
        _table(
            ("PMC", "raw ρ", "residual ρ", "FDR q", "关系类别", "用途"),
            top_rows,
        ),
    )


def _redundancy_section(report):
    redundancy = report.layer_reports.redundancy
    canonical = report.canonical_metric_sets.descriptive_canonical.features
    preview = "\n".join("- `{}`".format(item) for item in canonical[:30])
    if len(canonical) > 30:
        preview += "\n- ……其余 {} 个见结构化 JSON。".format(len(canonical) - 30)
    if not preview:
        preview = "_没有通过质量与冗余过滤的 canonical metric。_"
    return (
        "发现 {} 个观测向量簇；其中 {} 个跨 concept 相等簇只做注释、不强制删除。"
        "最终描述性 canonical set 共 {} 个。\n\n{}"
    ).format(
        redundancy.observational_cluster_count,
        len(redundancy.cross_concept_equal_vector_clusters),
        len(canonical),
        preview,
    )


def _signature_section(report):
    rows = []
    for axis in ("semantic_family", "op_type", "performance_regime"):
        for value, signature in sorted(
            report.kernel_signatures.signature_index.get(axis, {}).items()
        ):
            if signature.status == "insufficient_samples" and not signature.inherited_family_features:
                continue
            resolved = ", ".join(
                item.native_name for item in signature.resolved_features[:8]
            ) or ", ".join(signature.inherited_family_features[:8])
            diagnostics = ", ".join(
                item.native_name for item in signature.diagnostic_features[:5]
            )
            rows.append(
                (
                    axis,
                    value,
                    signature.sample_count,
                    signature.status,
                    resolved or "无",
                    diagnostics or "无",
                )
            )
    return _table(
        ("分类轴", "Kernel 类别", "样本", "状态", "解释型 PMC set", "诊断型 PMC set"),
        rows[:80],
    )


def _delta_section(report):
    delta = report.layer_reports.delta_relations
    if not report.delta_rulebook:
        return (
            "当前状态：`{}`；合法 implementation pair 为 {}，独立语义组为 {}，要求至少 {}。\n\n"
            "因此当前数据不能回答“某个 PMC 变大会显著导致 latency 如何变化”。"
            "跨 kernel 相关性不会被转换成优化规则。"
        ).format(
            delta.status,
            delta.valid_pair_count,
            delta.independent_pair_group_count,
            delta.minimum_pairs,
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
        ),
        rows,
    )


def _limitations_section(report):
    items = list(report.warnings)
    items.extend(report.layer_reports.correlation.limitations)
    items.extend(report.layer_reports.delta_relations.limitations)
    unique = tuple(dict.fromkeys(items))
    return "\n".join("- {}".format(_zh_warning(item)) for item in unique) or "- 无额外警告。"


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
