"""分析层共享的无第三方统计原语。"""

from __future__ import annotations

import hashlib
import math
import statistics


def median(values):
    return float(statistics.median(values))


def rankdata(values):
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def pearson(left, right):
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_ss = sum((x - left_mean) ** 2 for x in left)
    right_ss = sum((y - right_mean) ** 2 for y in right)
    if left_ss <= 0 or right_ss <= 0:
        return 0.0
    return max(-1.0, min(1.0, numerator / math.sqrt(left_ss * right_ss)))


def spearman(left, right):
    return pearson(rankdata(left), rankdata(right))


def distance_correlation(left, right):
    """一维样本的 biased sample distance correlation。"""
    if len(left) != len(right) or len(left) < 3:
        return 0.0
    size = len(left)

    def centered_distances(values):
        distances = [
            [abs(values[row] - values[column]) for column in range(size)]
            for row in range(size)
        ]
        row_means = [statistics.fmean(row) for row in distances]
        grand_mean = statistics.fmean(row_means)
        return [
            [
                distances[row][column]
                - row_means[row]
                - row_means[column]
                + grand_mean
                for column in range(size)
            ]
            for row in range(size)
        ]

    left_centered = centered_distances(left)
    right_centered = centered_distances(right)
    covariance = sum(
        left_centered[row][column] * right_centered[row][column]
        for row in range(size)
        for column in range(size)
    ) / (size * size)
    left_variance = sum(
        left_centered[row][column] ** 2
        for row in range(size)
        for column in range(size)
    ) / (size * size)
    right_variance = sum(
        right_centered[row][column] ** 2
        for row in range(size)
        for column in range(size)
    ) / (size * size)
    denominator = math.sqrt(max(0.0, left_variance * right_variance))
    if denominator <= 1e-18:
        return 0.0
    squared = max(0.0, covariance / denominator)
    return min(1.0, math.sqrt(squared))


def approximate_correlation_pvalue(rho, sample_count):
    if sample_count < 4:
        return 1.0
    clipped = max(-0.999999999, min(0.999999999, rho))
    z_score = abs(math.atanh(clipped)) * math.sqrt(sample_count - 3)
    return math.erfc(z_score / math.sqrt(2.0))


def benjamini_hochberg(pvalues):
    ordered = sorted(pvalues.items(), key=lambda item: (item[1], item[0]))
    result = {}
    running = 1.0
    total = len(ordered)
    for reverse_index in range(total - 1, -1, -1):
        key, pvalue = ordered[reverse_index]
        rank = reverse_index + 1
        running = min(running, pvalue * total / rank)
        result[key] = min(1.0, running)
    return result


def choose_transform(descriptor, values, value_semantics):
    if any(value < 0 for value in values) or "minus_control" in value_semantics:
        return "signed_asinh"
    if descriptor.target_equivalent:
        return "log"
    if descriptor.bounded_range is not None:
        return "logit"
    return "log1p"


def choose_transform_scale(values, transform):
    """固定非线性变换尺度，避免全局与分组分析使用不同坐标系。"""
    if transform != "signed_asinh":
        return None
    nonzero = [abs(value) for value in values if value != 0]
    return max(median(nonzero) if nonzero else 1.0, 1e-12)


def transform_vector(values, transform, bounded_range=None, scale=None):
    if not values:
        return []
    if transform == "signed_asinh":
        if scale is None:
            raise ValueError("signed_asinh transform requires a fixed scale")
        return [math.asinh(value / max(scale, 1e-12)) for value in values]
    if transform == "log":
        return [math.log(max(value, 1e-12)) for value in values]
    if transform == "logit":
        lower, upper = bounded_range or (0.0, 1.0)
        width = max(upper - lower, 1e-12)
        result = []
        for value in values:
            probability = (value - lower) / width
            probability = min(1.0 - 1e-6, max(1e-6, probability))
            result.append(math.log(probability / (1.0 - probability)))
        return result
    return [math.log1p(max(0.0, value)) for value in values]


def solve_linear_system(matrix, vector):
    size = len(vector)
    augmented = [list(matrix[row]) + [float(vector[row])] for row in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            continue
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0:
                continue
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [augmented[index][-1] for index in range(size)]


def ridge_fit(design, target, penalty=1.0):
    if not design:
        return []
    columns = len(design[0])
    gram = [[0.0 for _ in range(columns)] for _ in range(columns)]
    rhs = [0.0 for _ in range(columns)]
    for row, value in zip(design, target):
        for left in range(columns):
            rhs[left] += row[left] * value
            for right in range(columns):
                gram[left][right] += row[left] * row[right]
    for index in range(1, columns):
        gram[index][index] += penalty
    return solve_linear_system(gram, rhs)


def predict(design, coefficients):
    return [
        sum(value * coefficient for value, coefficient in zip(row, coefficients))
        for row in design
    ]


def cross_fitted_ridge(design, target, row_ids, folds=5, penalty=1.0):
    if len(design) < max(8, folds * 2):
        mean = statistics.fmean(target) if target else 0.0
        return [mean for _ in target]
    assignments = [
        int(hashlib.sha1(row_id.encode("utf-8")).hexdigest()[:8], 16) % folds
        for row_id in row_ids
    ]
    predictions = [0.0] * len(target)
    global_mean = statistics.fmean(target)
    for fold in range(folds):
        train_indices = [
            index for index, assignment in enumerate(assignments) if assignment != fold
        ]
        test_indices = [
            index for index, assignment in enumerate(assignments) if assignment == fold
        ]
        if not test_indices:
            continue
        if len(train_indices) < 3:
            for index in test_indices:
                predictions[index] = global_mean
            continue
        coefficients = ridge_fit(
            [design[index] for index in train_indices],
            [target[index] for index in train_indices],
            penalty=penalty,
        )
        fold_predictions = predict(
            [design[index] for index in test_indices], coefficients
        )
        for index, value in zip(test_indices, fold_predictions):
            predictions[index] = value
    return predictions


def r_squared(target, predictions):
    if not target:
        return 0.0
    mean = statistics.fmean(target)
    total = sum((value - mean) ** 2 for value in target)
    residual = sum(
        (value - prediction_value) ** 2
        for value, prediction_value in zip(target, predictions)
    )
    return 1.0 - residual / total if total > 0 else 0.0


def through_origin_effect(delta_x, delta_y):
    denominator = sum(value * value for value in delta_x)
    if denominator <= 0:
        return 0.0, None
    slope = sum(x * y for x, y in zip(delta_x, delta_y)) / denominator
    if len(delta_x) < 3:
        return slope, None
    residual_ss = sum((y - slope * x) ** 2 for x, y in zip(delta_x, delta_y))
    standard_error = math.sqrt(
        max(0.0, residual_ss / (len(delta_x) - 1) / denominator)
    )
    return slope, (slope - 1.96 * standard_error, slope + 1.96 * standard_error)
