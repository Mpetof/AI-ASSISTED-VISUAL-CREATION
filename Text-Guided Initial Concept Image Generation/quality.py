import numpy as np
import torch
import pyiqa
from scipy.stats import t as student_t

device = torch.device("cuda")
metrics = {
    "qalign": pyiqa.create_metric("qalign", device=device).eval(),
    "musiq": pyiqa.create_metric("musiq", device=device).eval(),
}


RAW_RANGES = {
    "qalign": {"min": 1.0, "max": 5.0},
    "musiq": {"min": 0.0, "max": 100.0},
}


MARGINS_RAW = {
    "qalign": 0.05 * (5.0 - 1.0),
    "musiq": 0.05 * (100.0 - 0.0),
}

MARGINS_NORM = {
    "qalign": 0.05,
    "musiq": 0.05,
}

def normalize_score(val, metric_name):

    r = RAW_RANGES[metric_name]
    return (val - r["min"]) / (r["max"] - r["min"])

def evaluate_image_quality(image_path):

    raw_scores = {}
    norm_scores = {}
    with torch.inference_mode():
        for name, metric in metrics.items():
            raw_val = float(metric(str(image_path)).item())
            if not np.isfinite(raw_val):
                raise ValueError(f"Non-finite {name} score for {image_path}")
            raw_scores[name] = raw_val
            norm_scores[name] = normalize_score(raw_val, name)
    return raw_scores, norm_scores

def mean_quality_scores(image_paths):

    if not image_paths:
        raise ValueError("At least one image is required.")
    raw_list = []
    norm_list = []
    for path in image_paths:
        raw, norm = evaluate_image_quality(path)
        raw_list.append(raw)
        norm_list.append(norm)
    raw_mean = {
        name: float(np.mean([r[name] for r in raw_list]))
        for name in metrics
    }
    norm_mean = {
        name: float(np.mean([r[name] for r in norm_list]))
        for name in metrics
    }

    normalized_quality_proxies = [norm_mean["qalign"], norm_mean["musiq"]]
    return raw_mean, norm_mean, normalized_quality_proxies

def paired_noninferiority(
    hierarchical_scores,
    baseline_scores,
    margin,
    alpha=0.05,
):

    h = np.asarray(hierarchical_scores, dtype=float)
    b = np.asarray(baseline_scores, dtype=float)
    if h.ndim != 1 or b.ndim != 1 or h.shape != b.shape:
        raise ValueError("Inputs must be matching one-dimensional arrays.")
    if len(h) < 2:
        raise ValueError("At least two paired tasks are required.")
    if not np.isfinite(h).all() or not np.isfinite(b).all():
        raise ValueError("Scores must be finite; resolve missing data first.")
    if margin <= 0 or not 0 < alpha < 1:
        raise ValueError("Require margin > 0 and 0 < alpha < 1.")
    differences = h - b
    n = len(differences)
    mean_difference = float(differences.mean())
    standard_error = float(
        differences.std(ddof=1) / np.sqrt(n)
    )
    if standard_error == 0:
        raise ValueError(
            "All task differences are identical; "
            "the t-based uncertainty estimate is degenerate."
        )
    degrees_of_freedom = n - 1
    t_statistic = (mean_difference + margin) / standard_error
    p_value = float(
        student_t.sf(t_statistic, df=degrees_of_freedom)
    )
    critical_value = student_t.ppf(
        1 - alpha, df=degrees_of_freedom
    )
    lower_bound = float(
        mean_difference - critical_value * standard_error
    )
    return {
        "n_tasks": n,
        "mean_H": float(h.mean()),
        "mean_baseline": float(b.mean()),
        "mean_difference": mean_difference,
        "margin": float(margin),
        "lower_bound": lower_bound,
        "one_sided_confidence": 1 - alpha,
        "p_value": p_value,
        "noninferior": bool(lower_bound > -margin),
    }

def validate_image_quality(
    images_by_task,
    baselines=("FF", "FL", "OA"),
    alpha=0.05,
    bonferroni=True,
    use_normalized=True, 
):
    baselines = tuple(baselines)

    conditions = ("H",) + baselines
    task_ids = sorted(images_by_task)
    for task_id in task_ids:
        for condition in conditions:
            paths = images_by_task[task_id].get(condition)
            if not paths:
                raise ValueError(f"Missing images: {task_id}, {condition}")

    task_scores = {}
    for task_id in task_ids:
        task_scores[task_id] = {}
        for cond in conditions:
            raw_mean, norm_mean, nqp = mean_quality_scores(images_by_task[task_id][cond])
            task_scores[task_id][cond] = {
                "raw": raw_mean,
                "norm": norm_mean,
                "normalized_quality_proxies": nqp
            }

    n_tests = len(metrics) * len(baselines)
    test_alpha = alpha / n_tests if bonferroni else alpha
    comparisons = {}
    MARGINS = MARGINS_NORM if use_normalized else MARGINS_RAW

    for metric_name in metrics:
        comparisons[metric_name] = {}
        h_scores = [
            task_scores[t]["H"]["norm"][metric_name] if use_normalized else task_scores[t]["H"]["raw"][metric_name]
            for t in task_ids
        ]
        for baseline in baselines:
            b_scores = [
                task_scores[t][baseline]["norm"][metric_name] if use_normalized else task_scores[t][baseline]["raw"][metric_name]
                for t in task_ids
            ]
            result = paired_noninferiority(
                h_scores,
                b_scores,
                margin=MARGINS[metric_name],
                alpha=test_alpha,
            )
            result["adjusted_p_value"] = min(
                1.0,
                result["p_value"] * n_tests
                if bonferroni else result["p_value"],
            )
            comparisons[metric_name][baseline] = result
    return {
        "task_scores": task_scores,
        "comparisons": comparisons,
        "correction": "bonferroni" if bonferroni else "none",
        "all_noninferior": all(
            result["noninferior"]
            for metric_results in comparisons.values()
            for result in metric_results.values()
        ),
    }
