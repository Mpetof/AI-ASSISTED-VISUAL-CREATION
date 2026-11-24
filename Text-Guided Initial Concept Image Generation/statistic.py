import numpy as np
import pandas as pd

from scipy.stats import wilcoxon, page_trend_test
from statsmodels.stats.multitest import multipletests


def task_level_means(records):

    df = pd.DataFrame(records)

    keys = [
        "model", "task", "condition", "metric",
        "category", "complexity",
    ]


    metadata_counts = df.groupby(["model", "task"])[
        ["category", "complexity"]
    ].nunique(dropna=False)

    if (metadata_counts > 1).any().any():
        raise ValueError("Inconsistent task metadata.")

    return (
        df.groupby(keys, dropna=False)["value"]
        .agg(
            value="mean",
            n_valid="count",
            n_total="size",
        )
        .reset_index()
    )


def paired_table(
    task_scores,
    model,
    metric,
    conditions,
    category=None,
    complexity=None,
):

    selected = task_scores[
        (task_scores["model"] == model)
        & (task_scores["metric"] == metric)
        & task_scores["condition"].isin(conditions)
    ].copy()

    if category is not None:
        selected = selected[selected["category"] == category]
    if complexity is not None:
        selected = selected[selected["complexity"] == complexity]

    if selected.duplicated(["task", "condition"]).any():
        raise ValueError("Expected one score per task and condition.")

    table = selected.pivot(
        index="task", columns="condition", values="value"
    ).reindex(columns=list(conditions))

    complete = table.dropna()

    if len(complete) < 2:
        raise ValueError("At least two complete tasks are required.")

    return complete, {
        "n_tasks_available": len(table),
        "n_tasks_used": len(complete),
        "n_tasks_excluded": len(table) - len(complete),
    }


def bootstrap_mean(values, n_boot=20000, seed=42):

    values = np.asarray(values, dtype=float)

    if values.ndim != 1 or len(values) < 2:
        raise ValueError("Provide at least two task-level values.")
    if not np.isfinite(values).all():
        raise ValueError("Resolve missing/non-finite values first.")

    rng = np.random.default_rng(seed)
    samples = rng.choice(
        values, size=(n_boot, len(values)), replace=True
    )
    estimate = float(values.mean())
    errors = samples.mean(axis=1) - estimate

    return {
        "mean": estimate,
        "ci95_low": float(estimate - np.quantile(errors, 0.975)),
        "ci95_high": float(estimate - np.quantile(errors, 0.025)),
        "errors": errors,
    }


def compare_conditions(
    task_scores,
    model,
    metric,
    condition_a="H",
    condition_b="FL",
    alternative="greater",
    **filters,
):

    table, counts = paired_table(
        task_scores, model, metric,
        [condition_a, condition_b], **filters,
    )

    differences = (
        table[condition_a] - table[condition_b]
    ).to_numpy()

    boot = bootstrap_mean(differences)

    ranked_differences = np.round(differences, 12)

    if np.all(ranked_differences == 0):
        statistic, p_value = 0.0, 1.0
    else:
        test = wilcoxon(
            ranked_differences,
            alternative=alternative,
            zero_method="pratt",
            method="auto",
        )
        statistic, p_value = float(test.statistic), float(test.pvalue)

    return {
        "model": model,
        "metric": metric,
        "comparison": f"{condition_a} - {condition_b}",
        **counts,
        "mean_a": float(table[condition_a].mean()),
        "mean_b": float(table[condition_b].mean()),
        "mean_difference": boot["mean"],
        "ci95_low": boot["ci95_low"],
        "ci95_high": boot["ci95_high"],
        "wilcoxon_statistic": statistic,
        "alternative": alternative,
        "p_value": p_value,
    }


def cumulative_trend(
    task_scores,
    model,
    metric="total_satisfaction",
    **filters,
):

    conditions = ["C0", "C1", "C2", "C3"]
    table, counts = paired_table(
        task_scores, model, metric, conditions, **filters
    )

    values = table.to_numpy()

    if np.all(values == values[:, :1]):
        statistic, p_value = 0.0, 1.0
    else:
        test = page_trend_test(
            values,
            predicted_ranks=[1, 2, 3, 4],
            method="asymptotic",
        )
        statistic, p_value = float(test.statistic), float(test.pvalue)

    return {
        "model": model,
        "metric": metric,
        **counts,
        "condition_means": table.mean().to_dict(),
        "page_L": statistic,
        "p_value": p_value,
    }


def quality_noninferiority(
    task_scores,
    model,
    metric,
    baseline="FL",
    margin=None,
    alpha=0.05,
    **filters,
):

    margins = {"qalign": 0.2, "musiq": 5.0}
    margin = margins[metric] if margin is None else float(margin)


    table, counts = paired_table(
        task_scores, model, metric, ["H", baseline], **filters
    )
    differences = (table["H"] - table[baseline]).to_numpy()

    if np.std(differences, ddof=1) == 0:
        raise ValueError("Degenerate differences: uncertainty is not estimable.")

    boot = bootstrap_mean(differences)
    errors = boot["errors"]

    lower_bound = float(
        boot["mean"] - np.quantile(errors, 1 - alpha)
    )

    p_value = float(
        (1 + np.sum(errors >= boot["mean"] + margin))
        / (len(errors) + 1)
    )

    return {
        "model": model,
        "metric": metric,
        "comparison": f"H - {baseline}",
        **counts,
        "mean_difference": boot["mean"],
        "ci95_low": boot["ci95_low"],
        "ci95_high": boot["ci95_high"],
        "one_sided_lower_bound": lower_bound,
        "margin": margin,
        "p_value": p_value,
        "noninferior_unadjusted": bool(lower_bound > -margin),
    }


def holm_by_family(results, hypothesis_families, alpha=0.05):

    df = pd.DataFrame(results).copy()
    df["family"] = df["hypothesis"].map(hypothesis_families)

    if df["family"].isna().any():
        raise ValueError("Every hypothesis needs a family assignment.")

    p = df["p_value"].to_numpy(dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid p-values.")

    df["p_holm"] = np.nan
    df["reject_holm"] = False

    for _, group in df.groupby("family"):
        reject, adjusted, _, _ = multipletests(
            group["p_value"].to_numpy(),
            alpha=alpha,
            method="holm",
        )
        df.loc[group.index, "p_holm"] = adjusted
        df.loc[group.index, "reject_holm"] = reject

    if "noninferior_unadjusted" in df:
        mask = df["noninferior_unadjusted"].notna()
        df.loc[mask, "noninferior_adjusted"] = (
            df.loc[mask, "noninferior_unadjusted"].astype(bool)
            & df.loc[mask, "reject_holm"].astype(bool)
        )

    return df