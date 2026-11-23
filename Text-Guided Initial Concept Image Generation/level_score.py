import torch
import t2v_metrics


scorer = t2v_metrics.VQAScore(model="clip-flant5-xxl")


def evaluate_image(image_path, questions, threshold=0.5):

    ids = [q["id"] for q in questions]

    known_ids = set(ids)
    for q in questions:
        for parent in q.get("parents", []):
            if parent not in known_ids:
                raise ValueError(f"Unknown parent: {parent}")

    likelihoods = {}

    with torch.inference_mode():
        for q in questions:
            score = scorer(
                images=[image_path],
                texts=[q["statement"]],
            )
            likelihoods[q["id"]] = float(score[0, 0].item())

    gated_scores = {}

    for q in questions:
        parents_satisfied = all(
            likelihoods[parent] >= threshold
            for parent in q.get("parents", [])
        )
        gated_scores[q["id"]] = (
            likelihoods[q["id"]] if parents_satisfied else 0.0
        )


    level_scores = {}

    for level in sorted({q["level"] for q in questions}):
        values = [
            gated_scores[q["id"]]
            for q in questions
            if q["level"] == level
        ]
        level_scores[level] = sum(values) / len(values)

    return likelihoods, gated_scores, level_scores


def mean_level_scores(image_paths, questions, threshold=0.5):

    if not image_paths:
        raise ValueError("At least one image is required.")

    results = [
        evaluate_image(path, questions, threshold)[2]
        for path in image_paths
    ]

    return {
        level: sum(result[level] for result in results) / len(results)
        for level in results[0]
    }


def incremental_gains(cumulative_scores):

    return {
        k: cumulative_scores[k][k] - cumulative_scores[k - 1][k]
        for k in (1, 2, 3)
    }