import torch
import t2v_metrics


scorer = t2v_metrics.VQAScore(model="clip-flant5-xxl")


def evaluate_constraints(
    image_path,
    questions,
    parent_threshold=0.5,
    satisfaction_threshold=0.5,
):

    likelihoods = {}

    with torch.inference_mode():
        for q in questions:
            score = scorer(
                images=[image_path],
                texts=[q["statement"]],
                question_template=(
                    'Does this figure show "{}"? '
                    'Please answer yes or no.'
                ),
                answer_template="Yes",
            )
            likelihoods[q["id"]] = float(score[0, 0].item())


    gated_scores = {}

    for q in questions:
        parents_satisfied = all(
            likelihoods[parent] >= parent_threshold
            for parent in q.get("parents", [])
        )
        gated_scores[q["id"]] = (
            likelihoods[q["id"]] if parents_satisfied else 0.0
        )


    satisfied = {
        qid: int(score >= satisfaction_threshold)
        for qid, score in gated_scores.items()
    }

    return likelihoods, gated_scores, satisfied


def constraint_retention(
    image_before,
    image_after,
    questions,
    current_level,
    parent_threshold=0.5,
    satisfaction_threshold=0.5,
    eligible_ids=None,
):

    if current_level not in (1, 2, 3):
        raise ValueError("current_level must be 1, 2, or 3.")

    if eligible_ids is not None:
        eligible_ids = set(eligible_ids)
        unknown = eligible_ids - {q["id"] for q in questions}
        if unknown:
            raise ValueError(f"Unknown eligible IDs: {sorted(unknown)}")


    before = evaluate_constraints(
        image_before,
        questions,
        parent_threshold,
        satisfaction_threshold,
    )[2]

    after = evaluate_constraints(
        image_after,
        questions,
        parent_threshold,
        satisfaction_threshold,
    )[2]


    old_ids = [
        q["id"]
        for q in questions
        if q["level"] < current_level
        and (eligible_ids is None or q["id"] in eligible_ids)
    ]


    previous_count = sum(before[qid] for qid in old_ids)


    retained_count = sum(
        before[qid] * after[qid]
        for qid in old_ids
    )

    retention = (
        retained_count / previous_count
        if previous_count > 0
        else None
    )

    lost_ids = [
        qid
        for qid in old_ids
        if before[qid] == 1 and after[qid] == 0
    ]

    return retention, previous_count, retained_count, lost_ids


def mean_constraint_retention(
    image_pairs,
    questions,
    current_level,
    parent_threshold=0.5,
    satisfaction_threshold=0.5,
    eligible_ids=None,
):



    results = [
        constraint_retention(
            image_before=before,
            image_after=after,
            questions=questions,
            current_level=current_level,
            parent_threshold=parent_threshold,
            satisfaction_threshold=satisfaction_threshold,
            eligible_ids=eligible_ids,
        )
        for before, after in image_pairs
    ]

    valid_scores = [
        result[0]
        for result in results
        if result[0] is not None
    ]

    excluded_count = len(results) - len(valid_scores)

    return {
        "mean_retention": (
            sum(valid_scores) / len(valid_scores)
            if valid_scores
            else None
        ),
        "valid_pairs": len(valid_scores),
        "excluded_pairs": excluded_count,
        "excluded_fraction": excluded_count / len(results),
    }