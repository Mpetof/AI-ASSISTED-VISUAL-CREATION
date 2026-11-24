from contextlib import nullcontext
from itertools import combinations

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(
    "IDEA-Research/grounding-dino-tiny"
)
detector = AutoModelForZeroShotObjectDetection.from_pretrained(
    "IDEA-Research/grounding-dino-tiny"
).to(device)
detector.eval()

sam_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
sam_config = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam_model = build_sam2(
    sam_config,
    sam_checkpoint,
    device=device,
)
sam_model.eval()
sam_predictor = SAM2ImagePredictor(sam_model)


def extract_subject_mask(
    image_path,
    subject,
    canvas_size=(1024, 1024),
    box_threshold=0.4,
    text_threshold=0.3,
):

    with Image.open(image_path) as source:
        image = source.convert("RGB")

    text = subject.strip().lower().rstrip(".") + "."

    inputs = processor(
        images=image,
        text=text,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        outputs = detector(**inputs)

        detection = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(image.height, image.width)],
        )[0]

    if len(detection["boxes"]) == 0:
        return None

    best_index = int(detection["scores"].argmax().item())
    box = detection["boxes"][best_index].detach().cpu().numpy()

    box[[0, 2]] = np.clip(box[[0, 2]], 0, image.width)
    box[[1, 3]] = np.clip(box[[1, 3]], 0, image.height)

    if box[2] <= box[0] or box[3] <= box[1]:
        return None

    use_bf16 = (
        device == "cuda" and torch.cuda.is_bf16_supported()
    )
    precision_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16 else nullcontext()
    )

    with torch.inference_mode(), precision_context:
        sam_predictor.set_image(np.asarray(image))

        masks, _, _ = sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box[None, :],
            multimask_output=False,
        )

    mask = np.asarray(masks).reshape(
        -1, image.height, image.width
    )[0].astype(bool)

    if not mask.any():
        return None

    mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
    mask_image = mask_image.resize(
        canvas_size,
        resample=Image.Resampling.NEAREST,
    )
    normalized_mask = np.asarray(mask_image) > 0

    return normalized_mask if normalized_mask.any() else None


def mask_iou(mask_a, mask_b):

    if mask_a is None or mask_b is None:
        return None

    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)

    if not mask_a.any() or not mask_b.any():
        return None

    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()

    return float(intersection / union)


def mean_pairwise_subject_iou(
    image_paths,
    subject,
    canvas_size=(1024, 1024),
    box_threshold=0.4,
    text_threshold=0.3,
):

    masks = [
        extract_subject_mask(
            image_path=path,
            subject=subject,
            canvas_size=canvas_size,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        for path in image_paths
    ]

    pairwise_ious = {
        (i, j): mask_iou(masks[i], masks[j])
        for i, j in combinations(range(len(masks)), 2)
    }

    valid_ious = [
        value for value in pairwise_ious.values()
        if value is not None
    ]

    failed_indices = [
        i for i, mask in enumerate(masks)
        if mask is None
    ]
    total_pairs = len(pairwise_ious)
    excluded_pairs = total_pairs - len(valid_ious)

    return {
        "mean_iou": (
            sum(valid_ious) / len(valid_ious)
            if valid_ious else None
        ),
        "valid_pairs": len(valid_ious),
        "excluded_pairs": excluded_pairs,
        "excluded_pair_fraction": excluded_pairs / total_pairs,
        "failed_image_indices": failed_indices,
        "failed_image_fraction": len(failed_indices) / len(masks),
        "pairwise_ious": pairwise_ious,
    }