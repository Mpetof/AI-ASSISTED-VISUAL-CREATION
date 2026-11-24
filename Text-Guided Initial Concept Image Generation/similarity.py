import torch
import torch.nn.functional as F
from PIL import Image
from dreamsim import dreamsim

# Load once and reuse.
device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = dreamsim(
    pretrained=True,
    dreamsim_type="ensemble",
    device=device,
)
model.eval()


def image_embedding(image_path):

    with Image.open(image_path) as image:
        image_tensor = preprocess(image.convert("RGB")).to(device)

    with torch.inference_mode():
        embedding = model.embed(image_tensor)

    return embedding.float().cpu()


def dreamsim_distance(image_path_a, image_path_b):
rity.

    embedding_a = image_embedding(image_path_a)
    embedding_b = image_embedding(image_path_b)

    distance = 1.0 - F.cosine_similarity(
        embedding_a,
        embedding_b,
        dim=-1,
    )

    return float(distance.clamp(min=0.0, max=2.0).item())


def mean_pairwise_dreamsim(image_paths):

    embeddings = torch.cat(
        [image_embedding(path) for path in image_paths],
        dim=0,
    )
    embeddings = F.normalize(embeddings, p=2, dim=-1)

    distances = (1.0 - embeddings @ embeddings.T).clamp(
        min=0.0,
        max=2.0,
    )

    n = len(image_paths)
    rows, cols = torch.triu_indices(n, n, offset=1)
    values = distances[rows, cols]

    return {
        "mean_distance": float(values.mean().item()),
        "pair_count": int(values.numel()),
        "pairwise_distances": {
            (i, j): float(distances[i, j].item())
            for i, j in zip(rows.tolist(), cols.tolist())
        },
    }