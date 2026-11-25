
from .common import summarize_records


def load_siglip(device='cuda', checkpoint='google/siglip-base-patch16-224'):
    from transformers import AutoProcessor, AutoModel
    return (AutoProcessor.from_pretrained(checkpoint),
            AutoModel.from_pretrained(checkpoint).to(device).eval())


def target_identity(image, target_label, candidate_labels, processor, model,
                    template='This is an image of {}.'):

    import torch

    if image is None:
        return {'Identity':0,'predicted_label':None}
    model.eval()
    inputs = processor(images=image,text=[template.format(c) for c in candidate_labels],
                       padding='max_length',truncation=True,return_tensors='pt')
    inputs = inputs.to(next(model.parameters()).device)
    with torch.inference_mode():
        logits = model(**inputs).logits_per_image[0]
        index = int(logits.argmax().item())
    label = candidate_labels[index]
    return {'Identity':int(label==target_label),'predicted_label':label}


def summarize_identity_accuracy(records, tool):
    adjusted=[]
    for row in records:
        if row['tier']=='T2' and row['tool']==tool:
            r=dict(row)
            if r.get('no_output',False):
                r['Identity']=0
            adjusted.append(r)
    result = summarize_records(adjusted,'Identity','T2',tool)
    result['percent'] = None if result['mean'] is None else 100*result['mean']
    return result
