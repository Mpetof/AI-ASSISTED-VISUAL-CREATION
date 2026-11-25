"""Record field: Usable. Threshold per run before task averaging."""
import math
from .common import summarize_records


def usable_flag(iou, lpips_score, no_output=False, iou_min=0.90, lpips_max=0.10):
    if no_output or iou is None or lpips_score is None:
        return 0
    if not math.isfinite(iou) or not math.isfinite(lpips_score):
        raise ValueError('Invalid score.')
    return int(iou >= iou_min and lpips_score <= lpips_max)


def summarize_usable_rate(records, tool):
    recalculated = []
    for row in records:
        if row['tier']=='T1' and row['tool']==tool:
            r = dict(row)
            r['Usable'] = usable_flag(r.get('IoU'),r.get('LPIPS'),r.get('no_output',False))
            recalculated.append(r)
    result = summarize_records(recalculated,'Usable','T1',tool)
    result['percent'] = None if result['mean'] is None else 100*result['mean']
    return result
