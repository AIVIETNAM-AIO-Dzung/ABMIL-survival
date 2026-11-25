import numpy as np
import torch
from lifelines.utils import concordance_index


def cox_ph_loss(risk, time, event):
    """Negative Cox partial log-likelihood (xử lý trùng thời gian kiểu Breslow).

    risk:  (B,) log-hazard dự đoán (càng cao càng nguy hiểm)
    time:  (B,) thời gian sống còn / theo dõi
    event: (B,) 1 = chết, 0 = censored
    Trả về None nếu batch không có biến cố nào (không có tín hiệu gradient).
    """
    risk, time, event = risk.view(-1), time.view(-1).to(risk.device), event.view(-1).to(risk.device).float()
    n_events = event.sum()
    if n_events == 0:
        return None
    # sắp time giảm dần -> risk set của mẫu i = các mẫu từ 0..i (cộng dồn)
    order = torch.argsort(time, descending=True)
    risk, time, event = risk[order], time[order], event[order]
    log_cum = torch.logcumsumexp(risk, dim=0)
    # Breslow: các mẫu cùng time dùng chung risk set đầy đủ (vị trí cuối của nhóm trùng)
    _, inverse, counts = torch.unique_consecutive(time, return_inverse=True, return_counts=True)
    last_idx = torch.cumsum(counts, 0) - 1
    log_cum = log_cum[last_idx[inverse]]
    return -((risk - log_cum) * event).sum() / n_events


def c_index(risk, time, event):
    """Harrell's C-index. risk cao = sống ngắn, nên đưa -risk vào lifelines."""
    risk, time, event = (np.asarray(x, dtype=float) for x in (risk, time, event))
    return concordance_index(time, -risk, event)
