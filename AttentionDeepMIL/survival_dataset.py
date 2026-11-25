"""Bước 5: mỗi bệnh nhân là một bag feature (nối feature mọi slide) + (time, event)."""
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SurvivalBags(Dataset):
    def __init__(self, labels_csv, feat_dir, case_ids=None, max_patches=None, seed=0):
        """
        labels_csv: output của scripts/02_make_labels.py (case_id, slide_ids, time, event, site)
        feat_dir:   thư mục chứa <slide_id>.pt (features/pt_files của CLAM)
        case_ids:   danh sách bệnh nhân thuộc split này (None = tất cả)
        max_patches: nếu đặt, lấy ngẫu nhiên tối đa chừng này patch mỗi lần (dùng khi train)
        """
        df = pd.read_csv(labels_csv)
        if case_ids is not None:
            df = df[df["case_id"].isin(set(case_ids))]
        self.feat_dir = feat_dir
        self.max_patches = max_patches
        self.rng = np.random.default_rng(seed)

        # chỉ giữ slide đã có feature; bỏ bệnh nhân không còn slide nào
        rows = []
        for r in df.itertuples():
            slides = [s for s in r.slide_ids.split(";")
                      if os.path.isfile(os.path.join(feat_dir, s + ".pt"))]
            if slides:
                rows.append((r.case_id, slides, float(r.time), int(r.event)))
        if len(rows) < len(df):
            print(f"[SurvivalBags] bỏ {len(df) - len(rows)} bệnh nhân không có feature")
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    @property
    def times(self):
        return np.array([r[2] for r in self.rows])

    @property
    def events(self):
        return np.array([r[3] for r in self.rows])

    def __getitem__(self, idx):
        case_id, slides, time, event = self.rows[idx]
        bag = torch.cat([torch.load(os.path.join(self.feat_dir, s + ".pt"), weights_only=True)
                         for s in slides], dim=0).float()
        if self.max_patches and bag.shape[0] > self.max_patches:
            keep = self.rng.choice(bag.shape[0], self.max_patches, replace=False)
            bag = bag[torch.from_numpy(np.sort(keep))]
        return bag, torch.tensor(time), torch.tensor(event), case_id


def collate_bags(batch):
    """Giữ bag dạng list (kích thước khác nhau), time/event stack thành tensor."""
    bags, times, events, case_ids = zip(*batch)
    return list(bags), torch.stack(times), torch.stack(events), list(case_ids)
