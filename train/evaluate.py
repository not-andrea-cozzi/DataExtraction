from __future__ import annotations

import pandas as pd
import torch

COLUMNS = (
    "mate_n",
    "n_legal",
    "clock_norm",
    "policy_valid",
    "policy_correct",
    "value_true",
    "value_pred",
)


@torch.no_grad()
def collect_predictions(
    model,
    loader,
    device: str,
    mate_range_min: int,
    use_amp: bool = False,
) -> pd.DataFrame:
    amp = use_amp and device.startswith("cuda")
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    rows = {name: [] for name in COLUMNS}
    empty = torch.tensor(-1, device=device)
    model.eval()

    for batch_data, node_offsets, legal_move_indices, policy_targets, mate_targets in loader:
        batch_data = batch_data.to(device)
        legal = [m.to(device) for m in legal_move_indices]

        with torch.autocast(device_type=device_type, enabled=amp):
            policy_logits, value_logits = model(batch_data, node_offsets, legal)

        value_pred = value_logits.argmax(dim=1).cpu().tolist()
        value_true = mate_targets.tolist()
        clock = batch_data.x[node_offsets, 2].float().cpu().tolist()
        policy_pred = torch.stack(
            [l.argmax() if l.numel() else empty for l in policy_logits]
        ).cpu().tolist()

        for i, target in enumerate(policy_targets):
            valid = target.numel() > 0
            rows["mate_n"].append(value_true[i] + mate_range_min)
            rows["n_legal"].append(int(legal_move_indices[i].numel()))
            rows["clock_norm"].append(clock[i])
            rows["policy_valid"].append(valid)
            rows["policy_correct"].append(bool(valid and policy_pred[i] in target.tolist()))
            rows["value_true"].append(value_true[i])
            rows["value_pred"].append(value_pred[i])

    return pd.DataFrame(rows)