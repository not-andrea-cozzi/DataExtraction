from __future__ import annotations

from typing import Dict

import torch
from torch_geometric.data import Data

_UINT8_MAX = 255
_INT16_MAX = 32767


def _assert_binary(t: torch.Tensor, name: str) -> None:
    if t.numel() and not torch.all((t == 0.0) | (t == 1.0)).item():
        raise ValueError(f"compress: '{name}' non binario.")


def _assert_uint8(t: torch.Tensor, name: str) -> None:
    if t.numel() and (int(t.min()) < 0 or int(t.max()) > _UINT8_MAX):
        raise ValueError(f"compress: '{name}' fuori da uint8.")


def _assert_int16_range(t: torch.Tensor, name: str) -> None:
    if t.numel() and (int(t.min()) < 0 or int(t.max()) > _INT16_MAX):
        raise ValueError(f"compress: '{name}' fuori da int16 [0,{_INT16_MAX}].")


def _time_by_edge_type(time_t: torch.Tensor, edge_attr: torch.Tensor) -> Dict[int, float]:
    types = edge_attr.argmax(dim=1)
    out: Dict[int, float] = {}
    for tid in torch.unique(types).tolist():
        vals = time_t[types == tid]
        first = float(vals[0])
        if not torch.allclose(vals, torch.full_like(vals, first)):
            raise ValueError(f"compress: 'time' non costante per edge_type={tid}.")
        out[int(tid)] = first
    return out


def compress_position_data(data: Data) -> Data:
    c = Data()
    time_t = edge_attr = None

    for key, v in data:
        if not torch.is_tensor(v):
            c[key] = v
            continue
        if key == "time":
            time_t = v
        elif key == "x":
            _assert_binary(v[:, :-1], "x[:, :-1]")
            c.x_binary = v[:, :-1].to(torch.bool)
            c.x_continuous = v[:, -1:].to(torch.float32)
        elif key == "edge_index":
            _assert_int16_range(v, key)
            c[key] = v.to(torch.int16)
        elif key == "legal_move_indices":
            _assert_int16_range(v, key)
            c[key] = v.to(torch.int16)
        elif key == "event_ids":
            _assert_uint8(v, key)
            c[key] = v.to(torch.uint8)
        elif key == "edge_attr":
            _assert_binary(v, key)
            c[key] = v.to(torch.bool)
            edge_attr = v
        else:
            c[key] = v

    if time_t is not None:
        if edge_attr is None:
            raise ValueError("compress: 'time' senza 'edge_attr'.")
        tbt = _time_by_edge_type(time_t, edge_attr)
        ids = sorted(tbt)
        c.time_type_ids = torch.tensor(ids, dtype=torch.long)
        c.time_type_values = torch.tensor([tbt[i] for i in ids], dtype=torch.float32)
    return c


def decompress_position_data(data: Data) -> Data:
    d = Data()
    skip = {"x_binary", "x_continuous", "time_type_ids", "time_type_values"}
    for key, v in data:
        if key in skip:
            continue
        if not torch.is_tensor(v):
            d[key] = v
        elif key == "edge_index":
            d[key] = v.to(torch.long)
        elif key == "legal_move_indices":
            d[key] = v.to(torch.long)
        elif key == "event_ids":
            d[key] = v.to(torch.long)
        elif key == "edge_attr":
            d[key] = v.to(torch.float32)
        else:
            d[key] = v

    d.x = torch.cat([data.x_binary.to(torch.float32), data.x_continuous.to(torch.float32)], dim=1)

    lookup = dict(zip(data.time_type_ids.tolist(), data.time_type_values.tolist()))
    types = d.edge_attr.argmax(dim=1)
    time_t = torch.zeros(types.shape[0], dtype=torch.float32)
    for tid, val in lookup.items():
        time_t[types == tid] = val
    d.time = time_t
    return d