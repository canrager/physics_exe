from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_parquet(path: Path, columns: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name, values in columns.items():
        if isinstance(values, np.ndarray):
            arrays[name] = pa.array(values.tolist() if values.dtype == object else values)
        else:
            arrays[name] = pa.array(values)
    table = pa.Table.from_pydict(arrays)
    pq.write_table(table, path, compression="zstd")


def read_parquet(path: Path) -> dict[str, Any]:
    table = pq.read_table(path)
    data: dict[str, Any] = {}
    for name in table.column_names:
        column = table[name]
        if pa.types.is_string(column.type):
            data[name] = np.array(column.to_pylist(), dtype=object)
        else:
            data[name] = column.to_numpy(zero_copy_only=False)
    return data

