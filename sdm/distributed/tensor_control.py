# SPDX-License-Identifier: Apache-2.0
"""Typed control transport: JSON descriptors and raw tensor storage, no pickle.

Numeric payloads stay on the collective device. Only the small descriptor is
decoded on the host. CPU/Gloo implements the same protocol for CPU sessions.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Callable, TypeVar

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup

from sdm import (
    CategoricalTensor,
    EnsembleTable,
    Stype,
    StringTensor,
    TableTensor,
)
from sdm.tensor.table import TableSchema

_T = TypeVar("_T")
_FAILED_TRANSFERS: list[dict[str, Any]] = []


def describe(value: Any, leaves: list[Tensor]) -> Any:
    """Explicit schema for the values used by sharded preprocessing/control."""
    if isinstance(value, Stype):
        return ["stype", str(value)]
    if value is None or type(value) in (str, int, float, bool):
        return ["scalar", value]
    if isinstance(value, torch.dtype):
        return ["dtype", str(value).removeprefix("torch.")]
    if isinstance(value, TableSchema):
        return ["schema", describe(dict(value.columns), leaves)]
    if isinstance(value, EnsembleTable):
        return [
            "ensemble",
            describe(list(value._iter_groups()), leaves),
            describe(value._locations, leaves),
        ]
    if isinstance(value, TableTensor):
        if value.active_stypes - {Stype.numerical, Stype.categorical}:
            raise TypeError(
                "Typed preprocessing supports numerical/categorical tables"
            )
        return [
            "table",
            describe(dict(value.columns), leaves),
            describe(value.numerical, leaves),
            describe(value.categorical, leaves),
        ]
    if isinstance(value, CategoricalTensor):
        return [
            "categorical",
            describe(value.code, leaves),
            describe(tuple(value.categories), leaves),
        ]
    if isinstance(value, StringTensor):
        data, offset = value.data_offset
        return [
            "string",
            describe(data, leaves),
            describe(offset, leaves),
            describe(value.valid, leaves),
            list(value.shape),
        ]
    if type(value) is Tensor:
        index = len(leaves)
        leaves.append(value)
        return [
            "tensor",
            index,
            str(value.dtype).removeprefix("torch."),
            list(value.shape),
        ]
    if isinstance(value, dict):
        return [
            "dict",
            [
                [describe(k, leaves), describe(v, leaves)]
                for k, v in value.items()
            ],
        ]
    if type(value) in (tuple, list):
        return [
            "tuple" if type(value) is tuple else "list",
            [describe(v, leaves) for v in value],
        ]
    raise TypeError(f"Unsupported typed control value: {type(value).__name__}")


def restore(node: Any, leaves: list[Tensor]) -> Any:
    kind = node[0]
    if kind == "scalar":
        return node[1]
    if kind == "stype":
        return Stype(node[1])
    if kind == "dtype":
        return getattr(torch, node[1])
    if kind == "tensor":
        return leaves[node[1]]
    if kind == "dict":
        return {restore(k, leaves): restore(v, leaves) for k, v in node[1]}
    if kind in ("tuple", "list"):
        values = [restore(v, leaves) for v in node[1]]
        return tuple(values) if kind == "tuple" else values
    if kind == "schema":
        return TableSchema(restore(node[1], leaves))
    if kind == "ensemble":
        return EnsembleTable(
            restore(node[1], leaves), restore(node[2], leaves)
        )
    if kind == "table":
        return TableTensor(
            columns=restore(node[1], leaves),
            numerical=restore(node[2], leaves),
            categorical=restore(node[3], leaves),
        )
    if kind == "categorical":
        return CategoricalTensor(
            restore(node[1], leaves), categories=restore(node[2], leaves)
        )
    if kind == "string":
        return StringTensor(
            restore(node[1], leaves),
            restore(node[2], leaves),
            restore(node[3], leaves),
            node[4],
        )
    raise ValueError(f"Unknown typed control descriptor: {kind}")


def descriptor(value: Any) -> tuple[bytes, list[Tensor]]:
    leaves: list[Tensor] = []
    tree = describe(value, leaves)
    specs = [
        [str(t.dtype).removeprefix("torch."), list(t.shape)] for t in leaves
    ]
    return json.dumps(
        [tree, specs], ensure_ascii=False, separators=(",", ":")
    ).encode(), leaves


def fingerprint(value: Any) -> bytes:
    header, leaves = descriptor(value)
    if leaves:
        raise TypeError(
            "Control fingerprints require metadata, not tensor values"
        )
    return hashlib.sha256(header).digest()


def pack(value: Any, device: torch.device) -> Tensor:
    header, leaves = descriptor(value)
    offset = (8 + len(header) + 15) // 16 * 16
    offsets = []
    for tensor in leaves:
        offsets.append(offset)
        offset = (
            (offset + tensor.numel() * tensor.element_size() + 15) // 16 * 16
        )
    wire = torch.zeros(offset, dtype=torch.uint8, device=device)
    wire[:8].copy_(
        torch.tensor([len(header)], dtype=torch.int64, device=device).view(
            torch.uint8
        )
    )
    wire[8 : 8 + len(header)].copy_(
        torch.tensor(list(header), dtype=torch.uint8, device=device)
    )
    for tensor, start in zip(leaves, offsets, strict=True):
        raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
        wire[start : start + raw.numel()].copy_(raw)
    return wire


def unpack(wire: Tensor) -> Any:
    length = int(wire[:8].view(torch.int64).item())
    tree, specs = json.loads(bytes(wire[8 : 8 + length].cpu().tolist()))
    offset = (8 + length + 15) // 16 * 16
    leaves = []
    for dtype, shape in specs:
        dtype = getattr(torch, dtype)
        size = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        leaves.append(wire[offset : offset + size].view(dtype).reshape(shape))
        offset = (offset + size + 15) // 16 * 16
    return restore(tree, leaves)


class TensorControl:
    def __init__(self, group: ProcessGroup, device: torch.device):
        self.group = group
        self.device = (
            device
            if dist.get_backend(group) == "nccl"
            else torch.device("cpu")
        )
        self.rank = dist.get_rank(group)
        self.world = dist.get_world_size(group)
        self.poisoned = False
        self._status = torch.empty(1, dtype=torch.int32, device=self.device)

    def check(self) -> None:
        if self.poisoned:
            raise RuntimeError(
                "Typed transport failed; use a fresh distributed session"
            )

    def broadcast_tensor(self, tensor: Tensor, source: int) -> None:
        try:
            dist.broadcast(tensor, src=source, group=self.group)
        except BaseException:
            self.poisoned = True
            _FAILED_TRANSFERS.append({"broadcast": tensor})
            raise

    def reduce(self, value: int, op: dist.ReduceOp = dist.ReduceOp.MAX) -> int:
        self.check()
        self._status.fill_(value)
        try:
            dist.all_reduce(self._status, op=op, group=self.group)
            return int(self._status.item())
        except BaseException:
            self.poisoned = True
            _FAILED_TRANSFERS.append({"status": self._status})
            raise

    def gather_bytes(self, value: bytes) -> list[bytes]:
        self.check()
        size = torch.tensor(
            [len(value)], dtype=torch.int64, device=self.device
        )
        sizes = [torch.empty_like(size) for _ in range(self.world)]
        dist.all_gather(sizes, size, group=self.group)
        lengths = torch.cat(sizes).cpu().tolist()
        buffer = torch.zeros(
            max(1, max(lengths)), dtype=torch.uint8, device=self.device
        )
        if value:
            buffer[: len(value)] = torch.tensor(
                list(value), dtype=torch.uint8, device=self.device
            )
        received = [torch.empty_like(buffer) for _ in range(self.world)]
        dist.all_gather(received, buffer, group=self.group)
        return [
            bytes(row[:length])
            for row, length in zip(
                torch.stack(received).cpu().tolist(), lengths, strict=True
            )
        ]

    def local_result(self, operation: Callable[[], _T]) -> _T:
        self.check()
        result = None
        error = ""
        try:
            result = operation()
        except Exception as exc:
            error = f"rank {self.rank}: {type(exc).__name__}: {exc}"
        if self.reduce(int(bool(error))):
            errors = self.gather_bytes(error.encode())
            raise RuntimeError(next(e.decode() for e in errors if e))
        return result

    def broadcast(self, value: Any, source: int) -> Any:
        self.check()
        is_source = dist.get_rank() == source
        wire = self.local_result(
            lambda: pack(value, self.device) if is_source else None
        )
        size = torch.tensor(
            [wire.numel() if is_source else 0],
            dtype=torch.int64,
            device=self.device,
        )
        self.broadcast_tensor(size, source)
        wire = self.local_result(
            lambda: (
                wire
                if is_source
                else torch.empty(
                    int(size.item()), dtype=torch.uint8, device=self.device
                )
            )
        )
        self.broadcast_tensor(wire, source)
        return unpack(wire)

    def scatter(self, values: list[Any] | None, source: int) -> Any:
        self.check()
        is_source = dist.get_rank() == source

        def prepare() -> list[Tensor]:
            if not is_source:
                return []
            if values is None or len(values) != self.world:
                raise ValueError("Scatter requires one payload per group rank")
            return [pack(v, self.device) for v in values]

        wires = self.local_result(prepare)
        source_local = dist.get_group_rank(self.group, source)
        sizes = torch.tensor(
            [w.numel() for w in wires] if is_source else [0] * self.world,
            dtype=torch.int64,
            device=self.device,
        )
        received_size = torch.empty_like(sizes)
        self.exchange(received_size, sizes, [1] * self.world, [1] * self.world)
        receives = received_size.cpu().tolist()
        sends = sizes.cpu().tolist()

        def allocate() -> tuple[Tensor, Tensor]:
            send = (
                torch.cat(wires)
                if wires
                else torch.empty(0, dtype=torch.uint8, device=self.device)
            )
            receive = torch.empty(
                sum(receives), dtype=torch.uint8, device=self.device
            )
            return send, receive

        send, receive = self.local_result(allocate)
        self.exchange(receive, send, receives, sends)
        assert all(
            length == 0
            for rank, length in enumerate(receives)
            if rank != source_local
        )
        return unpack(receive)

    def exchange(
        self,
        receive: Tensor,
        send: Tensor,
        receives: list[int],
        sends: list[int],
    ) -> None:
        self.check()
        state: dict[str, Any] = {"send": send, "receive": receive}
        try:
            state["work"] = dist.all_to_all_single(
                receive, send, receives, sends, group=self.group, async_op=True
            )
            state["work"].wait()
            if send.is_cuda:
                event = torch.cuda.Event()
                state["event"] = event
                event.record(torch.cuda.current_stream(send.device))
                event.synchronize()
        except BaseException:
            self.poisoned = True
            _FAILED_TRANSFERS.append(state)
            raise
