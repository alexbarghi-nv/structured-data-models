"""Stable merge of completed normalized attention summaries.

Experimental helper; not wired into the model. All ranks must provide matching
shapes, using (-inf LSE, zero output) for an empty local shard.
"""
import torch
import torch.distributed as dist
from torch import Tensor


def merge_summary(out: Tensor, lse: Tensor, group=None) -> tuple[Tensor, Tensor]:
    maximum = lse.float().clone(memory_format=torch.contiguous_format)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
    safe_max = maximum.masked_fill(maximum.isneginf(), 0)
    weight = (lse.float() - safe_max).exp().contiguous()
    numerator = (torch.where(weight.unsqueeze(-1) > 0, out.float(), 0) * weight.unsqueeze(-1)).contiguous()
    dist.all_reduce(weight, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(numerator, op=dist.ReduceOp.SUM, group=group)
    valid = weight > 0
    result = numerator / weight.clamp_min(torch.finfo(weight.dtype).tiny).unsqueeze(-1)
    global_lse = torch.where(valid, safe_max + weight.log(), maximum)
    return result, global_lse


class CudaSummaryTicket:
    """One ordered NCCL merge of a completed local tile, not a model API.

    Construct collectively in identical tile order on all ranks. Producer work
    must already be enqueued on the calling stream. Hold this object until
    ``consume``; no input or output storage may be reused sooner. No global
    device synchronize or CPU assumption about collective completion.
    """

    def __init__(self, out: Tensor, lse: Tensor, group, stream: torch.cuda.Stream):
        self.inputs = (out, lse)
        self.works = []
        self.buffers = []
        self.consumed = False
        if (not out.is_cuda or not lse.is_cuda or out.device != lse.device
                or stream.device != out.device):
            raise ValueError("CUDA tensors on one device required")
        if out.shape[:-1] != lse.shape:
            raise ValueError("Summary shapes must match")
        if dist.get_backend(group) != "nccl":
            raise ValueError("CUDA summary ticket requires NCCL")
        self.stream = stream
        producer = torch.cuda.current_stream(out.device)
        ready = torch.cuda.Event()
        ready.record(producer)
        self.ready = ready
        with torch.cuda.stream(stream):
            stream.wait_event(ready)
            out.record_stream(stream)
            lse.record_stream(stream)
            maximum = lse.float().clone(memory_format=torch.contiguous_format)
            self.buffers.append(maximum)
            work = dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group, async_op=True)
            self.works.append(work)
            work.wait()  # NCCL Work.wait establishes dependency on this stream.
            safe_max = maximum.masked_fill(maximum.isneginf(), 0)
            self.buffers.append(safe_max)
            weight = (lse.float() - safe_max).exp().contiguous()
            self.buffers.append(weight)
            numerator = (torch.where(weight.unsqueeze(-1) > 0, out.float(), 0) * weight.unsqueeze(-1)).contiguous()
            self.buffers.append(numerator)
            for tensor in (weight, numerator):
                work = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group, async_op=True)
                self.works.append(work)
                work.wait()
            result = numerator / weight.clamp_min(torch.finfo(weight.dtype).tiny).unsqueeze(-1)
            global_lse = torch.where(weight > 0, safe_max + weight.log(), maximum)
            self.buffers.extend((result, global_lse))
            for tensor in self.buffers:
                tensor.record_stream(stream)
            self.completed = torch.cuda.Event()
            self.completed.record(stream)
        self.consumed = False

    def consume(self) -> tuple[Tensor, Tensor]:
        if self.consumed:
            raise RuntimeError("Summary ticket already consumed")
        consumer = torch.cuda.current_stream(self.inputs[0].device)
        consumer.wait_event(self.completed)
        result, lse = self.buffers[-2:]
        result.record_stream(consumer)
        lse.record_stream(consumer)
        self.consumed = True
        return result, lse


class OneTicketQueue:
    """One slot, with explicit host completion throttle before reuse.

    Retain a partially constructed ticket on failure. The caller must abort the
    distributed worker; it must not submit recovery collectives or fall back.
    """

    def __init__(self):
        self.ticket = None
        self.poisoned = False

    def submit(self, factory, *args, **kwargs):
        if self.poisoned:
            raise RuntimeError("Summary queue is poisoned")
        if self.ticket is not None:
            if not self.ticket.consumed:
                raise RuntimeError("Consume the outstanding summary first")
            try:
                self.ticket.completed.synchronize()
            except BaseException:
                self.poisoned = True
                raise
            self.ticket = None
        # Assign before initialization so failures retain tensors and Work.
        ticket = factory.__new__(factory)
        self.ticket = ticket
        try:
            factory.__init__(ticket, *args, **kwargs)
        except BaseException:
            self.poisoned = True
            raise

    def drain(self):
        if self.poisoned or self.ticket is None:
            raise RuntimeError("No usable outstanding summary")
        try:
            return self.ticket.consume()
        except BaseException:
            self.poisoned = True
            raise

    def close(self):
        if self.poisoned:
            raise RuntimeError("Failed queue must remain owned until worker exit")
        if self.ticket is not None:
            try:
                self.ticket.completed.synchronize()
            except BaseException:
                self.poisoned = True
                raise
            self.ticket = None
