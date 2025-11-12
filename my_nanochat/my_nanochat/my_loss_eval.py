import torch
import torch.distributed as dist
import math

@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_bytes = torch.tensor(0.0, dtype=torch.int64, device=model.get_device())
    batch_iter = iter(batches)
    for _ in range(steps):
        x, y = next(batch_iter)
        loss2d = model(x, y, loss_reduction='none') # (B, T) <-- believe wrong, already flat
        loss2d = loss2d.view(-1) # flatten
        y = y.view(-1) # flatten
        if (y.int() < 0).any():
            # he comments:
            # slightly more complex code path if some target tokens are ignore_index (e.g. -1)
            # any target token < 0 is to be ignored: do NOT index token_bytes with negatives
            valid = y >= 0
            y_safe = torch.where(valid, y, torch.zeros_like(y)) # invalids will be 0
            num_bytes2d = torch.where(         # not sure why called num_bytes2d
                valid,
                token_bytes[y_safe],
                torch.zeros_like(y, dtype=token_bytes.dtype))
            total_nats += (loss2d * (num_bytes2d > 0)).sum() # sums loss excluding invalid y
            total_bytes += num_bytes2d.sum()
        else:
            num_bytes2d = token_bytes[y]
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
    # nearly there...now some DDP stuff
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
    # move both to cpu, calculate bpb and return
    total_nats = total_nats.item()
    total_bytes = total_bytes.item()
    if total_bytes == 0:
        return float('inf')
    bpb = total_nats / (math.log(2) * total_bytes)
    return bpb