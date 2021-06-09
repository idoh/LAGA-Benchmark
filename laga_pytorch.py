from dataclasses import dataclass
from typing import List
import torch
import torch.distributed as dist
from torch.optim import Optimizer
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters

@dataclass
class Anchor:
    flat: torch.Tensor
    unflat: List[torch.Tensor]

def init_anchors(params):
    anchor_unflat = params
    anchor_flat = parameters_to_vector(anchor_unflat).zero_()
    vector_to_parameters(anchor_flat, anchor_unflat)

    return Anchor(anchor_flat, anchor_unflat)


class _LAGA(Optimizer):
    def __init__(self, optimizer):
        super(self.__class__, self).__init__(optimizer.param_groups)
        self.params = [p for group in optimizer.param_groups for p in group['params'] if p.requires_grad]
        self.world_size = dist.get_world_size()
        self.init_gradients()
        self.grads = [p.grad for p in self.params]

        self.anchors = [init_anchors([p.clone() for p in self.params]) for _ in range(2)]
        vector_to_parameters(self.anchors[0].flat, self.grads)

        self.futures = []
        self._laga_step_count = 0

    def init_gradients(self):
        for p in self.params:
            p.grad = torch.zeros_like(p)

    def synchrnous_sync(self):
        self.anchors[0].flat.data.div_(self.world_size)
        dist.all_reduce(self.anchors[0].flat)

    def asynchrnous_sync(self):
        grad_anchor = self.anchors[self._laga_step_count % len(self.anchors)]
        buff_anchor = self.anchors[(1 + self._laga_step_count) % len(self.anchors)]

        grad_anchor.flat.data.div_(self.world_size)
        fut_all_reduce = dist.all_reduce(grad_anchor.flat, async_op=True)
        vector_to_parameters(buff_anchor.flat, self.grads)

        for fut in self.futures:
            fut.wait()

        self.futures = [fut_all_reduce]

    def step(self, closure=None):
        with torch.no_grad():
            self.asynchrnous_sync()

            self._laga_step_count += 1
            return super(self.__class__, self).step(closure)

def LAGA(optimizer):
    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override the `step()` method with an push_pull implementation.
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,), dict(_LAGA.__dict__))
    return cls(optimizer)
