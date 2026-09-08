"""Greedy-only compute sharding for an unquantized VanillaMarkov head.

The full checkpoint parameter stays available to the stochastic/eager paths.
Only the greedy graph projects a local W2 slice and exchanges top-1 pairs.
"""

import torch
import torch.nn.functional as F


class VanillaMarkovGreedyShard:
    def __init__(self, head, group, *, start, end, partition_width):
        self.head = head
        self.group = group
        self.start = start
        self.end = end
        self.partition_width = partition_width

    @classmethod
    def create(cls, head, lm_head, group):
        vocab = head.vocab_size
        indices = lm_head.shard_indices
        width = int(lm_head.num_embeddings_per_partition)
        rank = int(group.rank_in_group)
        tp = int(group.world_size)
        start = int(indices.org_vocab_start_index)
        end = int(indices.org_vocab_end_index)
        # Require the original vocabulary to occupy consecutive rank slices.
        # Added-vocabulary layouts and empty ranks retain the existing path.
        if (
            int(lm_head.org_vocab_size) != vocab
            or int(getattr(lm_head, "num_embeddings", vocab)) != vocab
            or vocab >= 2**24  # Candidate ids are represented exactly in FP32.
            or vocab <= (tp - 1) * width  # All ranks must select the same path.
            or int(lm_head.tp_size) != tp
            or int(lm_head.num_embeddings_padded) != width * tp
            or (start, end) != (rank * width, min((rank + 1) * width, vocab))
            or not 0 <= start < end <= vocab
            or head.markov_w2.weight.dtype != torch.bfloat16
            or head.markov_w1.weight.dtype != torch.bfloat16
            or lm_head.weight.dtype != torch.bfloat16
            or tuple(head.markov_w2.weight.shape) != (vocab, head.markov_rank)
            or getattr(head.markov_w2, "bias", None) is not None
        ):
            return None
        return cls(head, group, start=start, end=end, partition_width=width)

    def sample(self, base_local, *, first_prev_tokens, sync):
        from sglang.kernels.ops.speculative.dspark.dspark_greedy_top1_npu import (
            select_vanilla_global_top1_npu,
            select_vanilla_local_top1_npu,
        )

        bs, gamma, width = base_local.shape
        if width != self.partition_width or base_local.dtype != torch.bfloat16:
            raise ValueError("greedy logits do not match the configured LM head shard")
        if gamma == 0:
            return torch.empty((bs, 0), dtype=torch.long, device=base_local.device)
        weight = self.head.markov_w2.weight[self.start : self.end]
        tokens = []
        previous = first_prev_tokens.long()
        for step in range(gamma):
            bias = F.linear(self.head.get_prev_embeddings(previous), weight)
            local = select_vanilla_local_top1_npu(
                base_local[:, step, : self.end - self.start],
                bias,
                vocab_offset=self.start,
            )
            candidates = (
                self.group.all_gather(local, dim=1)
                if self.group.world_size > 1
                else local
            )
            previous = select_vanilla_global_top1_npu(
                candidates.view(bs, self.group.world_size, 2),
                vocab_size=self.head.vocab_size,
            )
            # Retain the caller's configured per-step speculative sync site.
            previous = sync(previous)
            tokens.append(previous)
        return torch.stack(tokens, dim=1)
