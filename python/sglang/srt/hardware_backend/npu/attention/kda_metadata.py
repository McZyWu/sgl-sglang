# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch


def mask_dense_verify_cache_indices(
    cache_indices: torch.Tensor, query_start_loc: torch.Tensor
) -> torch.Tensor:
    """Return int64 state indices with zero-length graph requests masked."""
    active_requests = query_start_loc[1:] > query_start_loc[:-1]
    cache_indices_i64 = cache_indices[: active_requests.shape[0]].to(torch.int64)
    return torch.where(active_requests, cache_indices_i64, -1)
