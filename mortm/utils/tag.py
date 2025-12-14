from typing import Dict, Type

import numpy as np

from mortm.train.tokenizer import *
from torch import Tensor
import torch




def extract_tagged_sequences_batch(
        batch_tensor: Type[Tensor | np.ndarray | list],
        start_tag_id: int,
        end_tag_id: int,
        include_tags: bool = True
) -> List[List[Tensor]]:
    """
    Extracts sequences enclosed by start and end tags from a batch of tensors.
    Optimized to minimize GPU-CPU synchronization points.

    Args:
        batch_tensor (torch.Tensor): 2D Tensor of shape (Batch, Sequence).
        start_tag_id (int): The ID representing the start tag.
        end_tag_id (int): The ID representing the end tag.
        include_tags (bool): Whether to include the tags in the extracted sequences.

    Returns:
        List[List[torch.Tensor]]: A list of length 'Batch'. Each element is a list of
                                  extracted 1D tensors found in that batch sample.
    """
    if isinstance(batch_tensor, np.ndarray) or isinstance(batch_tensor, list):
        batch_tensor = torch.tensor(batch_tensor)
    if batch_tensor.dim() == 1:
        batch_tensor = batch_tensor.unsqueeze(0)
    batch_size = batch_tensor.shape[0]
    results: List[List[Tensor]] = [[] for _ in range(batch_size)]

    # 1. Locate all tags across the entire batch at once (GPU operation)
    # Result shape: [N, 2] where each row is [batch_index, sequence_index]
    # as_tuple=False returns a 2D tensor of indices
    start_indices = (batch_tensor == start_tag_id).nonzero(as_tuple=False)
    end_indices = (batch_tensor == end_tag_id).nonzero(as_tuple=False)

    # 2. Transfer indices to CPU for logic processing (Minimize GPU sync)
    # Converting to list of lists: [[b, s], [b, s], ...]
    starts_list = start_indices.tolist()
    ends_list = end_indices.tolist()

    # 3. Organize indices by batch using a dictionary for O(1) access
    # dict structure: { batch_idx: [seq_idx1, seq_idx2, ...], ... }
    starts_by_batch: Dict[int, List[int]] = {b: [] for b in range(batch_size)}
    ends_by_batch: Dict[int, List[int]] = {b: [] for b in range(batch_size)}

    for b_idx, s_idx in starts_list:
        starts_by_batch[b_idx].append(s_idx)

    for b_idx, s_idx in ends_list:
        ends_by_batch[b_idx].append(s_idx)

    # 4. Process each batch sample purely on CPU logic
    for b in range(batch_size):
        s_indices = starts_by_batch[b]
        e_indices = ends_by_batch[b]

        if not s_indices or not e_indices:
            continue

        # Extract sequences using linear scan logic
        last_end_pos = -1
        e_search_start = 0
        num_ends = len(e_indices)

        for s_pos in s_indices:
            # Skip if this start tag is inside an already processed segment
            if s_pos <= last_end_pos:
                continue

            # Find the first end tag that appears after the current start tag
            found_e_pos = -1
            for k in range(e_search_start, num_ends):
                e_pos = e_indices[k]
                if e_pos > s_pos:
                    found_e_pos = e_pos
                    e_search_start = k + 1 # Optimization: start next search from here
                    break

            # If a valid pair is found, slice the original tensor
            if found_e_pos != -1:
                if include_tags:
                    # Slice: [s_pos : e_pos + 1]
                    segment = batch_tensor[b, s_pos : found_e_pos + 1]
                else:
                    # Slice: [s_pos + 1 : e_pos]
                    segment = batch_tensor[b, s_pos + 1 : found_e_pos]

                results[b].append(segment)
                last_end_pos = found_e_pos

    return results

