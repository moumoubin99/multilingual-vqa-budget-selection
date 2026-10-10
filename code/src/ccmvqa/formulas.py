"""Numerical implementations bound to the design's formulas.

All array helpers are deterministic and side-effect free so that deployment
contract tests can run on CPU before any model process starts.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np


def balanced_independence_target(
    budget: float,
    balanced_language_weight: np.ndarray,
    balanced_concept_weight: np.ndarray,
) -> np.ndarray:
    """MATH_SPEC eq1: fixed-budget balanced independence target."""
    language = np.asarray(balanced_language_weight, dtype=np.float64)
    concept = np.asarray(balanced_concept_weight, dtype=np.float64)
    if language.ndim != 1 or concept.ndim != 1:
        raise ValueError("language and concept weights must be one-dimensional")
    if budget <= 0 or np.any(language < 0) or np.any(concept < 0):
        raise ValueError("budget and weights must be nonnegative, with positive budget")
    if not np.isclose(language.sum(), 1.0) or not np.isclose(concept.sum(), 1.0):
        raise ValueError("language and concept weights must each sum to one")
    return float(budget) * np.outer(language, concept)


def capacity_aware_joint_allocation(
    cell_capacity: np.ndarray,
    row_targets: np.ndarray,
    column_targets: np.ndarray,
    *,
    tolerance: float = 1e-7,
    max_nfev: int = 20_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MATH_SPEC eq2: fit min(K_lc, exp(a_l+b_c)) to fixed marginals."""
    from scipy.optimize import least_squares

    capacity = np.asarray(cell_capacity, dtype=np.float64)
    rows = np.asarray(row_targets, dtype=np.float64)
    columns = np.asarray(column_targets, dtype=np.float64)
    if capacity.ndim != 2 or capacity.shape != (rows.size, columns.size):
        raise ValueError("capacity shape must match row and column marginals")
    if np.any(capacity < 0) or np.any(rows < 0) or np.any(columns < 0):
        raise ValueError("capacities and marginals must be nonnegative")
    if not np.isclose(rows.sum(), columns.sum(), atol=tolerance):
        raise ValueError("row and column totals differ")
    if np.any(capacity.sum(axis=1) + tolerance < rows) or np.any(
        capacity.sum(axis=0) + tolerance < columns
    ):
        raise ValueError("requested marginals exceed cell capacities")

    n_rows, n_columns = capacity.shape

    def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return theta[:n_rows], np.r_[theta[n_rows:], 0.0]

    def table(theta: np.ndarray) -> np.ndarray:
        a, b = unpack(theta)
        return np.minimum(capacity, np.exp(np.clip(a[:, None] + b[None, :], -60, 60)))

    def residual(theta: np.ndarray) -> np.ndarray:
        allocation = table(theta)
        # The final column residual is redundant once total mass is fixed.
        return np.r_[allocation.sum(axis=1) - rows, allocation.sum(axis=0)[:-1] - columns[:-1]]

    scale = max(float(rows.sum()) / max(1, capacity.size), 1e-9)
    initial = np.r_[np.full(n_rows, math.log(scale)), np.zeros(max(0, n_columns - 1))]
    result = least_squares(residual, initial, max_nfev=max_nfev, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    allocation = table(result.x)
    if not result.success or np.max(np.abs(residual(result.x))) > tolerance:
        raise RuntimeError("capacity-aware marginal fit did not converge to the registered marginals")
    language_dual, concept_dual = unpack(result.x)
    return allocation, language_dual, concept_dual


def _integral_transport_round(values: np.ndarray, row_totals: np.ndarray, column_totals: np.ndarray) -> np.ndarray:
    """Minimum-cost floor/ceiling rounding with exact integer marginals.

    The incremental cost for rounding fractional value ``x`` up instead of
    down is ``|x-ceil(x)|-|x-floor(x)| = 1-2*frac(x)``.  Successive shortest
    augmenting paths therefore implement the registered min-cost flow.  Edge
    insertion and strict-distance updates give canonical row-major tie breaks.
    """
    values = np.asarray(values, dtype=np.float64)
    row_totals = np.asarray(row_totals, dtype=np.int64)
    column_totals = np.asarray(column_totals, dtype=np.int64)
    floor = np.floor(values + 1e-10).astype(np.int64)
    row_need = row_totals - floor.sum(axis=1)
    column_need = column_totals - floor.sum(axis=0)
    if np.any(row_need < 0) or np.any(column_need < 0) or row_need.sum() != column_need.sum():
        raise ValueError("fractional table is incompatible with requested integer marginals")

    n_rows, n_columns = values.shape
    source = 0
    row_offset = 1
    column_offset = row_offset + n_rows
    sink = column_offset + n_columns
    graph: list[list[list[float | int]]] = [[] for _ in range(sink + 1)]

    def add_edge(left: int, right: int, capacity: int, cost: float = 0.0) -> None:
        graph[left].append([right, capacity, len(graph[right]), cost])
        graph[right].append([left, 0, len(graph[left]) - 1, -cost])

    for i, need in enumerate(row_need):
        add_edge(source, row_offset + i, int(need))
    fractional = values - floor
    for i in range(n_rows):
        for j in range(n_columns):
            if fractional[i, j] > 1e-9:
                add_edge(row_offset + i, column_offset + j, 1, 1.0 - 2.0 * float(fractional[i, j]))
    for j, need in enumerate(column_need):
        add_edge(column_offset + j, sink, int(need))

    flow = 0
    target = int(row_need.sum())
    while flow < target:
        parent: list[tuple[int, int] | None] = [None] * len(graph)
        distance = [math.inf] * len(graph)
        distance[source] = 0.0
        in_queue = [False] * len(graph)
        queue = deque([source])
        in_queue[source] = True
        while queue:
            node = queue.popleft()
            in_queue[node] = False
            for edge_index, edge in enumerate(graph[node]):
                destination, capacity, _, cost = edge
                candidate = distance[node] + float(cost)
                if int(capacity) > 0 and candidate < distance[int(destination)] - 1e-12:
                    distance[int(destination)] = candidate
                    parent[int(destination)] = (node, edge_index)
                    if not in_queue[int(destination)]:
                        queue.append(int(destination))
                        in_queue[int(destination)] = True
        if parent[sink] is None:
            raise RuntimeError("marginal-preserving transportation rounding is infeasible")
        node = sink
        while node != source:
            previous, edge_index = parent[node]  # type: ignore[misc]
            edge = graph[previous][edge_index]
            edge[1] = int(edge[1]) - 1
            graph[node][int(edge[2])][1] = int(graph[node][int(edge[2])][1]) + 1
            node = previous
        flow += 1

    rounded = floor.copy()
    for i in range(n_rows):
        for edge in graph[row_offset + i]:
            target_node = edge[0]
            if column_offset <= target_node < sink:
                reverse = graph[int(target_node)][int(edge[2])]
                if reverse[1] == 1:
                    rounded[i, int(target_node) - column_offset] += 1
    if not np.array_equal(rounded.sum(axis=1), row_totals) or not np.array_equal(
        rounded.sum(axis=0), column_totals
    ):
        raise AssertionError("transport rounding failed to preserve marginals")
    return rounded


def occupancy_allocation(
    joint_allocation: np.ndarray,
    vertex_allocation: np.ndarray,
    occupancy_lambda: float,
) -> np.ndarray:
    """MATH_SPEC eq3 with marginal-preserving transportation rounding."""
    joint = np.asarray(joint_allocation, dtype=np.float64)
    vertex = np.asarray(vertex_allocation, dtype=np.float64)
    if joint.shape != vertex.shape or joint.ndim != 2:
        raise ValueError("joint and vertex allocations must be equal-shape matrices")
    if occupancy_lambda not in (0.2, 0.8):
        raise ValueError("registered occupancy lambda must be 0.2 or 0.8")
    row_totals = np.rint(joint.sum(axis=1)).astype(np.int64)
    column_totals = np.rint(joint.sum(axis=0)).astype(np.int64)
    if not np.allclose(vertex.sum(axis=1), row_totals) or not np.allclose(
        vertex.sum(axis=0), column_totals
    ):
        raise ValueError("joint and vertex tables must share integer marginals")
    mixed = occupancy_lambda * joint + (1.0 - occupancy_lambda) * vertex
    return _integral_transport_round(mixed, row_totals, column_totals)


def selective_risk(
    correctness: np.ndarray,
    confidence: np.ndarray,
    item_mask: np.ndarray,
    item_id: np.ndarray,
    target_coverage: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MATH_SPEC eq4: ragged per-language top-k risk with frozen-id ties."""
    correctness = np.asarray(correctness, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    mask = np.asarray(item_mask, dtype=bool)
    ids = np.asarray(item_id)
    if correctness.shape != confidence.shape or correctness.shape != mask.shape:
        raise ValueError("correctness, confidence, and item_mask must have equal shape")
    if correctness.ndim != 2 or ids.shape != (correctness.shape[1],):
        raise ValueError("item ids must index the evaluation-item axis")
    if not 0 < target_coverage <= 1:
        raise ValueError("target coverage must be in (0, 1]")
    risks = np.zeros(correctness.shape[0], dtype=np.float64)
    accepted = np.zeros_like(mask)
    thresholds = np.full(correctness.shape[0], np.nan, dtype=np.float64)
    for language in range(correctness.shape[0]):
        candidates = np.flatnonzero(mask[language])
        k = int(math.floor(target_coverage * candidates.size))
        if k == 0:
            continue
        order = np.lexsort((ids[candidates], -confidence[language, candidates]))
        chosen = candidates[order[:k]]
        accepted[language, chosen] = True
        thresholds[language] = confidence[language, chosen[-1]]
        risks[language] = np.mean(1.0 - correctness[language, chosen])
    return risks, accepted, thresholds


def _trimmed_language_mean(values: np.ndarray, selector: np.ndarray) -> float:
    selected = np.asarray(values, dtype=np.float64)[np.asarray(selector, dtype=bool)]
    if selected.size == 0:
        raise ValueError("low-resource selector is empty")
    selected = np.sort(selected)
    trim = math.ceil(selected.size / 4) if selected.size >= 8 else 0
    retained = selected[trim : selected.size - trim]
    if retained.size == 0:
        raise ValueError("registered language trim removed every language")
    return float(np.mean(retained))


def occupancy_risk_gain(
    risk_lowocc: np.ndarray,
    risk_highocc: np.ndarray,
    lowres_selector: np.ndarray,
) -> tuple[np.ndarray, float]:
    """MATH_SPEC eq5: per-unit LowOcc−HighOcc TLM and its median."""
    low = np.asarray(risk_lowocc, dtype=np.float64)
    high = np.asarray(risk_highocc, dtype=np.float64)
    if low.shape != high.shape or low.ndim != 2:
        raise ValueError("occupancy risks must be equal-shape [unit, language] arrays")
    per_unit = np.asarray([_trimmed_language_mean(row, lowres_selector) for row in low - high])
    return per_unit, float(np.median(per_unit))


def joint_balance_interaction(
    risk_joint: np.ndarray,
    risk_language_only: np.ndarray,
    risk_concept_only: np.ndarray,
    risk_natnat: np.ndarray,
    lowres_selector: np.ndarray,
) -> tuple[np.ndarray, float]:
    """MATH_SPEC eq6: per-seed two-factor interaction and its median."""
    arrays = [np.asarray(x, dtype=np.float64) for x in (risk_joint, risk_language_only, risk_concept_only, risk_natnat)]
    if len({x.shape for x in arrays}) != 1 or arrays[0].ndim != 2:
        raise ValueError("all interaction risks must share [seed, language] shape")
    contrast = arrays[0] - arrays[1] - arrays[2] + arrays[3]
    per_seed = np.asarray([_trimmed_language_mean(row, lowres_selector) for row in contrast])
    return per_seed, float(np.median(per_seed))


def causal_lm_cross_entropy(logits, labels, ignore_index: int = -100):
    """Token-averaged causal LM negative log-likelihood used by training."""
    import torch
    import torch.nn.functional as functional

    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("logits [batch, token, vocab] and labels [batch, token] are required")
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    valid = shifted_labels.ne(ignore_index)
    safe_labels = shifted_labels.masked_fill(~valid, 0)
    log_probabilities = functional.log_softmax(shifted_logits.float(), dim=-1)
    target_log_probability = log_probabilities.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return causal_lm_negative_log_likelihood(target_log_probability, valid)


def causal_lm_negative_log_likelihood(target_log_probability, valid_mask):
    """Mean negative target-token log probability over non-masked tokens."""
    valid = valid_mask.to(dtype=target_log_probability.dtype)
    denominator = valid.sum()
    if denominator.item() <= 0:
        raise ValueError("causal LM loss requires at least one valid shifted target token")
    return -(target_log_probability * valid).sum() / denominator


def lora_scaling(alpha: int, rank: int) -> float:
    """Registered LoRA multiplier alpha/r used by PEFT's standard adapter."""
    if alpha <= 0 or rank <= 0:
        raise ValueError("LoRA alpha and rank must be positive")
    return float(alpha) / float(rank)


def lora_linear_output(base_output, inputs, lora_a, lora_b, *, alpha: int, rank: int, dropout_mask=None, dropout_p: float = 0.0):
    """Reference training LoRA transform W x + (alpha/r) B A D_p(x)."""
    if not 0 <= dropout_p < 1:
        raise ValueError("LoRA dropout must be in [0, 1)")
    if dropout_mask is not None:
        inputs = inputs * dropout_mask.to(dtype=inputs.dtype) / (1.0 - dropout_p)
    elif dropout_p:
        raise ValueError("a deterministic dropout mask is required by the reference path")
    return base_output + lora_scaling(alpha, rank) * ((inputs @ lora_a.T) @ lora_b.T)


def accumulated_loss(raw_loss, accumulation_steps: int):
    """Scale each micro-batch loss for effective-batch gradient accumulation."""
    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be positive")
    return raw_loss / accumulation_steps


def optimizer_step_due(micro_step: int, accumulation_steps: int, *, final_micro_step: bool) -> bool:
    """Step at a full accumulation boundary or flush the final partial batch."""
    if micro_step <= 0 or accumulation_steps <= 0:
        raise ValueError("micro_step and accumulation_steps must be positive")
    return micro_step % accumulation_steps == 0 or final_micro_step


def clip_gradient_global_norm(parameters, max_norm: float):
    """Apply g <- g*min(1, max_norm/||g||_2) through torch's exact runtime path."""
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    import torch

    return torch.nn.utils.clip_grad_norm_(parameters, max_norm, foreach=False)


def build_adamw(parameters, *, learning_rate: float):
    """Construct the exact AdamW path with all defaults frozen explicitly."""
    import torch

    return torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        amsgrad=False,
        maximize=False,
        foreach=False,
        capturable=False,
        differentiable=False,
        fused=None,
    )


def build_cosine_warmup_scheduler(optimizer, *, warmup_steps: int, training_steps: int):
    """Construct the exact Transformers cosine-with-warmup schedule used in training."""
    if warmup_steps < 0 or training_steps <= 0 or warmup_steps > training_steps:
        raise ValueError("invalid warmup/training step counts")
    from transformers import get_cosine_schedule_with_warmup

    return get_cosine_schedule_with_warmup(optimizer, warmup_steps, training_steps)


def scaled_dot_product_attention_reference(query, key, value, attention_mask=None):
    """Reference non-causal scaled dot-product attention for SDPA conformance."""
    import torch

    scores = query @ key.transpose(-2, -1) / math.sqrt(query.shape[-1])
    if attention_mask is not None:
        scores = scores + attention_mask
    return torch.softmax(scores, dim=-1) @ value


def language_model_logits_reference(hidden_state, lm_head_weight):
    """Reference bias-free Qwen language-model projection."""
    return hidden_state @ lm_head_weight.T


def qwen_forward_logits(model, batch):
    """Captured launch wrapper for Qwen2.5-VL with its configured SDPA path."""
    return model(**batch, use_cache=False).logits


def adamw_reference_update(
    parameter,
    gradient,
    first_moment,
    second_moment,
    *,
    step: int,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    weight_decay: float = 0.01,
):
    """Independent scalar/tensor AdamW recurrence for the pinned single-tensor branch."""
    if step <= 0:
        raise ValueError("step must be positive")
    next_first = beta1 * first_moment + (1.0 - beta1) * gradient
    next_second = beta2 * second_moment + (1.0 - beta2) * gradient.square()
    first_hat = next_first / (1.0 - beta1**step)
    second_hat = next_second / (1.0 - beta2**step)
    decayed = parameter * (1.0 - learning_rate * weight_decay)
    updated = decayed - learning_rate * first_hat / (second_hat.sqrt() + epsilon)
    return updated, next_first, next_second


def rms_norm_reference(hidden_state, weight, epsilon: float):
    """Qwen RMSNorm reference with float32 variance accumulation."""
    import torch

    original_dtype = hidden_state.dtype
    value = hidden_state.to(torch.float32)
    normalized = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + epsilon)
    return weight * normalized.to(original_dtype)


def swiglu_reference(
    hidden_state,
    gate_weight,
    up_weight,
    down_weight,
    gate_bias=None,
    up_bias=None,
    down_bias=None,
):
    """Qwen SwiGLU block reference, including the vision path's affine biases."""
    import torch.nn.functional as functional

    gate = functional.linear(hidden_state, gate_weight, gate_bias)
    up = functional.linear(hidden_state, up_weight, up_bias)
    return functional.linear(functional.silu(gate) * up, down_weight, down_bias)


def residual_add_reference(residual, branch_output):
    """Transformer residual connection reference."""
    return residual + branch_output


def rotate_half_reference(value):
    """Rotate the two last-dimension halves as used by Qwen RoPE."""
    import torch

    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def rotary_reference(value, cosine, sine):
    """Apply a precomputed rotary embedding with broadcast-compatible tensors."""
    return value * cosine + rotate_half_reference(value) * sine


def vision_patch_projection_reference(patches, convolution_weight):
    """Reference for non-overlapping Qwen vision patch Conv3d projection."""
    flattened = patches.reshape(patches.shape[0], -1)
    return flattened @ convolution_weight.reshape(convolution_weight.shape[0], -1).T


def vision_patch_merger_reference(
    hidden_state,
    norm_weight,
    epsilon,
    first_weight,
    first_bias,
    second_weight,
    second_bias,
    spatial_merge_size,
):
    """Reference Qwen patch merger: RMSNorm, flatten spatial group, GELU MLP."""
    import torch
    import torch.nn.functional as functional

    normalized = rms_norm_reference(hidden_state, norm_weight, epsilon)
    merged_width = hidden_state.shape[-1] * spatial_merge_size**2
    grouped = normalized.reshape(-1, merged_width)
    projected = functional.linear(grouped, first_weight, first_bias)
    return functional.linear(exact_gelu_reference(projected), second_weight, second_bias)


def exact_gelu_reference(value):
    """Exact GELU used by Qwen's patch merger (approximate='none')."""
    import torch

    return 0.5 * value * (1.0 + torch.erf(value / math.sqrt(2.0)))


def embedding_lookup_reference(input_ids, embedding_weight):
    """Reference token embedding lookup."""
    return embedding_weight[input_ids]


def linear_projection_reference(hidden_state, weight, bias=None):
    """Independent affine projection used for Q/K/V/O and vision QKV/proj checks."""
    return hidden_state @ weight.T + (0.0 if bias is None else bias)


def split_attention_heads_reference(projected, number_of_heads: int):
    """Map [batch, token, hidden] projections to [batch, head, token, head_dim]."""
    batch, tokens, width = projected.shape
    if width % number_of_heads:
        raise ValueError("projection width must divide number_of_heads")
    return projected.reshape(batch, tokens, number_of_heads, width // number_of_heads).transpose(1, 2)


def repeat_key_value_reference(hidden_state, groups: int):
    """Repeat grouped key/value heads to the query-head count."""
    if groups <= 0:
        raise ValueError("groups must be positive")
    return hidden_state.repeat_interleave(groups, dim=1)


def multimodal_rope_reference(query, key, cosine, sine, sections, unsqueeze_dim=1):
    """Independent Qwen temporal/height/width RoPE section routing."""
    import torch

    # Transformers >=5.17 performs the temporal/height/width section routing in
    # Qwen2_5_VLRotaryEmbedding.recomposition_frequencies and hands attention a
    # [batch, token, head_dim] tensor.  Older releases hand over the three grids
    # separately as [3, batch, token, head_dim] and route them here.
    if cosine.ndim == query.ndim - 1:
        routed_cosine = cosine.unsqueeze(unsqueeze_dim)
        routed_sine = sine.unsqueeze(unsqueeze_dim)
        return rotary_reference(query, routed_cosine, routed_sine), rotary_reference(
            key, routed_cosine, routed_sine
        )
    doubled = list(sections) * 2
    routed_cosine = torch.cat(
        [chunk[index % 3] for index, chunk in enumerate(cosine.split(doubled, dim=-1))], dim=-1
    ).unsqueeze(unsqueeze_dim)
    routed_sine = torch.cat(
        [chunk[index % 3] for index, chunk in enumerate(sine.split(doubled, dim=-1))], dim=-1
    ).unsqueeze(unsqueeze_dim)
    return rotary_reference(query, routed_cosine, routed_sine), rotary_reference(
        key, routed_cosine, routed_sine
    )


def causal_padding_mask_reference(attention_mask, query_length: int, dtype):
    """Create the additive causal-plus-padding mask used by SDPA training."""
    import torch

    batch, key_length = attention_mask.shape
    negative_infinity = torch.finfo(dtype).min
    causal = torch.full((query_length, key_length), negative_infinity, dtype=dtype)
    causal = torch.triu(causal, diagonal=1 + key_length - query_length)
    causal = causal.view(1, 1, query_length, key_length).expand(batch, 1, -1, -1).clone()
    padding = attention_mask[:, None, None, :].eq(0)
    return causal.masked_fill(padding, negative_infinity)


def image_placeholder_insert_reference(inputs_embeds, input_ids, image_token_id: int, image_features):
    """Replace image-token embedding slots with the ordered visual features."""
    mask = input_ids.eq(image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
    if int(mask.sum().item()) != image_features.numel():
        raise ValueError("image feature count does not match placeholder slots")
    return inputs_embeds.masked_scatter(mask, image_features.to(inputs_embeds.dtype).reshape(-1))


def one_image_mrope_positions_reference(
    prefix_length, grid_t, grid_h, grid_w, spatial_merge_size, suffix_length
):
    """Independent 3D position assignment for one image followed by text."""
    import torch

    llm_h, llm_w = grid_h // spatial_merge_size, grid_w // spatial_merge_size
    prefix = torch.arange(prefix_length).view(1, -1).expand(3, -1)
    t_index = torch.arange(grid_t).view(-1, 1).expand(-1, llm_h * llm_w).flatten()
    h_index = torch.arange(llm_h).view(1, -1, 1).expand(grid_t, -1, llm_w).flatten()
    w_index = torch.arange(llm_w).view(1, 1, -1).expand(grid_t, llm_h, -1).flatten()
    vision = torch.stack((t_index, h_index, w_index)) + prefix_length
    suffix_start = int(vision.max().item()) + 1
    suffix = torch.arange(suffix_length).view(1, -1).expand(3, -1) + suffix_start
    return torch.cat((prefix, vision, suffix), dim=1)


def vision_window_index_reference(
    grid_thw, *, window_size: int, spatial_merge_size: int, patch_size: int
):
    """Independent production Qwen vision-window partition and cumulative lengths."""
    import torch
    import torch.nn.functional as functional

    merger_window = window_size // spatial_merge_size // patch_size
    indices, cumulative, offset = [], [0], 0
    merge_unit = spatial_merge_size**2
    for grid_t, grid_h, grid_w in grid_thw.tolist():
        llm_h, llm_w = grid_h // spatial_merge_size, grid_w // spatial_merge_size
        index = torch.arange(grid_t * llm_h * llm_w).reshape(grid_t, llm_h, llm_w)
        pad_h = merger_window - llm_h % merger_window
        pad_w = merger_window - llm_w % merger_window
        windows_h = (llm_h + pad_h) // merger_window
        windows_w = (llm_w + pad_w) // merger_window
        padded = functional.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
        padded = padded.reshape(grid_t, windows_h, merger_window, windows_w, merger_window)
        padded = padded.permute(0, 1, 3, 2, 4).reshape(
            grid_t, windows_h * windows_w, merger_window, merger_window
        )
        lengths = padded.ne(-100).sum((2, 3)).reshape(-1)
        flattened = padded.reshape(-1)
        indices.append(flattened[flattened.ne(-100)] + offset)
        cumulative.extend((lengths.cumsum(0) * merge_unit + cumulative[-1]).tolist())
        offset += grid_t * llm_h * llm_w
    return torch.cat(indices), cumulative


def fast_image_patchify_reference(images, *, temporal_patch_size: int, patch_size: int, merge_size: int):
    """Reference Qwen fast-processor patch ordering after resize/rescale/normalize."""
    import torch

    if images.ndim == 4:
        images = images.unsqueeze(1)
    if images.shape[1] % temporal_patch_size:
        repeats = images[:, -1:].repeat(1, temporal_patch_size - 1, 1, 1, 1)
        images = torch.cat((images, repeats), dim=1)
    batch, frames, channels, height, width = images.shape
    grid_t, grid_h, grid_w = frames // temporal_patch_size, height // patch_size, width // patch_size
    patches = images.view(
        batch,
        grid_t,
        temporal_patch_size,
        channels,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    return patches.reshape(
        batch, grid_t * grid_h * grid_w, channels * temporal_patch_size * patch_size * patch_size
    ), (grid_t, grid_h, grid_w)


def vision_rotary_grid_reference(grid_thw, *, spatial_merge_size: int, rotary_module):
    """Reference Qwen vision height/width grid ordering and frequency lookup."""
    import torch

    position_ids = []
    for temporal, height, width in grid_thw.tolist():
        height_ids = torch.arange(height).unsqueeze(1).expand(-1, width)
        height_ids = height_ids.reshape(
            height // spatial_merge_size, spatial_merge_size, width // spatial_merge_size, spatial_merge_size
        ).permute(0, 2, 1, 3).flatten()
        width_ids = torch.arange(width).unsqueeze(0).expand(height, -1)
        width_ids = width_ids.reshape(
            height // spatial_merge_size, spatial_merge_size, width // spatial_merge_size, spatial_merge_size
        ).permute(0, 2, 1, 3).flatten()
        position_ids.append(torch.stack((height_ids, width_ids), dim=-1).repeat(temporal, 1))
    position_ids = torch.cat(position_ids, dim=0)
    full = rotary_module(grid_thw[:, 1:].max())
    return full[position_ids].flatten(1)


def vision_position_ids_reference(grid_thw, *, spatial_merge_size: int):
    """Independent height/width ids used by the shared vision-utils path."""
    import torch

    position_ids = []
    for temporal, height, width in grid_thw.tolist():
        height_ids = torch.arange(height).unsqueeze(1).expand(-1, width)
        height_ids = height_ids.reshape(
            height // spatial_merge_size, spatial_merge_size, width // spatial_merge_size, spatial_merge_size
        ).permute(0, 2, 1, 3).flatten()
        width_ids = torch.arange(width).unsqueeze(0).expand(height, -1)
        width_ids = width_ids.reshape(
            height // spatial_merge_size, spatial_merge_size, width // spatial_merge_size, spatial_merge_size
        ).permute(0, 2, 1, 3).flatten()
        position_ids.append(torch.stack((height_ids, width_ids), dim=-1).repeat(temporal, 1))
    return torch.cat(position_ids, dim=0)


def vision_axial_rotary_reference(position_ids, inverse_frequency, attention_scaling: float = 1.0):
    """Independent axial vision RoPE cos/sin construction for current Qwen."""
    import torch

    frequency = position_ids[..., None].float() * inverse_frequency.float()
    cosine = frequency.cos() * attention_scaling
    sine = frequency.sin() * attention_scaling
    cosine_hw = torch.cat((cosine[:, 0], cosine[:, 1]), dim=-1)
    sine_hw = torch.cat((sine[:, 0], sine[:, 1]), dim=-1)
    return torch.cat((cosine_hw, cosine_hw), dim=-1), torch.cat((sine_hw, sine_hw), dim=-1)


def rope_inverse_frequency_reference(head_dimension: int, theta: float, *, device=None):
    """Default Qwen RoPE initializer used by the production mRoPE wrapper."""
    import torch

    return 1.0 / (
        theta ** (torch.arange(0, head_dimension, 2, dtype=torch.float32, device=device) / head_dimension)
    )


def max_pixel_geometry_reference(width: int, height: int, max_pixels: int):
    """Geometry of the launch's first PIL resize stage."""
    if width <= 0 or height <= 0 or max_pixels <= 0:
        raise ValueError("image geometry and max_pixels must be positive")
    if width * height <= max_pixels:
        return width, height
    scale = math.sqrt(max_pixels / (width * height))
    return max(1, int(width * scale)), max(1, int(height * scale))


def smart_resize_geometry_reference(
    height: int, width: int, *, factor: int, min_pixels: int, max_pixels: int
):
    """Independent integer geometry for Qwen fast processor smart_resize."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute aspect ratio must be smaller than 200")
    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor
    if resized_height * resized_width > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_height = max(factor, math.floor(height / beta / factor) * factor)
        resized_width = max(factor, math.floor(width / beta / factor) * factor)
    elif resized_height * resized_width < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_height = math.ceil(height * beta / factor) * factor
        resized_width = math.ceil(width * beta / factor) * factor
    return resized_height, resized_width


def inverted_dropout_reference(inputs, mask, dropout_probability: float):
    """Seeded Bernoulli inverted-dropout transformation."""
    if not 0 <= dropout_probability < 1:
        raise ValueError("dropout probability must be in [0, 1)")
    return inputs * mask.to(inputs.dtype) / (1.0 - dropout_probability)


def lora_a_initialization_bound(input_dimension: int) -> float:
    """PEFT's default Kaiming-uniform LoRA-A bound for a=sqrt(5); LoRA-B starts at zero."""
    if input_dimension <= 0:
        raise ValueError("input_dimension must be positive")
    return 1.0 / math.sqrt(input_dimension)


def seeded_epoch_permutation(size: int, seed: int, epoch: int) -> list[int]:
    """Deterministic MT19937 epoch permutation used by the registered trainer."""
    import random

    if size <= 0 or seed < 0 or epoch < 0:
        raise ValueError("size must be positive and seed/epoch nonnegative")
    order = list(range(size))
    random.Random(seed + epoch).shuffle(order)
    return order


def lora_gradient_reference(dropped_inputs, lora_a, lora_b, output_gradient, *, scaling: float):
    """Chain-rule gradients for Y = scaling * (D_p(X) A^T) B^T."""
    projected = dropped_inputs @ lora_a.T
    gradient_b = scaling * output_gradient.T @ projected
    gradient_a = scaling * (output_gradient @ lora_b).T @ dropped_inputs
    return gradient_a, gradient_b
