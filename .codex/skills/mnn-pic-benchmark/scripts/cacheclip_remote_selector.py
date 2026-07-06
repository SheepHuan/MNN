#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _detach_past_key_values(past_key_values: Any) -> Any:
    import torch

    detached = copy.deepcopy(past_key_values)
    return _map_cache_tensors_in_place(
        detached,
        lambda tensor: tensor.detach().cpu().contiguous(),
    )


def _clone_past_key_values(past_key_values: Any, device: Any) -> Any:
    cloned = copy.deepcopy(past_key_values)
    return _map_cache_tensors_in_place(
        cloned,
        lambda tensor: tensor.detach().clone().to(device=device).contiguous(),
    )


def _map_cache_tensors_in_place(value: Any, fn: Any) -> Any:
    try:
        import torch
    except Exception:  # pragma: no cover
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, tuple):
        return tuple(_map_cache_tensors_in_place(item, fn) for item in value)
    if isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = _map_cache_tensors_in_place(item, fn)
        return value
    if isinstance(value, dict):
        for key, item in list(value.items()):
            value[key] = _map_cache_tensors_in_place(item, fn)
        return value
    for attr in ("keys", "values"):
        tensor = getattr(value, attr, None)
        if torch is not None and isinstance(tensor, torch.Tensor):
            setattr(value, attr, fn(tensor))
    layers = getattr(value, "layers", None)
    if isinstance(layers, list):
        for layer in layers:
            _map_cache_tensors_in_place(layer, fn)
    return value


def _load_hf_tokenizer(path: str, local_files_only: bool, trust_remote_code: bool) -> Any:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("transformers is required for CacheClip tokenizer loading") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        local_files_only=bool(local_files_only),
        trust_remote_code=bool(trust_remote_code),
        use_fast=True,
    )
    if not bool(getattr(tokenizer, "is_fast", False)):
        raise RuntimeError("CacheClip requires a fast tokenizer with offset_mapping support: %s" % path)
    return tokenizer


def _torch_dtype(name: str) -> Any:
    import torch

    normalized = str(name).strip().lower()
    if normalized == "float32":
        return torch.float32
    if normalized == "bfloat16":
        return torch.bfloat16
    if normalized == "float16":
        return torch.float16
    raise RuntimeError("Unsupported cacheclip_aux_torch_dtype=%r" % (name,))


def _resolve_aux_device(spec: str) -> Any:
    import torch

    normalized = str(spec).strip().lower() or "cpu"
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "cacheclip_aux_device=%r requested CUDA but torch.cuda.is_available() is false" % (spec,)
        )
    try:
        return torch.device(normalized)
    except RuntimeError as exc:
        raise RuntimeError("Unsupported cacheclip_aux_device=%r" % (spec,)) from exc


def _load_aux_causal_lm(
    path: str,
    dtype_name: str,
    local_files_only: bool,
    trust_remote_code: bool,
) -> Any:
    try:
        from transformers import AutoModelForCausalLM
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("transformers is required for CacheClip auxiliary model loading") from exc

    dtype = _torch_dtype(dtype_name)
    base_kwargs = {
        "local_files_only": bool(local_files_only),
        "trust_remote_code": bool(trust_remote_code),
    }
    attempts = (
        {"attn_implementation": "eager", "dtype": dtype},
        {"attn_implementation": "eager", "torch_dtype": dtype},
        {"dtype": dtype},
        {"torch_dtype": dtype},
    )
    last_type_error = None
    for extra_kwargs in attempts:
        try:
            return AutoModelForCausalLM.from_pretrained(path, **base_kwargs, **extra_kwargs)
        except TypeError as exc:
            last_type_error = exc
    if last_type_error is not None:
        raise last_type_error
    raise RuntimeError("unreachable CacheClip auxiliary model loader state")


def _decode_token_ids(tokenizer: Any, token_ids: Sequence[int]) -> str:
    return str(
        tokenizer.decode(
            [int(item) for item in token_ids],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )


def _encode_text(tokenizer: Any, text: str, add_special_tokens: bool) -> Dict[str, Any]:
    encoded = tokenizer(
        str(text),
        add_special_tokens=bool(add_special_tokens),
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    raw_offsets = encoded.pop("offset_mapping")
    if hasattr(raw_offsets, "detach"):
        offsets = raw_offsets[0].detach().cpu().tolist()
    else:
        offsets = list(raw_offsets)
    out = {"offset_mapping": offsets}
    for key, value in encoded.items():
        out[key] = value.cpu()
    return out


def _primary_token_offsets(tokenizer: Any, text: str, expected_tokens: int) -> List[Tuple[int, int]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = [
        (int(start), int(end))
        for start, end in encoded.get("offset_mapping", [])
        if int(end) > int(start)
    ]
    if len(offsets) != int(expected_tokens):
        raise RuntimeError(
            "CacheClip primary tokenizer offset mapping does not match original chunk token count: "
            "offsets=%d expected_tokens=%d" % (len(offsets), int(expected_tokens))
        )
    return offsets


def _spans_overlap(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    return int(left_start) < int(right_end) and int(right_start) < int(left_end)


def _aux_scores_for_cached_chunk(final_attention: Any, query_start: int, aux_chunk_indices: Sequence[int]) -> Dict[int, float]:
    import torch

    if final_attention.ndim != 3:
        raise RuntimeError("CacheClip expected final attention [heads, query, kv], got %r" % (tuple(final_attention.shape),))
    if not aux_chunk_indices:
        return {}
    query_count = int(final_attention.shape[1])
    query_indices = torch.arange(max(0, int(query_start)), query_count, dtype=torch.long)
    if query_indices.numel() <= 0:
        query_indices = torch.arange(query_count, dtype=torch.long)
    doc_tensor = torch.tensor([int(index) for index in aux_chunk_indices], dtype=torch.long)
    scores = final_attention.index_select(1, query_indices).index_select(2, doc_tensor).mean(dim=(0, 1))
    return {int(index): float(score) for index, score in zip(aux_chunk_indices, scores.tolist())}


def _project_aux_scores_to_primary_tokens(
    aux_scores: Dict[int, float],
    offset_mapping: Sequence[Sequence[int]],
    token_offsets: Sequence[Tuple[int, int]],
    chunk_char_start: int,
) -> List[float]:
    projected = []
    for local_start, local_end in token_offsets:
        global_start = int(chunk_char_start) + int(local_start)
        global_end = int(chunk_char_start) + int(local_end)
        values = []
        for aux_index, score in aux_scores.items():
            if _spans_overlap(
                int(offset_mapping[aux_index][0]),
                int(offset_mapping[aux_index][1]),
                global_start,
                global_end,
            ):
                values.append(float(score))
        projected.append(max(values) if values else 0.0)
    return projected


def _recompute_count(token_count: int, ratio: float, min_tokens: int) -> int:
    if token_count <= 0 or ratio <= 0.0:
        return 0
    count = int(math.floor(int(token_count) * min(max(float(ratio), 0.0), 1.0)))
    count = max(1, int(min_tokens), count)
    return min(count, int(token_count))


def _select_top_ratio_offsets(scores: Sequence[float], ratio: float, min_tokens: int) -> List[int]:
    indexed = [(float(score), int(index)) for index, score in enumerate(scores)]
    count = _recompute_count(len(indexed), ratio, min_tokens)
    if count <= 0:
        return []
    indexed.sort(key=lambda item: (-item[0], item[1]))
    selected = sorted(index for _score, index in indexed[:count])
    return selected


def _sliding_window_group_offsets(
    candidate_offsets: Iterable[int],
    token_count: int,
    window_tokens: int,
    min_candidates: int,
) -> List[int]:
    total = max(0, int(token_count))
    window = max(1, int(window_tokens))
    threshold = max(1, int(min_candidates))
    candidates = sorted({int(offset) for offset in candidate_offsets if 0 <= int(offset) < total})
    if total <= 0 or not candidates:
        return []
    candidate_set = set(candidates)
    grouped = set()
    for start in candidates:
        end = min(total, start + window)
        density = sum(1 for offset in range(start, end) if offset in candidate_set)
        if density >= threshold:
            grouped.update(range(start, end))
    return sorted(grouped)


def _cache_key(request: Dict[str, Any]) -> str:
    payload = {
        "primary_tokenizer": str(request["primary_tokenizer"]),
        "aux_model": str(request["aux_model"]),
        "aux_tokenizer": str(request.get("aux_tokenizer") or request["aux_model"]),
        "prefix_token_ids": [int(item) for item in request["prefix_token_ids"]],
        "doc_token_ids": [int(item) for item in request["doc_token_ids"]],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _prepare_aux_chunk_cache(request: Dict[str, Any]) -> Dict[str, Any]:
    import torch

    cache_root = str(request["aux_cache_root"])
    os.makedirs(cache_root, exist_ok=True)
    cache_path = os.path.join(cache_root, _cache_key(request) + ".pt")
    if os.path.isfile(cache_path):
        try:
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        except TypeError:
            cached = torch.load(cache_path, map_location="cpu")
        cached["cache_hit"] = True
        cached["cache_path"] = cache_path
        return cached

    primary_tokenizer = request["_primary_tokenizer"]
    aux_tokenizer = request["_aux_tokenizer"]
    aux_model = request["_aux_model"]
    aux_device = request["_aux_device"]

    prefix_text = _decode_token_ids(primary_tokenizer, request["prefix_token_ids"])
    doc_text = _decode_token_ids(primary_tokenizer, request["doc_token_ids"])
    combined_text = prefix_text + doc_text
    chunk_char_start = len(prefix_text)
    chunk_char_end = chunk_char_start + len(doc_text)
    token_offsets = _primary_token_offsets(primary_tokenizer, doc_text, len(request["doc_token_ids"]))
    encoded = _encode_text(aux_tokenizer, combined_text, True)
    offset_mapping = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    aux_chunk_indices = [
        int(index)
        for index, (start, end) in enumerate(offset_mapping)
        if int(end) > int(start) and _spans_overlap(int(start), int(end), chunk_char_start, chunk_char_end)
    ]
    if not aux_chunk_indices:
        raise RuntimeError("CacheClip auxiliary tokenizer produced no chunk tokens for the cached document span")

    token_count = int(encoded["input_ids"].shape[1])
    aux_max_length = int(request.get("aux_max_length", 0))
    if aux_max_length > 0 and token_count > aux_max_length:
        raise RuntimeError(
            "CacheClip auxiliary cache input exceeds aux_max_length: tokens=%d max_length=%d"
            % (token_count, aux_max_length)
        )

    with torch.inference_mode():
        outputs = aux_model(
            **{key: value.to(device=aux_device) for key, value in encoded.items() if key != "offset_mapping"},
            use_cache=True,
        )
    past_key_values = getattr(outputs, "past_key_values", None)
    if past_key_values is None:
        raise RuntimeError("CacheClip auxiliary model did not return past_key_values for chunk cache reuse")
    cached = {
        "past_key_values": _detach_past_key_values(past_key_values),
        "past_token_count": token_count,
        "offset_mapping": offset_mapping,
        "token_offsets": token_offsets,
        "aux_chunk_indices": aux_chunk_indices,
        "chunk_char_start": chunk_char_start,
        "chunk_char_end": chunk_char_end,
        "cache_hit": False,
        "cache_path": cache_path,
    }
    torch.save(cached, cache_path)
    return cached


def run_selector(request: Dict[str, Any]) -> Dict[str, Any]:
    import torch

    if int(request.get("group_min_candidates", 5)) > int(request.get("group_window_tokens", 8)):
        raise RuntimeError("group_min_candidates must be <= group_window_tokens")

    bootstrap_started = time.perf_counter()
    primary_tokenizer = _load_hf_tokenizer(
        str(request["primary_tokenizer"]),
        bool(request.get("aux_local_files_only", True)),
        bool(request.get("primary_trust_remote_code", False)),
    )
    aux_tokenizer = _load_hf_tokenizer(
        str(request.get("aux_tokenizer") or request["aux_model"]),
        bool(request.get("aux_local_files_only", True)),
        bool(request.get("aux_trust_remote_code", False)),
    )
    aux_model = _load_aux_causal_lm(
        str(request["aux_model"]),
        str(request.get("aux_torch_dtype", "float32")),
        bool(request.get("aux_local_files_only", True)),
        bool(request.get("aux_trust_remote_code", False)),
    )
    aux_device = _resolve_aux_device(str(request.get("aux_device", "cpu")))
    aux_model = aux_model.to(aux_device).eval()
    bootstrap_elapsed = time.perf_counter() - bootstrap_started

    request = dict(request)
    request["_primary_tokenizer"] = primary_tokenizer
    request["_aux_tokenizer"] = aux_tokenizer
    request["_aux_model"] = aux_model
    request["_aux_device"] = aux_device

    started = time.perf_counter()
    cache_started = time.perf_counter()
    cached = _prepare_aux_chunk_cache(request)
    cache_elapsed = time.perf_counter() - cache_started

    query_text = _decode_token_ids(primary_tokenizer, request["query_token_ids"])
    query_encoded = _encode_text(aux_tokenizer, query_text, False)
    query_input_ids = query_encoded["input_ids"].to(device=aux_device)
    query_token_count = int(query_input_ids.shape[1])
    if query_token_count <= 0:
        return {
            "selected_pic_local_indices": [],
            "selector_latency_s": 0.0,
            "selector_cache_hit": bool(cached.get("cache_hit", False)),
            "candidate_count": 0,
            "selected_count": 0,
            "cache_path": cached.get("cache_path", ""),
            "timings": {
                "bootstrap_load_s": float(bootstrap_elapsed),
                "cache_prepare_s": cache_elapsed,
                "query_attention_s": 0.0,
                "projection_s": 0.0,
                "ranking_grouping_s": 0.0,
                "total_s": 0.0,
            },
        }

    aux_max_length = int(request.get("aux_max_length", 0))
    total_aux_tokens = int(cached["past_token_count"]) + query_token_count
    if aux_max_length > 0 and total_aux_tokens > aux_max_length:
        raise RuntimeError(
            "CacheClip auxiliary input exceeds aux_max_length: tokens=%d max_length=%d"
            % (total_aux_tokens, aux_max_length)
        )

    query_tail_tokens = int(request.get("query_tail_tokens", 0))
    query_start = max(0, query_token_count - query_tail_tokens) if query_tail_tokens > 0 else 0
    query_started = time.perf_counter()
    attention_mask = torch.ones((1, total_aux_tokens), dtype=torch.long, device=aux_device)
    with torch.inference_mode():
        outputs = aux_model(
            input_ids=query_input_ids,
            attention_mask=attention_mask,
            past_key_values=_clone_past_key_values(cached["past_key_values"], aux_device),
            output_attentions=True,
            use_cache=True,
        )
    query_elapsed = time.perf_counter() - query_started
    attentions = getattr(outputs, "attentions", None)
    if not attentions:
        raise RuntimeError("CacheClip auxiliary model did not return attentions; use an eager HF attention backend")
    final_attention = attentions[-1][0].detach().to(torch.float32).cpu()

    project_started = time.perf_counter()
    aux_scores = _aux_scores_for_cached_chunk(final_attention, query_start, cached["aux_chunk_indices"])
    projected_scores = _project_aux_scores_to_primary_tokens(
        aux_scores,
        cached["offset_mapping"],
        cached["token_offsets"],
        int(cached["chunk_char_start"]),
    )
    project_elapsed = time.perf_counter() - project_started

    select_started = time.perf_counter()
    candidate_offsets = _select_top_ratio_offsets(
        projected_scores,
        float(request["ratio"]),
        int(request.get("min_tokens", 1)),
    )
    selected_offsets = _sliding_window_group_offsets(
        candidate_offsets,
        len(projected_scores),
        int(request.get("group_window_tokens", 8)),
        int(request.get("group_min_candidates", 5)),
    )
    select_elapsed = time.perf_counter() - select_started
    total_elapsed = time.perf_counter() - started

    return {
        "selected_pic_local_indices": [int(item) for item in selected_offsets],
        "selector_latency_s": float(total_elapsed),
        "selector_cache_hit": bool(cached.get("cache_hit", False)),
        "candidate_count": len(candidate_offsets),
        "selected_count": len(selected_offsets),
        "cache_path": str(cached.get("cache_path", "")),
        "aux_device": str(aux_device),
        "aux_model": str(request["aux_model"]),
        "primary_tokenizer": str(request["primary_tokenizer"]),
        "timings": {
            "bootstrap_load_s": float(bootstrap_elapsed),
            "cache_prepare_s": float(cache_elapsed),
            "query_attention_s": float(query_elapsed),
            "projection_s": float(project_elapsed),
            "ranking_grouping_s": float(select_elapsed),
            "total_s": float(total_elapsed),
        },
    }


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
        result = run_selector(payload)
        sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
        sys.stdout.flush()
        return 0
    except Exception as exc:
        sys.stdout.write(json.dumps({"error": str(exc)}, ensure_ascii=False) + "\n")
        sys.stdout.flush()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
