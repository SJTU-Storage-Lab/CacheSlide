"""Bounded CCPE calibration and immutable, adapter-bound profile artifacts."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .artifacts import AdapterBundle, file_sha256
from .contracts import RequestPlan, digest
from .position import CCPEProfile, ChunkIdentity, CoPE, CoPEPositionTrace

_FORMAT = "cacheslide-profiles-v1"
_MAX_JSON_BYTES = 8 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _integer(value: object, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _label(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 256
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"invalid {name}")
    return value


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"invalid {name} SHA-256")
    return value


def _fields(value: object, expected: set[str], name: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"invalid {name} fields")
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _nonfinite_json(value: str) -> None:
    raise ValueError(f"nonfinite JSON value: {value}")


def _artifact_file(directory: Path, name: str) -> Path:
    path = directory / name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"artifact requires a regular, non-symlink {name}")
    return path


def _chunks(plan: RequestPlan) -> tuple[ChunkIdentity, ...]:
    return tuple(
        ChunkIdentity(chunk_id, token_digest, length)
        for chunk_id, token_digest, length in plan.fixed_layout
    )


def _read_manifest(path: Path) -> dict:
    if path.stat().st_size > _MAX_JSON_BYTES:
        raise ValueError("profile metadata exceeds size budget")
    metadata = json.loads(
        path.read_text(),
        object_pairs_hook=_unique_object,
        parse_constant=_nonfinite_json,
    )
    _fields(
        metadata,
        {
            "format",
            "adapter_identity",
            "trained_profile_version",
            "num_layers",
            "query_heads",
            "max_positions",
            "profiles",
            "tensors_sha256",
            "manifest_sha256",
        },
        "profile manifest",
    )
    checksum = _hash(metadata["manifest_sha256"], "manifest")
    payload = {
        key: value for key, value in metadata.items() if key != "manifest_sha256"
    }
    if digest(payload) != checksum:
        raise ValueError("profile manifest checksum mismatch")
    return metadata


class ProfileBundle:
    """Load only canonical profiles, checking metadata before tensor allocation.

    ``max_elements`` bounds the sum of all stored tensor elements, including
    ordinal position vectors. A miss remains explicit through ``get`` returning
    None; callers must then use contextual attention rather than extrapolation.
    """

    def __init__(
        self, path: str | Path, adapter_identity: str, max_elements: int = 1_000_000
    ):
        _integer(max_elements, "max_elements")
        _hash(adapter_identity, "adapter identity")
        self.path = Path(path).resolve(strict=True)
        metadata = _read_manifest(_artifact_file(self.path, "manifest.json"))
        if metadata["format"] != _FORMAT:
            raise ValueError("unsupported CacheSlide profile format version")
        if metadata["adapter_identity"] != adapter_identity:
            raise ValueError("profile adapter identity mismatch")
        version = _label(metadata["trained_profile_version"], "profile version")
        layers = _integer(metadata["num_layers"], "num_layers")
        heads = _integer(metadata["query_heads"], "query_heads")
        max_positions = _integer(metadata["max_positions"], "max_positions", 2)
        entries = metadata["profiles"]
        if not isinstance(entries, dict) or not entries:
            raise ValueError("profile manifest requires nonempty profiles")
        specifications = {}
        expected_tensors = {}
        total = 0
        for key, entry in entries.items():
            _hash(key, "profile key")
            _fields(
                entry,
                {"layer", "chunks", "histogram_bin_width", "sample_count"},
                "profile entry",
            )
            layer = _integer(entry["layer"], "layer", 0)
            if layer >= layers:
                raise ValueError("profile layer outside model geometry")
            raw_chunks = entry["chunks"]
            if not isinstance(raw_chunks, list) or not 1 <= len(raw_chunks) <= 1024:
                raise ValueError("profile requires ordered fixed chunks")
            chunks = []
            for raw in raw_chunks:
                _fields(raw, {"role", "content_hash", "length"}, "chunk identity")
                chunks.append(
                    ChunkIdentity(
                        _label(raw["role"], "chunk role"),
                        _hash(raw["content_hash"], "chunk"),
                        _integer(raw["length"], "chunk length"),
                    )
                )
            if len({chunk.role for chunk in chunks}) != len(chunks):
                raise ValueError("duplicate ordered fixed chunk identity")
            width = entry["histogram_bin_width"]
            if (
                type(width) not in (float, int)
                or not math.isfinite(width)
                or width <= 0
            ):
                raise ValueError("invalid histogram bin width")
            _integer(entry["sample_count"], "sample_count")
            count = sum(chunk.length for chunk in chunks)
            total += heads * count * count + 2 * count
            if total > max_elements:
                raise ValueError("profile tensors exceed max_elements budget")
            expected_tensors[f"{key}.positions"] = ((heads, count, count), "F32")
            for suffix in ("query_positions", "key_positions"):
                expected_tensors[f"{key}.{suffix}"] = ((count,), "I64")
            specifications[key] = (entry, tuple(chunks), count)

        tensors_path = _artifact_file(self.path, "profiles.safetensors")
        if tensors_path.stat().st_size > max_elements * 8 + _MAX_JSON_BYTES:
            raise ValueError("profile tensor file exceeds size budget")
        if file_sha256(tensors_path) != _hash(metadata["tensors_sha256"], "tensor"):
            raise ValueError("profile tensor checksum mismatch")
        self._profiles: dict[str, CCPEProfile] = {}
        # Slices inspect the safetensors header without materializing any tensor.
        with safe_open(tensors_path, framework="pt", device="cpu") as source:
            if set(source.keys()) != set(expected_tensors):
                raise ValueError("profile tensor keys mismatch")
            for name, (shape, dtype) in expected_tensors.items():
                tensor_slice = source.get_slice(name)
                if (
                    tuple(tensor_slice.get_shape()) != shape
                    or tensor_slice.get_dtype() != dtype
                ):
                    raise ValueError("profile tensor layout/dtype mismatch")
            for key, (entry, chunks, count) in specifications.items():
                positions = source.get_tensor(f"{key}.positions")
                query_positions = source.get_tensor(f"{key}.query_positions")
                key_positions = source.get_tensor(f"{key}.key_positions")
                ordinals = torch.arange(count)
                if not torch.equal(query_positions, ordinals) or not torch.equal(
                    key_positions, ordinals
                ):
                    raise ValueError(
                        "profile query/key layout must be canonical ordinals"
                    )
                if (
                    not torch.isfinite(positions).all()
                    or (positions < 0).any()
                    or (positions > max_positions - 1).any()
                ):
                    raise ValueError("profile positions must be finite and bounded")
                if positions.triu(diagonal=1).count_nonzero():
                    raise ValueError("profile has noncausal canonical positions")
                self._profiles[key] = CCPEProfile(
                    trained_profile_version=version,
                    checkpoint_id=adapter_identity,
                    chunks=chunks,
                    query_positions=query_positions,
                    key_positions=key_positions,
                    canonical_positions=positions,
                    max_positions=max_positions,
                    histogram_bin_width=float(entry["histogram_bin_width"]),
                    sample_count=entry["sample_count"],
                )
        self.metadata = metadata
        self.identity = digest(metadata)
        self.adapter_identity = adapter_identity

    def get(self, plan: RequestPlan, layer: int) -> CCPEProfile | None:
        """Select by exact namespace/task/ordered fixed layout and model layer."""
        _integer(layer, "layer", 0)
        key = plan.profile_key(self.adapter_identity, layer)
        profile = self._profiles.get(key)
        if profile is None:
            return None
        if self.metadata["profiles"][key]["layer"] != layer:
            raise ValueError("profile key refers to a different layer")
        count = len(plan.fixed_indices)
        profile.lookup(
            _chunks(plan),
            torch.arange(count),
            torch.arange(count),
            checkpoint_id=self.adapter_identity,
            trained_profile_version=self.metadata["trained_profile_version"],
        )
        return profile


def calibrate_profiles(
    model_dir: str | Path,
    adapter_dir: str | Path,
    requests: Sequence[RequestPlan],
    output: str | Path,
    *,
    device: str = "cpu",
    max_elements: int = 1_000_000,
    trained_profile_version: str = "v1",
) -> Path:
    """Run frozen trained attention on full contexts and save joint CCPE modes.

    Full source logits remain in memory only long enough to verify contextual
    gates, including dynamic-token contributions. Artifacts contain no source
    traces, logits, request text, token IDs, or dynamic chunk metadata.
    """
    _integer(max_elements, "max_elements")
    _label(trained_profile_version, "trained_profile_version")
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"profile output already exists: {output}")
    if not requests:
        raise ValueError("calibration requires at least one request")
    plans = []
    for request in requests:
        if not isinstance(request, RequestPlan):
            raise ValueError("calibration requires validated RequestPlan inputs")
        plan = RequestPlan.parse(request.to_json(), request.token_ids)
        if not plan.fixed_indices:
            raise ValueError("each calibration request needs a reusable fixed chunk")
        plans.append(plan)
    bundle = AdapterBundle(adapter_dir, model_dir)
    layers, heads = (
        bundle.config[name] for name in ("num_hidden_layers", "num_attention_heads")
    )
    chunks_by_key = {}
    layer_by_key = {}
    # Budget the whole run before creating quadratic traces, not just each sample.
    required = 0
    for plan in plans:
        length, fixed = len(plan.token_ids), len(plan.fixed_indices)
        required += layers * heads * (length * length + fixed * fixed)
        for layer in range(layers):
            key = plan.profile_key(bundle.identity, layer)
            if key not in chunks_by_key:
                required += heads * fixed * fixed + 2 * fixed
                chunks_by_key[key] = _chunks(plan)
                layer_by_key[key] = layer
        if required > max_elements:
            raise ValueError(
                "CCPE calibration source traces exceed max_elements budget"
            )

    from .reference import ReferenceLlama

    model = ReferenceLlama.from_checkpoint(
        model_dir,
        rank=bundle.metadata["rank"],
        max_positions=bundle.metadata["max_positions"],
        device=device,
        dtype=torch.float32,
    )
    model.load_adapters(bundle)
    model.eval().requires_grad_(False)
    traces: dict[str, list[CoPEPositionTrace]] = {key: [] for key in chunks_by_key}
    for plan in plans:
        fixed_indices = torch.tensor(plan.fixed_indices, dtype=torch.long)
        observed = set()

        def observe(
            layer: int, query: torch.Tensor, key: torch.Tensor, cope: CoPE
        ) -> None:
            if type(layer) is not int or not 0 <= layer < layers or layer in observed:
                raise ValueError("reference observer emitted an invalid/repeated layer")
            length = len(plan.token_ids)
            if query.shape != (
                length,
                heads,
                bundle.config["head_dim"],
            ) or key.shape != (
                length,
                bundle.config["num_key_value_heads"],
                bundle.config["head_dim"],
            ):
                raise ValueError("reference observer Q/K layout mismatch")
            trace = cope.position_trace(
                query,
                key,
                torch.arange(length, device=query.device),
                checkpoint_id=bundle.identity,
                max_elements=max_elements,
            ).project(fixed_indices, fixed_indices, canonical_positions=True)
            traces[plan.profile_key(bundle.identity, layer)].append(trace)
            observed.add(layer)

        with torch.inference_mode():
            model.forward(
                torch.tensor(plan.token_ids, dtype=torch.long, device=device),
                observer=observe,
            )
        if observed != set(range(layers)):
            raise ValueError("reference model did not observe every attention layer")

    profiles = {
        key: CCPEProfile.calibrate(
            samples,
            chunks_by_key[key],
            trained_profile_version=trained_profile_version,
            max_elements=max_elements,
        )
        for key, samples in traces.items()
    }
    tensors = {}
    entries = {}
    for key, profile in sorted(profiles.items()):
        tensors[f"{key}.positions"] = profile.canonical_positions.float().contiguous()
        tensors[f"{key}.query_positions"] = profile.query_positions.contiguous()
        tensors[f"{key}.key_positions"] = profile.key_positions.contiguous()
        entries[key] = {
            "layer": layer_by_key[key],
            "chunks": [
                {
                    "role": chunk.role,
                    "content_hash": chunk.content_hash,
                    "length": chunk.length,
                }
                for chunk in profile.chunks
            ],
            "histogram_bin_width": profile.histogram_bin_width,
            "sample_count": profile.sample_count,
        }
    # Reserve a new directory only after calibration succeeds; never overwrite.
    output.mkdir(parents=True, exist_ok=False)
    tensors_path = output / "profiles.safetensors"
    save_file(tensors, str(tensors_path))
    metadata = {
        "format": _FORMAT,
        "adapter_identity": bundle.identity,
        "trained_profile_version": trained_profile_version,
        "num_layers": layers,
        "query_heads": heads,
        "max_positions": bundle.metadata["max_positions"],
        "profiles": entries,
        "tensors_sha256": file_sha256(tensors_path),
    }
    metadata["manifest_sha256"] = digest(metadata)
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return output
