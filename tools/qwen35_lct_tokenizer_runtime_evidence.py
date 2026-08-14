from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

PR65_HEAD = "eff0f7c64fd65b0cefce29f2510bab96130e799a"
DONOR_SHA256 = "65b753ea835627f7b511143c6ceb976525c7f21f5df8c664bc0a9c23d1c49921"
SOURCE_SET_SHA256 = "ec3a6c03cdb1bd88dc01a2cd745e6aed6a582512543c66088eb6e1193dfab0b2"
SOURCE_SET_IDENTITY = "415e8ec8ecce62446286129ee575aa51794be74c9f506b69639b71a3dcb8192"
REFERENCE_TOKENIZER_MECHANISM = "sha256:e46db70ab44031617b77c5f4c0254d9959e78a1b255d5d45c45ff5c6424facf4"
REFERENCE_TOKENIZER_HOLDOUT = "aabb23a0f1572b466b39ffefcf3510903dd25768ad95ac32836cb0c48cb919b9"
LLAMA_COMMIT = "74ade52741203e5c8f81eaf06a96cb1cfe15f2a3"
CUDA_QUANTIZE_BLOB = "39a500a170414a9dfeb798d22f562585fdbf3a84"
CUDA_VECDOT_BLOB = "d1741cc8d7babdf6726f9928a3c908ae2df12dd6"

WIDTH = 5120
VOCAB = 248320
QK_K = 256
Q6K_BLOCK_BYTES = 210
Q6K_BLOCKS_PER_ROW = WIDTH // QK_K
Q6K_ROW_BYTES = Q6K_BLOCKS_PER_ROW * Q6K_BLOCK_BYTES
Q8_1_BLOCK = 32
Q4K_SCALE_BYTES = 12
Q4K_BLOCK_BYTES = 2 + 2 + Q4K_SCALE_BYTES + QK_K // 2
Q4K_ROW_BYTES = (WIDTH // QK_K) * Q4K_BLOCK_BYTES

OUTPUT_START = 10992928
OUTPUT_BYTES = 1042944000
OUTPUT_RAW_SHA256 = "d8dff3670047a7e703fe5a32c33dc12dc8f2498d031d8e3b5f9d675e103536a1"
EMBED_START = 1053957408
EMBED_BYTES = 715161600
EMBED_RAW_SHA256 = "205ceab06ee626b80158f5745cecee15de32f84f1de7a17e1210b2bea32d4e0e"

OUTPUT_MAX_ABS = 5.0e-4
OUTPUT_NRMSE = 8.84687028317136e-7
EMBED_MAX_ABS = 2.0e-6
EMBED_NRMSE = 5.0e-7

FIT_CONTEXTS = tuple(f"f{i:03d}" for i in range(1, 49))
HOLDOUT_CONTEXTS = tuple(f"h{i:03d}" for i in range(1, 17))


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def address(value) -> str:
    return "sha256:" + (value if isinstance(value, str) and len(value) == 64 else digest(value))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def signed_write(path: Path, core: dict) -> dict:
    if path.exists():
        raise RuntimeError(f"refusing to replace immutable receipt: {path}")
    value = {**core, "identity": digest(core)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return value


def signed_load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    identity = value.get("identity")
    core = {key: item for key, item in value.items() if key != "identity"}
    if identity != digest(core):
        raise RuntimeError(f"signed receipt identity mismatch: {path}")
    return value


def load_source(path: Path) -> dict:
    if file_sha256(path) != SOURCE_SET_SHA256:
        raise RuntimeError("representative source set changed")
    source = json.loads(path.read_text(encoding="utf-8"))
    if source.get("identity") != SOURCE_SET_IDENTITY or source.get("donor_identity") != f"sha256:{DONOR_SHA256}":
        raise RuntimeError("representative source authority changed")
    components = source["evidence"]["tokenizer_embedding"]["component_manifest"]["components"]
    by_name = {row["name"]: row for row in components}
    output = by_name.get("output.weight")
    embedding = by_name.get("token_embd.weight")
    if not output or not embedding:
        raise RuntimeError("tokenizer source components are incomplete")
    expected_output = ([WIDTH, VOCAB], 14, OUTPUT_START, OUTPUT_BYTES, OUTPUT_RAW_SHA256)
    actual_output = (output["dimensions"], output["ggml_type"], output["storage_start"], output["storage_span_bytes"], output["raw_sha256"])
    expected_embedding = ([WIDTH, VOCAB], 12, EMBED_START, EMBED_BYTES, EMBED_RAW_SHA256)
    actual_embedding = (embedding["dimensions"], embedding["ggml_type"], embedding["storage_start"], embedding["storage_span_bytes"], embedding["raw_sha256"])
    if actual_output != expected_output or actual_embedding != expected_embedding:
        raise RuntimeError("tokenizer component binding changed")
    return source


def read_tokens(path: Path) -> list[int]:
    tokens = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        position, token = line.split("\t")
        if int(position) != index:
            raise RuntimeError(f"token position changed: {path}")
        value = int(token)
        if value < 0 or value >= VOCAB:
            raise RuntimeError(f"token id outside vocabulary: {value}")
        tokens.append(value)
    if not tokens:
        raise RuntimeError(f"empty token list: {path}")
    return tokens


def tensor_file(context: Path, name: str) -> Path:
    lines = (context / "tensor-index.tsv").read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "name\tgraph_name\toccurrence\ttype\tshape\tbytes\tfile":
        raise RuntimeError("tensor index schema changed")
    matches = []
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) == 7 and fields[0] == name:
            matches.append(fields)
    if len(matches) != 1:
        raise RuntimeError(f"tensor binding is not unique: {context.name}/{name}")
    fields = matches[0]
    if fields[3] != "f32":
        raise RuntimeError(f"tensor type changed: {context.name}/{name}")
    path = context / fields[6]
    if path.stat().st_size != int(fields[5]):
        raise RuntimeError(f"tensor byte count changed: {context.name}/{name}")
    return path


def get_scale_min_k4(index: int, scales: memoryview) -> tuple[int, int]:
    if index < 4:
        return scales[index] & 0x3F, scales[index + 4] & 0x3F
    scale = (scales[index + 4] & 0x0F) | ((scales[index - 4] & 0xC0) >> 2)
    minimum = ((scales[index + 4] >> 4) & 0x0F) | ((scales[index] & 0xC0) >> 2)
    return scale, minimum


def dequantize_q4k_row(raw: bytes) -> np.ndarray:
    if len(raw) != Q4K_ROW_BYTES:
        raise RuntimeError("Q4_K row byte count changed")
    out = np.empty(WIDTH, dtype=np.float32)
    cursor = 0
    target = 0
    for _ in range(WIDTH // QK_K):
        block = memoryview(raw)[cursor:cursor + Q4K_BLOCK_BYTES]
        d = np.frombuffer(block[:2], dtype="<f2", count=1)[0].astype(np.float32)
        dmin = np.frombuffer(block[2:4], dtype="<f2", count=1)[0].astype(np.float32)
        scales = block[4:4 + Q4K_SCALE_BYTES]
        qs = np.frombuffer(block[4 + Q4K_SCALE_BYTES:], dtype=np.uint8, count=QK_K // 2)
        for group in range(4):
            lo_scale, lo_min = get_scale_min_k4(2 * group, scales)
            hi_scale, hi_min = get_scale_min_k4(2 * group + 1, scales)
            packed = qs[32 * group:32 * (group + 1)]
            low = (packed & 0x0F).astype(np.float32)
            high = (packed >> 4).astype(np.float32)
            out[target:target + 32] = np.float32(d * np.float32(lo_scale)) * low - np.float32(dmin * np.float32(lo_min))
            out[target + 32:target + 64] = np.float32(d * np.float32(hi_scale)) * high - np.float32(dmin * np.float32(hi_min))
            target += 64
        cursor += Q4K_BLOCK_BYTES
    return out


def dequantize_q6k_rows(raw: bytes, rows: int) -> np.ndarray:
    if len(raw) != rows * Q6K_ROW_BYTES:
        raise RuntimeError("Q6_K batch byte count changed")
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(rows, Q6K_BLOCKS_PER_ROW, Q6K_BLOCK_BYTES)
    ql = blocks[:, :, :128]
    qh = blocks[:, :, 128:192]
    scales = blocks[:, :, 192:208].view(np.int8).astype(np.float32)
    d = np.ascontiguousarray(blocks[:, :, 208:210]).view("<f2").reshape(rows, Q6K_BLOCKS_PER_ROW).astype(np.float32)
    values = np.empty((rows, Q6K_BLOCKS_PER_ROW, QK_K), dtype=np.float32)
    scale_index = np.repeat(np.arange(2, dtype=np.int64), 16)
    for half in range(2):
        ql0 = ql[:, :, half * 64:half * 64 + 32]
        ql1 = ql[:, :, half * 64 + 32:half * 64 + 64]
        qh0 = qh[:, :, half * 32:half * 32 + 32]
        local = scales[:, :, half * 8:half * 8 + 8]
        q1 = ((ql0 & 0x0F) | (((qh0 >> 0) & 0x03) << 4)).astype(np.int16) - 32
        q2 = ((ql1 & 0x0F) | (((qh0 >> 2) & 0x03) << 4)).astype(np.int16) - 32
        q3 = ((ql0 >> 4) | (((qh0 >> 4) & 0x03) << 4)).astype(np.int16) - 32
        q4 = ((ql1 >> 4) | (((qh0 >> 6) & 0x03) << 4)).astype(np.int16) - 32
        base = half * 128
        values[:, :, base:base + 32] = d[:, :, None] * local[:, :, scale_index + 0] * q1.astype(np.float32)
        values[:, :, base + 32:base + 64] = d[:, :, None] * local[:, :, scale_index + 2] * q2.astype(np.float32)
        values[:, :, base + 64:base + 96] = d[:, :, None] * local[:, :, scale_index + 4] * q3.astype(np.float32)
        values[:, :, base + 96:base + 128] = d[:, :, None] * local[:, :, scale_index + 6] * q4.astype(np.float32)
    return values.reshape(rows, WIDTH)


def cuda_roundf(values: np.ndarray) -> np.ndarray:
    return np.copysign(np.floor(np.abs(values) + np.float32(0.5)), values)


def q8_1_roundtrip(columns: np.ndarray) -> np.ndarray:
    count = columns.shape[1]
    blocks = np.ascontiguousarray(columns.T).reshape(count, WIDTH // Q8_1_BLOCK, Q8_1_BLOCK)
    amax = np.max(np.abs(blocks), axis=2).astype(np.float32, copy=False)
    d = (amax / np.float32(127.0)).astype(np.float32, copy=False)
    safe = np.where(d == np.float32(0.0), np.float32(1.0), d)
    q = cuda_roundf(blocks / safe[:, :, None])
    q = np.where(d[:, :, None] == np.float32(0.0), np.float32(0.0), q).astype(np.int8)
    stored_d = d.astype(np.float16).astype(np.float32)
    return np.ascontiguousarray((q.astype(np.float32) * stored_d[:, :, None]).reshape(count, WIDTH).T)


def metric(actual: np.ndarray, expected: np.ndarray) -> dict:
    if actual.shape != expected.shape:
        raise RuntimeError(f"metric shape mismatch: {actual.shape} != {expected.shape}")
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    reference = expected.astype(np.float64)
    rms = math.sqrt(float(np.mean(delta * delta)))
    ref_rms = math.sqrt(float(np.mean(reference * reference)))
    return {
        "elements": int(actual.size),
        "max_abs": float(np.max(np.abs(delta))),
        "rms": rms,
        "reference_rms": ref_rms,
        "nrmse": rms / max(ref_rms, 1.0e-30),
    }


def evaluate_embedding(contexts: tuple[str, ...], atlas: Path, donor: Path) -> dict:
    rows = []
    with donor.open("rb") as handle:
        for context_id in contexts:
            context = atlas / context_id
            token_ids = read_tokens(context / "tokens.tsv")
            expected = np.fromfile(tensor_file(context, "model.input_embed"), dtype="<f4")
            expected = expected.reshape((WIDTH, len(token_ids)), order="F")
            predicted = np.empty_like(expected)
            for position, token in enumerate(token_ids):
                handle.seek(EMBED_START + token * Q4K_ROW_BYTES)
                predicted[:, position] = dequantize_q4k_row(handle.read(Q4K_ROW_BYTES))
            row = metric(predicted, expected)
            row["context"] = context_id
            rows.append(row)
    return aggregate_metrics(rows)


def aggregate_metrics(rows: list[dict]) -> dict:
    total_elements = sum(row["elements"] for row in rows)
    sum_sq = sum(row["rms"] ** 2 * row["elements"] for row in rows)
    reference_sq = sum(row["reference_rms"] ** 2 * row["elements"] for row in rows)
    rms = math.sqrt(sum_sq / total_elements)
    reference_rms = math.sqrt(reference_sq / total_elements)
    return {
        "contexts": len(rows),
        "elements": total_elements,
        "max_abs": max(row["max_abs"] for row in rows),
        "nrmse": rms / max(reference_rms, 1.0e-30),
        "max_context_nrmse": max(row["nrmse"] for row in rows),
        "per_context": rows,
    }


def evaluate_output(contexts: tuple[str, ...], atlas: Path, supplement: Path, donor: Path, row_batch: int) -> dict:
    hidden = []
    targets = []
    for context_id in contexts:
        norm = supplement / context_id / "result_norm.f32"
        logits = atlas / context_id / "logits.f32"
        if norm.stat().st_size != WIDTH * 4 or logits.stat().st_size != VOCAB * 4:
            raise RuntimeError(f"tokenizer output payload changed: {context_id}")
        hidden.append(np.fromfile(norm, dtype="<f4"))
        targets.append(np.memmap(logits, dtype="<f4", mode="r", shape=(VOCAB,)))
    hidden_matrix = q8_1_roundtrip(np.stack(hidden, axis=1).astype(np.float32, copy=False))
    count = len(contexts)
    sum_sq = np.zeros(count, dtype=np.float64)
    ref_sq = np.zeros(count, dtype=np.float64)
    max_abs = np.zeros(count, dtype=np.float64)
    predicted_top = np.full(count, -1, dtype=np.int64)
    predicted_top_value = np.full(count, -np.inf, dtype=np.float32)
    target_top = np.full(count, -1, dtype=np.int64)
    target_top_value = np.full(count, -np.inf, dtype=np.float32)
    raw_hash = hashlib.sha256()
    with donor.open("rb") as handle:
        handle.seek(OUTPUT_START)
        for start in range(0, VOCAB, row_batch):
            rows = min(row_batch, VOCAB - start)
            raw = handle.read(rows * Q6K_ROW_BYTES)
            raw_hash.update(raw)
            weights = dequantize_q6k_rows(raw, rows)
            predicted = np.matmul(weights, hidden_matrix).astype(np.float32, copy=False)
            expected = np.stack([target[start:start + rows] for target in targets], axis=1).astype(np.float32, copy=False)
            delta = predicted.astype(np.float64) - expected.astype(np.float64)
            sum_sq += np.sum(delta * delta, axis=0)
            ref_sq += np.sum(expected.astype(np.float64) ** 2, axis=0)
            max_abs = np.maximum(max_abs, np.max(np.abs(delta), axis=0))
            p_index = np.argmax(predicted, axis=0)
            p_value = predicted[p_index, np.arange(count)]
            update = p_value > predicted_top_value
            predicted_top_value[update] = p_value[update]
            predicted_top[update] = start + p_index[update]
            t_index = np.argmax(expected, axis=0)
            t_value = expected[t_index, np.arange(count)]
            update = t_value > target_top_value
            target_top_value[update] = t_value[update]
            target_top[update] = start + t_index[update]
    if raw_hash.hexdigest() != OUTPUT_RAW_SHA256:
        raise RuntimeError("output.weight source hash changed during execution")
    rows = []
    for index, context_id in enumerate(contexts):
        rms = math.sqrt(float(sum_sq[index]) / VOCAB)
        ref_rms = math.sqrt(float(ref_sq[index]) / VOCAB)
        rows.append({
            "context": context_id,
            "elements": VOCAB,
            "max_abs": float(max_abs[index]),
            "rms": rms,
            "reference_rms": ref_rms,
            "nrmse": rms / max(ref_rms, 1.0e-30),
            "predicted_top_token": int(predicted_top[index]),
            "reference_top_token": int(target_top[index]),
            "top_token_match": bool(predicted_top[index] == target_top[index]),
        })
    aggregate = aggregate_metrics(rows)
    aggregate["top_token_matches"] = sum(row["top_token_match"] for row in rows)
    aggregate["top_token_total"] = len(rows)
    return aggregate


def mechanism_core() -> dict:
    return {
        "schema": "qsa.qwen35-lct-tokenizer-runtime-mechanism.v1",
        "pr65_head": PR65_HEAD,
        "donor_sha256": DONOR_SHA256,
        "source_set_sha256": SOURCE_SET_SHA256,
        "reference_tokenizer_mechanism": REFERENCE_TOKENIZER_MECHANISM,
        "llama_commit": LLAMA_COMMIT,
        "cuda_quantize_blob": CUDA_QUANTIZE_BLOB,
        "cuda_vecdot_blob": CUDA_VECDOT_BLOB,
        "embedding_operator": "Q4_K physical row lookup -> F32 vector",
        "output_operator": "Q6_K output.weight x CUDA-Q8_1 result_norm",
        "q8_1": "block=32; d=amax/127; q=CUDA roundf(x/d); stored d=fp16",
        "fit_contexts": list(FIT_CONTEXTS),
        "holdout_contexts": list(HOLDOUT_CONTEXTS),
        "thresholds": {
            "embedding_max_abs": EMBED_MAX_ABS,
            "embedding_nrmse": EMBED_NRMSE,
            "output_max_abs": OUTPUT_MAX_ABS,
            "output_nrmse": OUTPUT_NRMSE,
            "output_top_token_match": True,
        },
    }


def passed(evidence: dict) -> bool:
    embedding = evidence["embedding"]
    output = evidence["output"]
    return (
        embedding["max_abs"] <= EMBED_MAX_ABS
        and embedding["max_context_nrmse"] <= EMBED_NRMSE
        and output["max_abs"] <= OUTPUT_MAX_ABS
        and output["max_context_nrmse"] <= OUTPUT_NRMSE
        and output["top_token_matches"] == output["top_token_total"]
    )


def run_fit(args) -> int:
    load_source(args.source_fragments)
    if file_sha256(args.donor) != DONOR_SHA256:
        raise RuntimeError("donor identity changed")
    mechanism = mechanism_core()
    mechanism_identity = address(mechanism)
    evidence = {
        "embedding": evaluate_embedding(FIT_CONTEXTS, args.atlas, args.donor),
        "output": evaluate_output(FIT_CONTEXTS, args.atlas, args.supplement, args.donor, args.row_batch),
    }
    ok = passed(evidence)
    signed_write(args.output, {
        "schema": "qsa.qwen35-lct-tokenizer-runtime-evidence.v1",
        "phase": "fit",
        "mechanism": mechanism,
        "mechanism_identity": mechanism_identity,
        "evidence": evidence,
        "fit_pass": ok,
        "holdout_authorized": ok,
        "qsa_runtime_equivalence_verified": False,
    })
    return 0 if ok else 2


def run_holdout(args) -> int:
    fit = signed_load(args.fit_receipt)
    mechanism = mechanism_core()
    mechanism_identity = address(mechanism)
    if not fit.get("fit_pass") or not fit.get("holdout_authorized") or fit.get("mechanism_identity") != mechanism_identity:
        raise RuntimeError("fit receipt does not authorize frozen tokenizer holdout")
    load_source(args.source_fragments)
    if file_sha256(args.donor) != DONOR_SHA256:
        raise RuntimeError("donor identity changed")
    evidence = {
        "embedding": evaluate_embedding(HOLDOUT_CONTEXTS, args.atlas, args.donor),
        "output": evaluate_output(HOLDOUT_CONTEXTS, args.atlas, args.supplement, args.donor, args.row_batch),
    }
    ok = passed(evidence)
    operator_core = {
        "schema": "qsa.qwen35-lct-tokenizer-operator-receipt.v1",
        "mechanism_identity": mechanism_identity,
        "reference_holdout_identity": REFERENCE_TOKENIZER_HOLDOUT,
        "fit_receipt_sha256": file_sha256(args.fit_receipt),
        "evidence": evidence,
        "operator_mismatch_count": 0 if ok else 1,
        "operator_equivalence_verified": ok,
    }
    operator_root = address(operator_core)
    signed_write(args.output, {
        "schema": "qsa.qwen35-lct-tokenizer-runtime-evidence.v1",
        "phase": "holdout",
        "mechanism": mechanism,
        "mechanism_identity": mechanism_identity,
        "fit_receipt_sha256": file_sha256(args.fit_receipt),
        "evidence": evidence,
        "operator_receipt_root": operator_root,
        "operator_mismatch_count": 0 if ok else 1,
        "operator_equivalence_verified": ok,
        "qsa_runtime_equivalence_verified": ok,
    })
    return 0 if ok else 2


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    sub = value.add_subparsers(dest="phase", required=True)
    for name in ("fit", "holdout"):
        command = sub.add_parser(name)
        command.add_argument("--donor", type=Path, required=True)
        command.add_argument("--source-fragments", type=Path, required=True)
        command.add_argument("--atlas", type=Path, required=True)
        command.add_argument("--supplement", type=Path, required=True)
        command.add_argument("--row-batch", type=int, default=512)
        command.add_argument("--output", type=Path, required=True)
        if name == "holdout":
            command.add_argument("--fit-receipt", type=Path, required=True)
    return value


def main() -> int:
    args = parser().parse_args()
    if args.row_batch <= 0 or VOCAB % args.row_batch:
        raise RuntimeError("row batch must be a positive divisor of vocabulary size")
    if args.phase == "fit":
        return run_fit(args)
    return run_holdout(args)


if __name__ == "__main__":
    raise SystemExit(main())
