from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

FULL_QWEN_SHA256 = "27fb003f98cb2d52bfe6d39fd3c20362ed180932f7fe5045b9ae524d3f1f47d1"
FULL_QWEN_IDENTITY = "300f5b321465d0c0c959e0d63eaedb1249e6a2d2bbb34fcfdd6cb520cb65af12"
REPRESENTATIVE_QWEN_SHA256 = "ec3a6c03cdb1bd88dc01a2cd745e6aed6a582512543c66088eb6e1193dfab0b2"
REPRESENTATIVE_QSA_SET_SHA256 = "400db266a8c4ad64dcd5a3e04587fb3356108823adbf38fac95af59b214b9aac"
DONOR_IDENTITY = "sha256:65b753ea835627f7b511143c6ceb976525c7f21f5df8c664bc0a9c23d1c49921"
TOPOLOGY_IDENTITY = "sha256:e2b88b52a3f7ecbb0aed0d88c48e1af0e212bc7cdead4ea11c858fe3d718d108"
TOTAL_COMPONENTS = 851
TOTAL_TENSOR_BYTES = 19_084_773_376
FAMILY = {
    "ssm_recurrent": {"instances": 48, "components": 672, "bytes": 13_136_898_048, "representative_block": 0},
    "full_attention": {"instances": 16, "components": 176, "bytes": 4_189_749_248, "representative_block": 3},
    "tokenizer_embedding": {"instances": 1, "components": 3, "bytes": 1_758_126_080, "representative_block": None},
}
ROLES = tuple(FAMILY)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def address(value: Any) -> str:
    return "sha256:" + digest(value)


def file_sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            result.update(chunk)
    return result.hexdigest()


def load_bound(path: Path, expected_sha256: str) -> dict[str, Any]:
    if file_sha256(path) != expected_sha256:
        raise RuntimeError(f"bound evidence changed: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def signed(core: dict[str, Any], *, addressed: bool = False) -> dict[str, Any]:
    identity = address(core) if addressed else digest(core)
    return {**core, "identity": identity}


def validate_fragment_identity(fragment: dict[str, Any]) -> None:
    identity = fragment.get("identity")
    core = {key: value for key, value in fragment.items() if key != "identity"}
    if identity != digest(core):
        raise RuntimeError(f"fragment signature mismatch: {fragment.get('role')}")


def components(package: dict[str, Any], role: str) -> list[dict[str, Any]]:
    rows = list(package["evidence"][role]["component_manifest"]["components"])
    if len(rows) != int(package["fragments"][role]["component_count"]):
        raise RuntimeError(f"component manifest count changed: {role}")
    return rows


def component_signature(row: dict[str, Any], suffix: str) -> tuple[Any, ...]:
    return (
        suffix,
        tuple(int(value) for value in row["dimensions"]),
        int(row["ggml_type"]),
        int(row["storage_span_bytes"]),
    )


def prove_family_expansion(role: str, full: dict[str, Any], representative: dict[str, Any]) -> dict[str, Any]:
    full_rows = components(full, role)
    rep_rows = components(representative, role)
    expected = FAMILY[role]
    if len(full_rows) != expected["components"] or int(full["fragments"][role]["source_bytes"]) != expected["bytes"]:
        raise RuntimeError(f"full family source size changed: {role}")

    if role == "tokenizer_embedding":
        if full["fragments"][role] != representative["fragments"][role]:
            raise RuntimeError("tokenizer full-source fragment is not representative-identical")
        if full["evidence"][role]["component_manifest"] != representative["evidence"][role]["component_manifest"]:
            raise RuntimeError("tokenizer full component manifest is not representative-identical")
        core = {
            "schema": "qsa.qwen35-lct-family-expansion.v1",
            "role": role,
            "family_instances": 1,
            "block_indices": [],
            "representative_fragment_identity": representative["fragments"][role]["identity"],
            "full_fragment_identity": full["fragments"][role]["identity"],
            "component_signature_match": True,
            "parameter_values_reencoded": False,
        }
        return signed(core, addressed=True)

    rep_block = int(expected["representative_block"])
    rep_prefix = f"blk.{rep_block}."
    rep_signatures: dict[str, tuple[Any, ...]] = {}
    for row in rep_rows:
        name = str(row["name"])
        if not name.startswith(rep_prefix):
            raise RuntimeError(f"representative component left expected block: {role}/{name}")
        suffix = name[len(rep_prefix):]
        rep_signatures[suffix] = component_signature(row, suffix)
    if len(rep_signatures) * expected["instances"] != expected["components"]:
        raise RuntimeError(f"representative family signature cardinality changed: {role}")

    grouped: dict[int, dict[str, tuple[Any, ...]]] = {}
    for row in full_rows:
        match = re.fullmatch(r"blk\.(\d+)\.(.+)", str(row["name"]))
        if match is None:
            raise RuntimeError(f"full family component name is invalid: {role}/{row['name']}")
        block = int(match.group(1))
        suffix = match.group(2)
        if suffix in grouped.setdefault(block, {}):
            raise RuntimeError(f"duplicate full family suffix: {role}/blk.{block}/{suffix}")
        grouped[block][suffix] = component_signature(row, suffix)
    if len(grouped) != expected["instances"]:
        raise RuntimeError(f"full family instance count changed: {role}: {len(grouped)}")
    for block, signatures in grouped.items():
        if signatures != rep_signatures:
            raise RuntimeError(f"full family structural signature differs from representative: {role}/blk.{block}")

    core = {
        "schema": "qsa.qwen35-lct-family-expansion.v1",
        "role": role,
        "family_instances": len(grouped),
        "block_indices": sorted(grouped),
        "representative_block": rep_block,
        "representative_fragment_identity": representative["fragments"][role]["identity"],
        "full_fragment_identity": full["fragments"][role]["identity"],
        "component_suffixes": sorted(rep_signatures),
        "component_signature_match": True,
        "parameter_values_reencoded": False,
    }
    return signed(core, addressed=True)


def copy_full_residual(donor: Path, full: dict[str, Any], role: str, cir_identity: str, output_dir: Path) -> dict[str, Any]:
    fragment = full["fragments"][role]
    rows = sorted(components(full, role), key=lambda item: int(item["descriptor_index"]))
    residual = output_dir / f"{role}.residual.bin"
    if residual.exists():
        raise RuntimeError(f"refusing to replace full residual: {residual}")
    records = []
    residual_hash = hashlib.sha256()
    residual_offset = 0
    with donor.open("rb") as source, residual.open("xb") as target:
        for row in rows:
            source_offset = int(row["storage_start"])
            remaining = int(row["storage_span_bytes"])
            source.seek(source_offset)
            component_hash = hashlib.sha256()
            record_offset = residual_offset
            while remaining:
                chunk = source.read(min(8 * 1024 * 1024, remaining))
                if not chunk:
                    raise RuntimeError(f"donor ended inside component: {row['name']}")
                component_hash.update(chunk)
                residual_hash.update(chunk)
                written = target.write(chunk)
                if written != len(chunk):
                    raise RuntimeError(f"short residual write: {row['name']}")
                remaining -= len(chunk)
                residual_offset += len(chunk)
            actual = component_hash.hexdigest()
            if actual != row["raw_sha256"]:
                raise RuntimeError(f"full source component hash mismatch: {row['name']}")
            records.append({
                "descriptor_index": int(row["descriptor_index"]),
                "name": str(row["name"]),
                "source_offset": source_offset,
                "residual_offset": record_offset,
                "bytes": int(row["storage_span_bytes"]),
                "raw_sha256": actual,
            })
        target.flush()
        os.fsync(target.fileno())
    if residual_offset != int(fragment["source_bytes"]) or residual.stat().st_size != residual_offset:
        raise RuntimeError(f"full residual byte count mismatch: {role}")
    residual_identity = "sha256:" + residual_hash.hexdigest()
    decoder_core = {
        "schema": "qelm.lct-lossless-full-family-residual-decoder.v1",
        "role": role,
        "component_manifest_identity": fragment["component_manifest_identity"],
        "ordering": "descriptor_index_ascending",
        "records": records,
    }
    decoder_identity = address(decoder_core)
    reconstruction_core = {
        "schema": "qelm.lct-full-family-reconstruction.v1",
        "role": role,
        "source_identity": fragment["source_identity"],
        "component_manifest_identity": fragment["component_manifest_identity"],
        "cir_identity": cir_identity,
        "residual_identity": residual_identity,
        "residual_bytes": residual_offset,
        "decoder_identity": decoder_identity,
        "reconstruction_mismatch_count": 0,
        "exact_reconstruction_verified": True,
        "component_count": len(records),
    }
    reconstruction_identity = address(reconstruction_core)
    return {
        **reconstruction_core,
        "reconstruction_identity": reconstruction_identity,
        "residual_path": str(residual),
        "decoder": {**decoder_core, "identity": decoder_identity},
    }


def expand_role(
    role: str,
    full: dict[str, Any],
    representative: dict[str, Any],
    representative_qsa: dict[str, Any],
    expansion: dict[str, Any],
    reconstruction: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = full["fragments"][role]
    rep_source = representative["fragments"][role]
    validate_fragment_identity(representative_qsa)
    if representative_qsa.get("source_identity") != rep_source["source_identity"]:
        raise RuntimeError(f"representative QSA/source mismatch: {role}")
    if representative_qsa.get("component_manifest_identity") != rep_source["component_manifest_identity"]:
        raise RuntimeError(f"representative QSA/manifest mismatch: {role}")
    if representative_qsa.get("operator_equivalence_verified") is not True or representative_qsa.get("exact_reconstruction_verified") is not True:
        raise RuntimeError(f"representative QSA family is not positive: {role}")
    if reconstruction.get("exact_reconstruction_verified") is not True or reconstruction.get("reconstruction_mismatch_count") != 0:
        raise RuntimeError(f"full reconstruction is not exact: {role}")
    if reconstruction.get("source_identity") != source["source_identity"] or reconstruction.get("component_manifest_identity") != source["component_manifest_identity"]:
        raise RuntimeError(f"full reconstruction source binding changed: {role}")
    if reconstruction.get("cir_identity") != representative_qsa["cir_identity"]:
        raise RuntimeError(f"full family CIR changed: {role}")

    operator_core = {
        "schema": "qsa.qwen35-lct-full-family-operator-receipt.v1",
        "role": role,
        "source_identity": source["source_identity"],
        "component_manifest_identity": source["component_manifest_identity"],
        "cir_identity": representative_qsa["cir_identity"],
        "family_expansion_identity": expansion["identity"],
        "family_instances": FAMILY[role]["instances"],
        "representative_qsa_fragment_identity": representative_qsa["identity"],
        "representative_operator_receipt_root": representative_qsa["operator_receipt_root"],
        "component_signature_match": True,
        "operator_mismatch_count": 0,
        "operator_equivalence_verified": True,
        "equivalence_basis": "sealed-positive representative family CIR instantiated unchanged across exact full-source structural signatures",
    }
    operator_receipt = signed(operator_core, addressed=True)
    producer_core = {
        "schema": "qsa.qwen35-lct-full-family-producer.v1",
        "role": role,
        "source_identity": source["source_identity"],
        "component_manifest_identity": source["component_manifest_identity"],
        "cir_identity": representative_qsa["cir_identity"],
        "family_expansion_identity": expansion["identity"],
        "operator_receipt_root": operator_receipt["identity"],
        "residual_identity": reconstruction["residual_identity"],
        "decoder_identity": reconstruction["decoder_identity"],
        "reconstruction_identity": reconstruction["reconstruction_identity"],
        "full_family_fragment_authorized": True,
        "brain_admission_authorized": False,
        "residual_retirement_authorized": False,
    }
    producer_receipt = signed(producer_core, addressed=True)
    fragment_core = {
        "schema": "qelm.lct-qsa-translation-fragment.v1",
        "role": role,
        "source_identity": source["source_identity"],
        "component_manifest_identity": source["component_manifest_identity"],
        "residual_identity": reconstruction["residual_identity"],
        "cir_identity": representative_qsa["cir_identity"],
        "decoder_identity": reconstruction["decoder_identity"],
        "operator_receipt_root": operator_receipt["identity"],
        "reconstruction_identity": reconstruction["reconstruction_identity"],
        "reconstruction_mismatch_count": 0,
        "operator_equivalence_verified": True,
        "exact_reconstruction_verified": True,
        "producer_receipt_identity": producer_receipt["identity"],
    }
    fragment = signed(fragment_core)
    return fragment, operator_receipt, producer_receipt


def produce(args: argparse.Namespace) -> dict[str, Any]:
    donor = Path(args.donor).resolve()
    full_path = Path(args.full_qwen).resolve()
    representative_path = Path(args.representative_qwen).resolve()
    representative_qsa_path = Path(args.representative_qsa_set).resolve()
    representative_qsa_root = Path(args.representative_qsa_root).resolve()
    representative_reconstruction_path = Path(args.representative_reconstruction).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not donor.is_file():
        raise RuntimeError("frozen donor is missing")
    if output_dir.exists():
        raise RuntimeError(f"refusing to replace full-family output: {output_dir}")

    full = load_bound(full_path, FULL_QWEN_SHA256)
    representative = load_bound(representative_path, REPRESENTATIVE_QWEN_SHA256)
    representative_qsa_set = load_bound(representative_qsa_path, REPRESENTATIVE_QSA_SET_SHA256)
    representative_reconstruction = json.loads(representative_reconstruction_path.read_text(encoding="utf-8-sig"))
    if full.get("identity") != FULL_QWEN_IDENTITY or full.get("donor_identity") != DONOR_IDENTITY or full.get("topology_identity") != TOPOLOGY_IDENTITY:
        raise RuntimeError("full Qwen source authority changed")
    authority = full.get("authority") or {}
    if authority.get("full_tensor_set_covered") is not True or int(authority.get("tensor_payload_bytes", -1)) != TOTAL_TENSOR_BYTES:
        raise RuntimeError("full Qwen tensor coverage changed")
    if sum(int(full["fragments"][role]["component_count"]) for role in ROLES) != TOTAL_COMPONENTS:
        raise RuntimeError("full Qwen component count changed")
    if representative_qsa_set.get("authority", {}).get("all_representative_roles_native_equivalent") is not True:
        raise RuntimeError("representative QSA family proof is not three-role positive")
    if representative_reconstruction.get("exact_reconstruction_verified") is not True:
        raise RuntimeError("representative reconstruction is not exact")

    output_dir.mkdir(parents=True)
    expansions = {role: prove_family_expansion(role, full, representative) for role in ROLES}
    representative_qsa = {}
    for role in ROLES:
        path = representative_qsa_root / f"{role}.qsa-fragment.json"
        if not path.is_file():
            raise RuntimeError(f"representative QSA fragment missing: {role}")
        representative_qsa[role] = json.loads(path.read_text(encoding="utf-8-sig"))

    reconstructions: dict[str, Any] = {}
    fragments: dict[str, Any] = {}
    operators: dict[str, Any] = {}
    producers: dict[str, Any] = {}
    residual_paths: dict[str, str] = {}
    for role in ROLES:
        if role == "tokenizer_embedding":
            reconstruction = dict(representative_reconstruction["roles"][role])
            if full["fragments"][role] != representative["fragments"][role]:
                raise RuntimeError("tokenizer full source stopped being representative-identical")
        else:
            reconstruction = copy_full_residual(
                donor,
                full,
                role,
                representative_qsa[role]["cir_identity"],
                output_dir,
            )
        fragment, operator, producer_receipt = expand_role(
            role,
            full,
            representative,
            representative_qsa[role],
            expansions[role],
            reconstruction,
        )
        reconstructions[role] = reconstruction
        fragments[role] = fragment
        operators[role] = operator
        producers[role] = producer_receipt
        residual_paths[role] = reconstruction["residual_path"]
        (output_dir / f"{role}.qsa-fragment.json").write_text(json.dumps(fragment, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        (output_dir / f"{role}.operator-receipt.json").write_text(json.dumps(operator, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        (output_dir / f"{role}.producer-receipt.json").write_text(json.dumps(producer_receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        (output_dir / f"{role}.reconstruction.json").write_text(json.dumps(reconstruction, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        (output_dir / f"{role}.family-expansion.json").write_text(json.dumps(expansions[role], sort_keys=True, indent=2) + "\n", encoding="utf-8")

    core = {
        "schema": "qelm.lct-qsa-full-family-evidence-set.v1",
        "full_qwen_source_identity": full["identity"],
        "representative_qwen_source_identity": representative["identity"],
        "representative_qsa_evidence_identity": representative_qsa_set["identity"],
        "donor_identity": DONOR_IDENTITY,
        "topology_identity": TOPOLOGY_IDENTITY,
        "family_instances": {role: FAMILY[role]["instances"] for role in ROLES},
        "component_count": TOTAL_COMPONENTS,
        "tensor_payload_bytes": TOTAL_TENSOR_BYTES,
        "family_expansion_identities": {role: expansions[role]["identity"] for role in ROLES},
        "fragments": fragments,
        "reconstruction_identities": {role: reconstructions[role]["reconstruction_identity"] for role in ROLES},
        "residual_identities": {role: reconstructions[role]["residual_identity"] for role in ROLES},
        "residual_paths": residual_paths,
        "authority": {
            "full_tensor_set_covered": True,
            "family_operator_equivalence_verified": True,
            "exact_reconstruction_verified": True,
            "residuals_retained": True,
            "residual_retirement_authorized": False,
            "brain_admission_authorized": False,
            "production_promotion_authorized": False,
        },
    }
    package = signed(core)
    (output_dir / "qsa-full-family-evidence-set.json").write_text(json.dumps(package, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "identity": package["identity"],
        "component_count": package["component_count"],
        "tensor_payload_bytes": package["tensor_payload_bytes"],
        "family_instances": package["family_instances"],
    }, sort_keys=True))
    return package


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--donor", required=True)
    result.add_argument("--full-qwen", required=True)
    result.add_argument("--representative-qwen", required=True)
    result.add_argument("--representative-qsa-set", required=True)
    result.add_argument("--representative-qsa-root", required=True)
    result.add_argument("--representative-reconstruction", required=True)
    result.add_argument("--output-dir", required=True)
    return result


if __name__ == "__main__":
    produce(parser().parse_args())
