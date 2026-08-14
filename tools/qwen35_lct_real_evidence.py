from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

PR65_HEAD = "eff0f7c64fd65b0cefce29f2510bab96130e799a"
SOURCE_SET_SHA256 = "ec3a6c03cdb1bd88dc01a2cd745e6aed6a582512543c66088eb6e1193dfab0b2"
FIT_SHA256 = "a7c66691e9a798379a39a2e49795fbe6844f9065e33bfc58604634e094a4e08a"
HOLDOUT_SHA256 = "c92a2042370c3e2db2780260762136c02a88a7ecc9f92553fb06d6d3c3816f4b"
RECONSTRUCTION_SHA256 = "041e571f4466543c06b900ab83150445ae2968bc65ef562079923dfd27e37699"
TOKENIZER_FIT_SHA256 = "30f0b8b7646409b9ae27570b03c4ed8fdd25be8f77acbef2d5a81f205a6af1e6"
TOKENIZER_HOLDOUT_SHA256 = "76600dcb15df27428ef1a043ca0d76ece07b6e82baef4d37fce6aaa88dea25d4"
FIT_IDENTITY = "7f3152032c56343513d992041e06d1a3ecb65a133d3f258fab3b4421049e818d"
HOLDOUT_IDENTITY = "436e109fb7928e05d2f066ad5181787b0793f0717e44f5de459d0e844dbbc80d"
REFERENCE_OPERATOR_ROOT = "sha256:5ed8bd9908d50b7d6cef5682ffdb7e04660e69448954c2c39054f9fde2bfa5fb"
TOKENIZER_FIT_IDENTITY = "466fbc74c9d2fe6e8e3de14a9f5ffc58a03151049d24492fcdcddba4948523d0"
TOKENIZER_HOLDOUT_IDENTITY = "aabb23a0f1572b466b39ffefcf3510903dd25768ad95ac32836cb0c48cb919b9"
TOKENIZER_OPERATOR_ROOT = "sha256:a43080cda5e4fbeef7a9d36534e0b73866423d495caab0ed4e337627d0a5f8ab"
TOKENIZER_MECHANISM_IDENTITY = "sha256:e46db70ab44031617b77c5f4c0254d9959e78a1b255d5d45c45ff5c6424facf4"
REFERENCE_CIRS = {
    "ssm_recurrent": "sha256:be6c04a161458f5c6bf15681bd543de5d57b77a2395e10503959edfe3418fa0f",
    "full_attention": "sha256:6db6bd50c2abb8aa8f1e0ca360c5c728b91cbe70afe821c91423735753728672",
    "tokenizer_embedding": "sha256:f3b244d5be92439281f9999344e6ebfbe5c30eabdd5f17f7b622f3b86027ae74",
}
ROLES = ("ssm_recurrent", "full_attention", "tokenizer_embedding")


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def address(value: Any) -> str:
    return "sha256:" + digest(value)


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def load_bound(path: Path, expected_sha256: str) -> dict[str, Any]:
    if file_sha256(path) != expected_sha256:
        raise RuntimeError(f"bound evidence changed: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def helper_output(helper: Path, *arguments: str) -> str:
    completed = subprocess.run(
        [str(helper), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def signed(core: dict[str, Any], *, addressed: bool = True) -> dict[str, Any]:
    identity = address(core) if addressed else digest(core)
    return {**core, "identity": identity}


def role_reference(
    role: str,
    fit: dict[str, Any],
    holdout: dict[str, Any],
    tokenizer_fit: dict[str, Any],
    tokenizer_holdout: dict[str, Any],
) -> dict[str, Any]:
    if role != "tokenizer_embedding":
        reference_role = holdout["roles"][role]
        if reference_role.get("cir_identity") != REFERENCE_CIRS[role]:
            raise RuntimeError(f"reference CIR changed: {role}")
        if reference_role.get("reference_operator_equivalence_verified") is not True:
            raise RuntimeError(f"reference operator is not positive: {role}")
        return {
            "fit_identity": fit["identity"],
            "holdout_identity": holdout["identity"],
            "mechanism_identity": fit["mechanism_identity"],
            "operator_receipt_root": holdout["operator_receipt_root"],
            "basis": "independently executed frozen source operator on fit-frozen sealed holdout",
        }

    tokenizer_role = holdout["roles"][role]
    if tokenizer_role.get("cir_identity") != REFERENCE_CIRS[role]:
        raise RuntimeError("tokenizer reference CIR changed")
    if tokenizer_role.get("embedding_lookup_verified") is not True:
        raise RuntimeError("tokenizer input embedding lookup is not verified")
    if tokenizer_fit.get("identity") != TOKENIZER_FIT_IDENTITY or tokenizer_fit.get("mechanism_identity") != TOKENIZER_MECHANISM_IDENTITY:
        raise RuntimeError("tokenizer CUDA fit identity changed")
    if tokenizer_fit.get("holdout_authorized") is not True or tokenizer_fit.get("authority", {}).get("holdout_numeric_values_interpreted") is not False:
        raise RuntimeError("tokenizer CUDA fit did not freeze before holdout")
    if tokenizer_holdout.get("identity") != TOKENIZER_HOLDOUT_IDENTITY or tokenizer_holdout.get("operator_receipt_root") != TOKENIZER_OPERATOR_ROOT:
        raise RuntimeError("tokenizer CUDA holdout identity changed")
    if tokenizer_holdout.get("tokenizer_output_reference_verified") is not True:
        raise RuntimeError("tokenizer CUDA output reference is not verified")
    if tokenizer_holdout.get("authority", {}).get("reference_holdout_positive") is not True or tokenizer_holdout.get("failures") != []:
        raise RuntimeError("tokenizer CUDA sealed holdout is not positive")
    if tokenizer_holdout.get("authority", {}).get("qsa_runtime_equivalence_verified") is not False:
        raise RuntimeError("tokenizer reference improperly self-authorized QSA")
    return {
        "fit_identity": tokenizer_fit["identity"],
        "holdout_identity": tokenizer_holdout["identity"],
        "mechanism_identity": tokenizer_fit["mechanism_identity"],
        "operator_receipt_root": tokenizer_holdout["operator_receipt_root"],
        "basis": "exact input embedding boundary plus source-bound result_norm and sealed CUDA Q6_K x Q8_1 MMVQ output projection",
    }


def produce(args: argparse.Namespace) -> dict[str, Any]:
    helper = Path(args.helper).resolve()
    source_path = Path(args.source_fragments).resolve()
    fit_path = Path(args.fit_receipt).resolve()
    holdout_path = Path(args.holdout_receipt).resolve()
    reconstruction_path = Path(args.reconstruction_receipt).resolve()
    tokenizer_fit_path = Path(args.tokenizer_fit_receipt).resolve()
    tokenizer_holdout_path = Path(args.tokenizer_holdout_receipt).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise RuntimeError(f"refusing to replace evidence directory: {output_dir}")
    if not helper.is_file():
        raise RuntimeError("QSA real-evidence helper is missing")

    source_set = load_bound(source_path, SOURCE_SET_SHA256)
    fit = load_bound(fit_path, FIT_SHA256)
    holdout = load_bound(holdout_path, HOLDOUT_SHA256)
    reconstruction = load_bound(reconstruction_path, RECONSTRUCTION_SHA256)
    tokenizer_fit = load_bound(tokenizer_fit_path, TOKENIZER_FIT_SHA256)
    tokenizer_holdout = load_bound(tokenizer_holdout_path, TOKENIZER_HOLDOUT_SHA256)
    if fit.get("identity") != FIT_IDENTITY or holdout.get("identity") != HOLDOUT_IDENTITY:
        raise RuntimeError("reference receipt identity changed")
    if holdout.get("operator_receipt_root") != REFERENCE_OPERATOR_ROOT:
        raise RuntimeError("reference operator root changed")
    if holdout.get("authority", {}).get("reference_holdout_positive") is not True:
        raise RuntimeError("reference holdout is not positive")
    if reconstruction.get("exact_reconstruction_verified") is not True:
        raise RuntimeError("reference reconstruction is not exact")

    output_dir.mkdir(parents=True)
    helper_sha = file_sha256(helper)
    fragments: dict[str, Any] = {}
    operator_receipts: dict[str, Any] = {}
    producer_receipts: dict[str, Any] = {}

    for role in ROLES:
        qwen = source_set["fragments"][role]
        reconstructed = reconstruction["roles"][role]
        reference = role_reference(role, fit, holdout, tokenizer_fit, tokenizer_holdout)
        qsa_cir = helper_output(helper, "cir", role)
        if qsa_cir != REFERENCE_CIRS[role]:
            raise RuntimeError(f"QSA/reference CIR mismatch: {role}: {qsa_cir} != {REFERENCE_CIRS[role]}")
        if reconstructed.get("cir_identity") != qsa_cir:
            raise RuntimeError(f"reconstruction CIR mismatch: {role}")
        if reconstructed.get("exact_reconstruction_verified") is not True or reconstructed.get("reconstruction_mismatch_count") != 0:
            raise RuntimeError(f"reference reconstruction failed: {role}")
        residual_path = Path(reconstructed["residual_path"])
        if not residual_path.is_file() or "sha256:" + file_sha256(residual_path) != reconstructed["residual_identity"]:
            raise RuntimeError(f"residual bytes changed: {role}")

        operator_core = {
            "schema": "qsa.qwen35-lct-operator-equivalence-receipt.v1",
            "role": role,
            "pr65_head": PR65_HEAD,
            "qwen_fragment_identity": qwen["identity"],
            "source_identity": qwen["source_identity"],
            "component_manifest_identity": qwen["component_manifest_identity"],
            "qsa_cir_identity": qsa_cir,
            "reference_cir_identity": REFERENCE_CIRS[role],
            "cir_identity_match": True,
            "reference_fit_identity": reference["fit_identity"],
            "reference_holdout_identity": reference["holdout_identity"],
            "reference_mechanism_identity": reference["mechanism_identity"],
            "reference_operator_receipt_root": reference["operator_receipt_root"],
            "reference_operator_equivalence_verified": True,
            "operator_mismatch_count": 0,
            "residual_identity": reconstructed["residual_identity"],
            "decoder_identity": reconstructed["decoder_identity"],
            "reconstruction_identity": reconstructed["reconstruction_identity"],
            "reconstruction_mismatch_count": 0,
            "exact_reconstruction_verified": True,
            "equivalence_basis": "accepted PR65 CIR exact match to " + reference["basis"],
            "native_qsa_numeric_executor_used": False,
        }
        operator_receipt = signed(operator_core)
        operator_root = operator_receipt["identity"]
        producer_core = {
            "schema": "qsa.qwen35-lct-real-evidence-producer.v1",
            "role": role,
            "pr65_head": PR65_HEAD,
            "helper_sha256": helper_sha,
            "qsa_cir_identity": qsa_cir,
            "operator_receipt_root": operator_root,
            "reference_operator_receipt_root": reference["operator_receipt_root"],
            "residual_identity": reconstructed["residual_identity"],
            "reconstruction_identity": reconstructed["reconstruction_identity"],
            "qsa_fragment_authorized": True,
            "brain_admission_authorized": False,
            "semantic_truth_verified": False,
            "residual_retirement_authorized": False,
        }
        producer_receipt = signed(producer_core)
        fragment_text = helper_output(
            helper,
            "finalize",
            role,
            reconstructed["residual_identity"],
            reconstructed["decoder_identity"],
            operator_root,
            reconstructed["reconstruction_identity"],
            producer_receipt["identity"],
        )
        fragment = json.loads(fragment_text)
        fragment_core = {key: value for key, value in fragment.items() if key != "identity"}
        if fragment.get("identity") != digest(fragment_core):
            raise RuntimeError(f"QSA fragment signature mismatch: {role}")
        if fragment.get("cir_identity") != qsa_cir or fragment.get("operator_equivalence_verified") is not True:
            raise RuntimeError(f"QSA fragment lost equivalence binding: {role}")
        if fragment.get("exact_reconstruction_verified") is not True or fragment.get("reconstruction_mismatch_count") != 0:
            raise RuntimeError(f"QSA fragment lost reconstruction gate: {role}")

        operator_receipts[role] = operator_receipt
        producer_receipts[role] = producer_receipt
        fragments[role] = fragment
        (output_dir / f"{role}.operator-receipt.json").write_text(json.dumps(operator_receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        (output_dir / f"{role}.producer-receipt.json").write_text(json.dumps(producer_receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        (output_dir / f"{role}.qsa-fragment.json").write_text(json.dumps(fragment, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    package_core = {
        "schema": "qelm.lct-qsa-representative-evidence-set.v1",
        "pr65_head": PR65_HEAD,
        "source_fragment_set_identity": source_set["identity"],
        "fit_identity": fit["identity"],
        "holdout_identity": holdout["identity"],
        "reference_operator_receipt_root": holdout["operator_receipt_root"],
        "tokenizer_fit_identity": tokenizer_fit["identity"],
        "tokenizer_holdout_identity": tokenizer_holdout["identity"],
        "tokenizer_reference_operator_receipt_root": tokenizer_holdout["operator_receipt_root"],
        "reconstruction_identity": reconstruction["identity"],
        "native_equivalent_roles": list(ROLES),
        "residual_only_roles": [],
        "fragments": fragments,
        "operator_receipt_identities": {role: operator_receipts[role]["identity"] for role in ROLES},
        "producer_receipt_identities": {role: producer_receipts[role]["identity"] for role in ROLES},
        "authority": {
            "representative_only": True,
            "qsa_operator_equivalence_verified_for_native_roles": True,
            "all_representative_roles_native_equivalent": True,
            "brain_admission_authorized": False,
            "semantic_truth_verified": False,
            "residual_retirement_authorized": False,
        },
    }
    package = signed(package_core, addressed=False)
    (output_dir / "qsa-representative-evidence-set.json").write_text(json.dumps(package, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return package


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--helper", required=True)
    result.add_argument("--source-fragments", required=True)
    result.add_argument("--fit-receipt", required=True)
    result.add_argument("--holdout-receipt", required=True)
    result.add_argument("--reconstruction-receipt", required=True)
    result.add_argument("--tokenizer-fit-receipt", required=True)
    result.add_argument("--tokenizer-holdout-receipt", required=True)
    result.add_argument("--output-dir", required=True)
    return result


if __name__ == "__main__":
    package = produce(parser().parse_args())
    print(json.dumps({
        "identity": package["identity"],
        "native_equivalent_roles": package["native_equivalent_roles"],
        "residual_only_roles": package["residual_only_roles"],
    }, sort_keys=True))
