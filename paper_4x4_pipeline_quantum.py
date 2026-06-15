"""Qiskit 4x4 paper-pipeline simulation.

This is the quantum-side entrypoint for the paper framework demo. It stays in
the quantum simulation folder and exports a normalized stage schema so a neutral
comparison script can compare it with the separate classical implementation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import quantum_image_encryption as qie


def normalize_stage_matrices(metrics: dict, arnold_iterations: int) -> dict:
    matrices = metrics["matrices"]
    return {
        "input": matrices["Input 4x4 matrix"],
        "neqr_decoded": matrices["NEQR decoded matrix"],
        "after_arnold_h2": matrices[f"After improved Arnold MADD scrambling h={arnold_iterations}"],
        "after_csgc": matrices["After CSGC pre-diffusion"],
        "tln_key_matrix": matrices["TLN key matrix"],
        "after_tln_diffusion": matrices["After TLN diffusion"],
        "after_sbox": matrices["After q=8 S-box substitution"],
        "appendix_toy_sbox_input": matrices["Appendix toy S-box input indices"],
        "appendix_toy_sbox_output": matrices["Appendix toy S-box output matrix"],
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "paper_4x4_output"))
    parser.add_argument("--arnold-n", type=int, default=6)
    parser.add_argument("--arnold-iterations", type=int, default=2)
    parser.add_argument("--arnold-r", type=int, default=11)
    parser.add_argument("--arnold-z", type=int, default=19)
    parser.add_argument("--use-exorcism4", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    outdir = Path(args.out)
    metrics = qie.build_paper_framework_demo(
        outdir=outdir,
        arnold_n=args.arnold_n,
        arnold_iterations=args.arnold_iterations,
        arnold_r=args.arnold_r,
        arnold_z=args.arnold_z,
        use_exorcism4=args.use_exorcism4,
    )

    stages = normalize_stage_matrices(metrics, args.arnold_iterations)
    metrics["artifact_kind"] = "quantum_4x4_paper_pipeline"
    metrics["settings"] = {
        "arnold_r": int(args.arnold_r),
        "arnold_z": int(args.arnold_z),
        "arnold_iterations": int(args.arnold_iterations),
        "arnold_demo_n": int(args.arnold_n),
        "password": "paper-framework-demo-key",
        "salt_hex": "00112233445566778899aabbccddeeff",
        "nonce_hex": "102132435465768798a9bacb",
        "diffusion_warmup": 1000,
        "diffusion_stride": 11,
        "dt": 0.005,
        "q": 8,
        "use_exorcism4": bool(args.use_exorcism4),
    }
    metrics["stage_order"] = list(stages.keys())
    metrics["pipeline_stages"] = stages
    (outdir / "metrics.json").write_text(json.dumps(qie._jsonable(metrics), indent=2, sort_keys=True))

    print("Quantum 4x4 paper pipeline completed")
    print(f"  output={outdir}")
    print(f"  metrics={outdir / 'metrics.json'}")
    print(f"  matrices={outdir / 'matrices.md'}")
    print(f"  resources={outdir / 'resource_table.md'}")
    print(f"  esop={outdir / 'esop_comparison_table.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
