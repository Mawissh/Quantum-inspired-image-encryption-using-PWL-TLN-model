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
    print("Quantum 4x4 paper pipeline completed")
    print(f"  output={outdir}")
    print(f"  metrics={outdir / 'metrics.json'}")
    print(f"  matrices={outdir / 'matrices.md'}")
    print(f"  resources={outdir / 'resource_table.md'}")
    print(f"  esop={outdir / 'esop_comparison_table.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
