#!/usr/bin/env python3
"""Generate the paper-framework quantum figure panels and Markdown tables."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate

import quantum_image_encryption as qie


QISKIT_CIRCUIT_FOLD = 25
MAX_CIRCUIT_PART_HEIGHT = 4200
MIN_CIRCUIT_PART_HEIGHT = 1200
MIN_SPLIT_GAP_HEIGHT = 95


MATRIX_OUTPUTS = [
    ("01_input_matrix.png", "Input 4x4 matrix"),
    ("03_neqr_decoded_matrix.png", "NEQR decoded matrix"),
    ("05_qat_scrambled_matrix.png", "After improved Arnold MADD scrambling h=2"),
    ("08_csgc_output_matrix.png", "After CSGC pre-diffusion"),
    ("10_tln_diffused_matrix.png", "After TLN diffusion"),
    ("12_sbox_substituted_matrix.png", "After q=8 S-box substitution"),
    ("14_appendix_toy_sbox_matrix.png", "Appendix toy S-box output matrix"),
]

DIRECT_CIRCUIT_OUTPUTS = [
    ("02_neqr_circuit.png", "neqr_4x4", "NEQR circuit"),
    ("07_csgc_circuit.png", "csgc_stage", "CSGC pre-diffusion circuit"),
    ("09_tln_diffusion_circuit.png", "tln_diffusion_stage", "TLN-keyed diffusion circuit"),
    ("13_appendix_toy_sbox_circuit.png", "appendix_toy_sbox", "Appendix-style 4x4 toy S-box circuit"),
]


def render_matrix_png(matrix, title: str, outpath: Path) -> None:
    rows = [[str(int(value)) for value in row] for row in matrix]
    fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=180)
    ax.axis("off")
    ax.set_title(title, fontsize=10, pad=10)
    table = ax.table(cellText=rows, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.15, 1.3)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def render_text_png(text: str, title: str, outpath: Path) -> None:
    lines = text.splitlines()
    width = min(max(max((len(line) for line in lines), default=80) * 0.09, 8), 24)
    height = min(max(len(lines) * 0.22 + 1.0, 3), 18)
    fig, ax = plt.subplots(figsize=(width, height), dpi=180)
    ax.axis("off")
    ax.set_title(title, fontsize=10, pad=8)
    ax.text(
        0.01,
        0.98,
        text,
        va="top",
        ha="left",
        family="monospace",
        fontsize=6,
        transform=ax.transAxes,
    )
    fig.savefig(outpath, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def render_table_png(rows, columns, title: str, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, max(1.8, 0.42 * (len(rows) + 1))), dpi=180)
    ax.axis("off")
    ax.set_title(title, fontsize=10, pad=10)
    table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.2)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _blank_row_flags(img: Image.Image) -> list[bool]:
    width, height = img.size
    step = max(1, width // 800)
    dark_threshold = max(8, width // step // 100)
    pixels = img.load()
    flags: list[bool] = []
    for y in range(height):
        dark = 0
        for x in range(0, width, step):
            r, g, b = pixels[x, y]
            if min(r, g, b) < 245:
                dark += 1
                if dark > dark_threshold:
                    break
        flags.append(dark <= dark_threshold)
    return flags


def _split_positions_from_blank_gaps(flags: list[bool], height: int) -> list[int]:
    positions: list[int] = []
    start: int | None = None
    for index, is_blank in enumerate(flags):
        if is_blank and start is None:
            start = index
        elif not is_blank and start is not None:
            if index - start >= MIN_SPLIT_GAP_HEIGHT and start > 0 and index < height:
                positions.append((start + index) // 2)
            start = None
    return positions


def split_qiskit_folded_png(source: Path, outdir: Path, stem: str) -> list[Path]:
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(source) as img:
        img.load()
        img = img.convert("RGB")
        width, height = img.size
        if height <= MAX_CIRCUIT_PART_HEIGHT:
            target = outdir / f"{stem}.png"
            img.save(target)
            return [target]

        split_positions = _split_positions_from_blank_gaps(_blank_row_flags(img), height)
        if not split_positions:
            target = outdir / f"{stem}.png"
            img.save(target)
            return [target]

        boundaries = [0]
        current = 0
        while current + MAX_CIRCUIT_PART_HEIGHT < height:
            target = current + MAX_CIRCUIT_PART_HEIGHT
            before_target = [
                pos
                for pos in split_positions
                if current + MIN_CIRCUIT_PART_HEIGHT <= pos <= target
            ]
            if before_target:
                split_at = before_target[-1]
            else:
                after_target = [
                    pos
                    for pos in split_positions
                    if pos > current + MIN_CIRCUIT_PART_HEIGHT
                ]
                if not after_target:
                    break
                split_at = after_target[0]
            if split_at <= current:
                break
            boundaries.append(split_at)
            current = split_at
        boundaries.append(height)

        paths: list[Path] = []
        total = len(boundaries) - 1
        for index, (top, bottom) in enumerate(zip(boundaries, boundaries[1:]), start=1):
            crop = img.crop((0, top, width, bottom))
            target = outdir / f"{stem}_part_{index:02d}.png" if total > 1 else outdir / f"{stem}.png"
            crop.save(target)
            paths.append(target)
        return paths


def copy_circuit_png(
    raw_dir: Path,
    artifact_name: str,
    title: str,
    outpath: Path,
    *,
    split_circuit_images: bool,
) -> list[Path]:
    source_png = raw_dir / "circuits" / f"{artifact_name}.png"
    if source_png.exists() and source_png.stat().st_size:
        if split_circuit_images:
            return split_qiskit_folded_png(source_png, outpath.parent, outpath.stem)
        shutil.copyfile(source_png, outpath)
        return [outpath]

    source_txt = raw_dir / "circuits" / f"{artifact_name}.txt"
    if not source_txt.exists():
        raise FileNotFoundError(f"missing circuit artifact: {artifact_name}")
    render_text_png(source_txt.read_text(encoding="utf-8"), title, outpath)
    return [outpath]


def render_qiskit_circuit(qc, outpath: Path, *, split_circuit_images: bool) -> list[Path]:
    with tempfile.TemporaryDirectory(prefix=f"{outpath.stem}_") as tmp:
        source = Path(tmp) / outpath.name
        qc.draw(output="mpl", filename=str(source), fold=QISKIT_CIRCUIT_FOLD)
        plt.close("all")
        if split_circuit_images:
            return split_qiskit_folded_png(source, outpath.parent, outpath.stem)
        shutil.copyfile(source, outpath)
        return [outpath]


def render_qat_4x4_decomposed_pages(args: argparse.Namespace, outdir: Path) -> list[Path]:
    image = qie.PAPER_FRAMEWORK_INPUT_4X4.copy()
    neqr_qc, neqr_info = qie.build_neqr_circuit(image, q=8, measure=False, name="NEQR_4x4")
    qat_qc = qie.append_igqat_to_neqr(
        neqr_qc,
        neqr_info,
        a=args.arnold_r,
        b=args.arnold_z,
        iterations=args.arnold_iterations,
        gate_level=False,
    )
    decomposed = qat_qc.decompose(gates_to_decompose=["unitary"], reps=1)
    return render_qiskit_circuit(
        decomposed,
        outdir / "04_qat_4x4_decomposed_circuit.png",
        split_circuit_images=args.split_circuit_images,
    )


def render_normal_sbox_circuit(args: argparse.Namespace, outdir: Path) -> list[Path]:
    sbox_qc, _ = qie.build_sbox_stage_circuit(
        qie.ESOP_INTENSITY_SBOX,
        synthesis="anf",
        name="SBOX_q8_stage",
    )
    return render_qiskit_circuit(
        sbox_qc,
        outdir / "11_sbox_q8_normal_circuit.png",
        split_circuit_images=args.split_circuit_images,
    )


def render_esop_optimized_circuits(args: argparse.Namespace, metrics: dict, outdir: Path) -> None:
    image = qie.PAPER_FRAMEWORK_INPUT_4X4.copy()
    key_matrix = metrics["matrices"]["TLN key matrix"]

    neqr_optimized, _ = qie.build_neqr_esop_circuit(
        image,
        q=8,
        synthesis="exorcism4",
        quality=2,
        max_dist=4,
        max_steps=10000,
        name="NEQR_4x4_ESOP_optimized",
    )
    render_qiskit_circuit(
        neqr_optimized,
        outdir / "17_neqr_esop_optimized_circuit.png",
        split_circuit_images=args.split_circuit_images,
    )

    tln_optimized, _ = qie.build_tln_diffusion_stage_circuit(
        key_matrix,
        q=8,
        use_esop_key_xor=True,
        synthesis="exorcism4",
        quality=2,
        max_dist=4,
        max_steps=10000,
        name="TLN_diffusion_ESOP_optimized",
    )
    render_qiskit_circuit(
        tln_optimized,
        outdir / "18_tln_xor_esop_optimized_circuit.png",
        split_circuit_images=args.split_circuit_images,
    )

    sbox_optimized, _ = qie.build_sbox_stage_circuit(
        qie.ESOP_INTENSITY_SBOX,
        synthesis="exorcism4",
        quality=0,
        max_dist=3,
        max_steps=10000,
        name="SBOX_q8_ESOP_optimized",
    )
    render_qiskit_circuit(
        sbox_optimized,
        outdir / "19_sbox_q8_esop_optimized_circuit.png",
        split_circuit_images=args.split_circuit_images,
    )


def append_madd_resource_schematic(qc, source, target, coeff: int, n: int) -> None:
    pg1 = Gate(f"MADD {coeff} PG1", 2, [])
    tr2 = Gate(f"MADD {coeff} TR2", 2, [])
    qc.barrier()
    for index in range(n - 1):
        qc.append(pg1, [source[index], target[(index + 1) % n]])
    for index in range(n - 2):
        qc.append(tr2, [source[index + 1], target[(index + 2) % n]])
    for index in range(5 * n - 10):
        qc.cx(source[index % n], target[(index + index // n) % n])


def render_qat_m6_decomposed_pages(args: argparse.Namespace, outdir: Path) -> list[Path]:
    n = int(args.arnold_n)
    y = QuantumRegister(n, "y")
    x = QuantumRegister(n, "x")
    qc = QuantumCircuit(y, x, name=f"IGQAT_MADD_n{n}_PG1_TR2_XOR")
    for _ in range(int(args.arnold_iterations)):
        append_madd_resource_schematic(qc, list(y), list(x), args.arnold_r, n)
        append_madd_resource_schematic(qc, list(x), list(y), args.arnold_z, n)

    return render_qiskit_circuit(
        qc,
        outdir / "06_qat_m6_madd_decomposition_schematic.png",
        split_circuit_images=args.split_circuit_images,
    )


def _md_escape(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _ops_summary(ops: dict) -> str:
    if not ops:
        return "-"
    return ", ".join(f"{key}: {value}" for key, value in sorted(ops.items()))


def write_esop_depth_markdown(metrics: dict, outpath: Path) -> None:
    lines = [
        "# Table 13-style ESOP depth comparison",
        "",
        "Circuit depths with and without reduction using EXORCISM-4 for the 4 x 4 demonstration.",
        "",
        "| Operation | With ESOP minimization | Without ESOP minimization |",
        "| --- | ---: | ---: |",
    ]
    labels = {
        "NEQR encoding": "NEQR",
        "QAT": "QAT/GQAT",
        "CSGC": "CSGC",
        "Chaotic XOR diffusion": "TLN XORing",
        "Substitution box": "S-box",
    }
    for row in metrics.get("complete_depth_comparison", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_escape(labels.get(row.get("operation", ""), row.get("operation", ""))),
                    _md_escape(row.get("with_esop_depth", "")),
                    _md_escape(row.get("without_esop_depth", "")),
                ]
            )
            + " |"
        )
    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_resource_table_markdown(metrics: dict, outpath: Path) -> None:
    lines = [
        "# Quantum resource table",
        "",
        "## m=6 MADD decomposition counts",
        "",
        "| Width | Iterations | PG1 | TR2 | XOR | T-depth | Total depth | Ancilla |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    demo_metrics = metrics.get("arnold", {}).get("demo_n_metrics") or {}
    if demo_metrics:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_escape(demo_metrics.get("width", "")),
                    _md_escape(metrics.get("arnold", {}).get("iterations", "")),
                    _md_escape(demo_metrics.get("PG1", "")),
                    _md_escape(demo_metrics.get("TR2", "")),
                    _md_escape(demo_metrics.get("XOR", "")),
                    _md_escape(demo_metrics.get("T_depth", "")),
                    _md_escape(demo_metrics.get("total_depth", "")),
                    _md_escape(demo_metrics.get("ancilla", "")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Circuit resources",
            "",
        "| Component | Qubits | Depth | Operations | Note |",
        "| --- | ---: | ---: | --- | --- |",
        ]
    )
    for row in metrics.get("resource_rows", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_escape(row.get("component", "")),
                    _md_escape(row.get("qubits", "")),
                    _md_escape(row.get("depth", "")),
                    _md_escape(_ops_summary(row.get("ops", {}))),
                    _md_escape(row.get("note", "")),
                ]
            )
            + " |"
        )
    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_metrics(args: argparse.Namespace, raw_dir: Path):
    try:
        return qie.build_paper_framework_demo(
            outdir=raw_dir,
            arnold_n=args.arnold_n,
            arnold_iterations=args.arnold_iterations,
            arnold_r=args.arnold_r,
            arnold_z=args.arnold_z,
            use_exorcism4=True,
        )
    except Exception as exc:
        print(f"EXORCISM-4 path unavailable, continuing without it: {exc}")
        return qie.build_paper_framework_demo(
            outdir=raw_dir,
            arnold_n=args.arnold_n,
            arnold_iterations=args.arnold_iterations,
            arnold_r=args.arnold_r,
            arnold_z=args.arnold_z,
            use_exorcism4=False,
        )


def generate_pngs(args: argparse.Namespace) -> list[Path]:
    outdir = Path(args.outdir).resolve()
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)

    with tempfile.TemporaryDirectory(prefix="quantum_framework_raw_") as raw_tmp:
        raw_dir = Path(raw_tmp)
        metrics = build_metrics(args, raw_dir)
        plt.close("all")

        matrices = metrics["matrices"]
        for filename, title in MATRIX_OUTPUTS:
            render_matrix_png(matrices[title], title, outdir / filename)

        for filename, artifact_name, title in DIRECT_CIRCUIT_OUTPUTS:
            copy_circuit_png(
                raw_dir,
                artifact_name,
                title,
                outdir / filename,
                split_circuit_images=args.split_circuit_images,
            )

        render_qat_4x4_decomposed_pages(args, outdir)
        render_qat_m6_decomposed_pages(args, outdir)
        render_normal_sbox_circuit(args, outdir)
        render_esop_optimized_circuits(args, metrics, outdir)
        write_esop_depth_markdown(metrics, outdir / "15_esop_depth_table.md")
        write_resource_table_markdown(metrics, outdir / "16_resource_table.md")

    return sorted(path for path in outdir.iterdir() if path.is_file())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="quantum_framework_outputs", help="output folder for PNG figures and MD tables")
    parser.add_argument("--arnold-n", type=int, default=6, help="paper-scale MADD/QAT dimension parameter")
    parser.add_argument("--arnold-iterations", type=int, default=2, help="4x4 Arnold iterations")
    parser.add_argument("--arnold-r", type=int, default=11, help="Arnold control parameter r")
    parser.add_argument("--arnold-z", type=int, default=19, help="Arnold control parameter z")
    parser.add_argument(
        "--split-circuit-images",
        action="store_true",
        help="split long folded Qiskit circuit PNGs into safe page-sized parts",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = generate_pngs(args)
    all_outputs = [path for path in Path(args.outdir).resolve().iterdir() if path.is_file()]
    unsupported = [path for path in all_outputs if path.suffix.lower() not in {".png", ".md"}]
    if unsupported:
        raise RuntimeError(f"unsupported outputs generated: {unsupported}")
    print(f"generated {len(outputs)} output files")
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
