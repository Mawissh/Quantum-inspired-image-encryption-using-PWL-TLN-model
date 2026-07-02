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




