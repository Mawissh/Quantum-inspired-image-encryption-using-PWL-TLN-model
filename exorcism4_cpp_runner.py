from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from exorcism4_minimizer import ESOPCover, parse_pla, parse_pla_text, write_esop_pla


ROOT = Path(__file__).resolve().parent
CPP_DIR = ROOT / "exorcism_cpp"
CPP_BIN = CPP_DIR / "build" / "exorcism"


def _original_exe_path() -> Path:
    for candidate in (
        ROOT / "exorcism4_original.exe",
        ROOT.parent.parent / "exorcism4_original.exe",
    ):
        if candidate.exists():
            return candidate
    return ROOT / "exorcism4_original.exe"


ORIGINAL_EXE = _original_exe_path()

def original_exorcism_command() -> list[str] | None:
    """Return a command prefix for the recovered EXORCISM Ver.4.7 binary.

    The original public release found on the archived Alan Mishchenko webpage is
    a 32-bit Windows console executable. It is the closest available artifact to
    the rm01_heu.pdf experiments, but it requires Wine on Linux.
    """
    if not ORIGINAL_EXE.exists():
        return None
    wine = shutil.which("wine") or shutil.which("wine64")
    if wine is None:
        return None
    return [wine, str(ORIGINAL_EXE)]

def run_original_single_output(
    cover: ESOPCover,
    quality: int = 0,
    verbosity: int = 0,
    timeout: float = 30.0,
) -> tuple[ESOPCover, str, float]:
    """Run the recovered original EXORCISM Ver.4.7 executable through Wine."""
    cmd_prefix = original_exorcism_command()
    if cmd_prefix is None:
        raise RuntimeError(
            "Exact original EXORCISM Ver.4.7 mode is unavailable: "
            f"{ORIGINAL_EXE} exists={ORIGINAL_EXE.exists()}, but Wine is not installed."
        )

    with tempfile.TemporaryDirectory(prefix="exorcism4_original_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        in_path = tmpdir_path / "input.pla"
        in_path.write_text(write_esop_pla([cover]))
        out_path = tmpdir_path / "input.esop"
        cmd = cmd_prefix + ["-q", str(int(quality)), "-v", str(int(verbosity)), str(in_path)]
        start = time.perf_counter()
        proc = subprocess.run(cmd, cwd=tmpdir_path, text=True, capture_output=True, timeout=timeout, check=False)
        elapsed = time.perf_counter() - start
        output = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0:
            raise RuntimeError(f"Original EXORCISM Ver.4.7 failed with code {proc.returncode}:\n{output}")
        if not out_path.exists():
            candidates = sorted(tmpdir_path.glob("*.esop"))
            if not candidates:
                raise RuntimeError(f"Original EXORCISM Ver.4.7 produced no .esop output:\n{output}")
            out_path = candidates[0]
        covers, _ = parse_pla_text(out_path.read_text())
        minimized = covers[0]
        if cover.n_vars <= 16 and not cover.equivalent(minimized):
            raise AssertionError("Original EXORCISM Ver.4.7 changed the represented Boolean function.")
        return minimized, output, elapsed

if __name__ == "__main__":
    raise SystemExit(main())
