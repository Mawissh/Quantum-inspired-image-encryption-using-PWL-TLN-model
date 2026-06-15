"""Run and verify the C++ EXORCISM-style backend.

This runner uses the C++ implementation cloned from:

    https://github.com/boschmitt/exorcism

The local copy has been patched to build with the current compiler, to respect
single-output PLA rows with output 0, and to include a distance-4 ExorLink pass.
The backend remains single-output, so this wrapper splits multi-output PLAs into
independent single-output ESOP runs and verifies each minimized cover by
truth-table equivalence for computationally feasible input sizes.

Run:
    python exorcism4_cpp_runner.py --self-test
    python exorcism4_cpp_runner.py --demo-paper
    python exorcism4_cpp_runner.py --pla input.pla --out output.pla
"""

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


def build_backend() -> None:
    subprocess.run(["make", "CMAKE_BUILD_TYPE=Release"], cwd=CPP_DIR, check=True)


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


def backend_status() -> dict:
    original_cmd = original_exorcism_command()
    return {
        "patched_cpp_backend": str(CPP_BIN),
        "original_exorcism_exe": str(ORIGINAL_EXE) if ORIGINAL_EXE.exists() else None,
        "original_exorcism_runnable": original_cmd is not None,
        "original_exorcism_command": original_cmd,
    }


def cover_stats(cover: ESOPCover) -> tuple[int, int]:
    return cover.n_cubes(), cover.literal_count()


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


def run_cpp_single_output(
    cover: ESOPCover,
    timeout: float = 30.0,
    verbose: bool = False,
) -> tuple[ESOPCover, str, float]:
    if not CPP_BIN.exists():
        build_backend()

    with tempfile.TemporaryDirectory(prefix="exorcism4_cpp_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        in_path = tmpdir_path / "input.pla"
        out_path = tmpdir_path / "output.pla"
        in_path.write_text(write_esop_pla([cover]))

        cmd = [str(CPP_BIN)]
        if verbose:
            cmd.append("-v")
        cmd.extend([str(in_path), str(out_path)])

        start = time.perf_counter()
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=timeout, check=False)
        elapsed = time.perf_counter() - start
        output = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0:
            raise RuntimeError(f"C++ EXORCISM backend failed with code {proc.returncode}:\n{output}")
        covers, _ = parse_pla_text(out_path.read_text())
        if len(covers) != 1:
            raise AssertionError("C++ backend returned more than one output cover.")
        minimized = covers[0]
        if cover.n_vars <= 16 and not cover.equivalent(minimized):
            raise AssertionError("C++ backend changed the represented Boolean function.")
        return minimized, output, elapsed


def minimize_pla_with_cpp(
    input_path: str | Path,
    output_path: str | Path | None = None,
    timeout: float = 30.0,
    max_truth_vars: int = 16,
    verbose: bool = False,
    exact_original: bool = False,
    quality: int = 0,
) -> list[ESOPCover]:
    covers, metadata = parse_pla(input_path, max_truth_vars=max_truth_vars)
    print(f"Read PLA: {metadata}")
    minimized_covers: list[ESOPCover] = []
    for index, cover in enumerate(covers):
        before = cover_stats(cover)
        if exact_original:
            minimized, backend_output, elapsed = run_original_single_output(
                cover,
                quality=quality,
                verbosity=2 if verbose else 0,
                timeout=timeout,
            )
        else:
            minimized, backend_output, elapsed = run_cpp_single_output(cover, timeout=timeout, verbose=verbose)
        after = cover_stats(minimized)
        print(
            f"output[{index}] cubes {before[0]}->{after[0]} "
            f"lits {before[1]}->{after[1]} time={elapsed:.4f}s"
        )
        if verbose and backend_output.strip():
            print(backend_output.rstrip())
        minimized_covers.append(minimized)

    if output_path is not None:
        Path(output_path).write_text(write_esop_pla(minimized_covers))
        print(f"Wrote minimized PLA: {output_path}")
    return minimized_covers


def paper_style_example_cover() -> ESOPCover:
    return ESOPCover.from_patterns(["0-0-", "0100", "0001", "1-0-", "11--"])


def run_self_test(timeout: float = 30.0) -> dict:
    build_backend()

    start_cover = paper_style_example_cover()
    minimized, _, elapsed = run_cpp_single_output(start_cover, timeout=timeout, verbose=True)
    assert start_cover.equivalent(minimized)
    assert minimized.n_cubes() == 3

    zero_row_pla = """
.i 2
.o 1
.p 2
00 0
11 1
.e
"""
    covers, _ = parse_pla_text(zero_row_pla)
    minimized_zero, _, _ = run_cpp_single_output(covers[0], timeout=timeout)
    assert covers[0].equivalent(minimized_zero)
    assert minimized_zero.truth_table() == (0, 0, 0, 1)

    return {
        "backend_status": backend_status(),
        "backend": str(CPP_BIN),
        "paper_style_start_cubes": start_cover.n_cubes(),
        "paper_style_result_cubes": minimized.n_cubes(),
        "paper_style_seconds": elapsed,
        "zero_output_rows_respected": True,
    }


def run_demo_paper(timeout: float = 5.0) -> dict:
    print("C++ EXORCISM backend verification")
    print(f"backend: {CPP_BIN}")
    print(f"original EXORCISM Ver.4.7 executable: {ORIGINAL_EXE if ORIGINAL_EXE.exists() else 'not recovered'}")
    print(f"original executable runnable through Wine: {original_exorcism_command() is not None}")
    print("Local patch: build fix, output-row filtering, distance-4 ExorLink pass.")
    print()

    result = {"self_test": run_self_test(timeout=max(timeout, 5.0))}
    print("Self-test passed:", result["self_test"])
    print()

    for bench_name in ("bench00.pla", "bench01.pla"):
        bench_path = CPP_DIR / "benchmarks" / bench_name
        print(f"Trying bundled benchmark {bench_name} with timeout={timeout:.1f}s")
        try:
            covers, _ = parse_pla(bench_path)
            before = cover_stats(covers[0])
            minimized, _, elapsed = run_cpp_single_output(covers[0], timeout=timeout)
            after = cover_stats(minimized)
            print(f"  completed: cubes {before[0]}->{after[0]} lits {before[1]}->{after[1]} time={elapsed:.4f}s")
            result[bench_name] = {"status": "completed", "before": before, "after": after, "seconds": elapsed}
        except subprocess.TimeoutExpired:
            print("  timed out: not computationally practical with the current exact distance-4 Python-session patch.")
            result[bench_name] = {"status": "timeout", "timeout": timeout}

    print()
    print("Paper Table 2/3/4 exact reproduction still requires original benchmark PLAs and comparator tools.")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="Build the C++ backend.")
    parser.add_argument("--self-test", action="store_true", help="Build and run local backend self-tests.")
    parser.add_argument("--demo-paper", action="store_true", help="Run feasible paper-style backend experiments.")
    parser.add_argument("--pla", help="Input PLA file.")
    parser.add_argument("--out", help="Output minimized ESOP PLA file.")
    parser.add_argument(
        "--exact-original",
        action="store_true",
        help="Use the recovered original EXORCISM Ver.4.7 Windows executable through Wine. Fails if Wine is unavailable.",
    )
    parser.add_argument("--quality", type=int, default=0, help="Original EXORCISM -q value when --exact-original is used.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-output C++ backend timeout in seconds.")
    parser.add_argument("--max-truth-vars", type=int, default=16, help="Exact SOP-to-ESOP truth-table conversion limit.")
    parser.add_argument("--verbose", action="store_true", help="Show C++ backend verbose output.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    did_something = False

    if args.build:
        build_backend()
        print(f"Built backend: {CPP_BIN}")
        did_something = True

    if args.self_test:
        print("Self-test passed:", run_self_test(timeout=args.timeout))
        did_something = True

    if args.demo_paper:
        run_demo_paper(timeout=args.timeout)
        did_something = True

    if args.pla:
        minimize_pla_with_cpp(
            args.pla,
            output_path=args.out,
            timeout=args.timeout,
            max_truth_vars=args.max_truth_vars,
            verbose=args.verbose,
            exact_original=args.exact_original,
            quality=args.quality,
        )
        did_something = True

    if not did_something:
        print("Nothing to do. Use --build, --self-test, --demo-paper, or --pla input.pla.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
