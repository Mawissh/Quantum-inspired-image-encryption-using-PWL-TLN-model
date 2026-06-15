"""Standalone EXORCISM-4 ESOP minimizer.

This module implements the algorithmic pieces described in:

    A. Mishchenko and M. Perkowski, "Fast Heuristic Minimization of
    Exclusive-Sums-of-Products", rm01_heu.pdf.

It is an algorithmic implementation intended for local experiments and
Qiskit-circuit preprocessing. It follows the minimization flow and acceptance
rules in the paper; it is not a bit-for-bit port of the original C++/CUDD
EXORCISM-4 executable, so it does not claim parity with the published 2001
runtime/benchmark tables.

Implemented:
    - ternary ESOP cubes
    - PLA parsing and ESOP PLA writing
    - exact small truth-table starting-cover generation using the same
      Shannon, positive Davio, and negative Davio expansion choices as the
      Pseudo-Kronecker starting-cover algorithm
    - ExorLink distance-1 through distance-4 cube-pair transformations
    - Fig. 2 minimization flow with aggressive, last-gasp, and refinement
      passes
    - Table 1 acceptance criteria for cube transformations
    - distance-0/1 ESOP reduction after generated cubes interact with the
      current cover
    - embedded self-tests and paper-style local demonstrations

Run:
    python exorcism4_minimizer.py --self-test
    python exorcism4_minimizer.py --demo-paper --trace
    python exorcism4_minimizer.py --pla input.pla --quality 5 --out output.pla
"""

from __future__ import annotations

import argparse
import itertools
import random
import time
from dataclasses import dataclass
from pathlib import Path


TERNARY_ABSENT = None


@dataclass(frozen=True)
class PassStats:
    """One minimization attempt block in the paper-level algorithm trace."""

    stage: str
    strategy: str
    distance: int
    attempts: int
    accepted: int
    before_cubes: int
    after_cubes: int
    before_literals: int
    after_literals: int

    @property
    def cube_delta(self) -> int:
        return self.after_cubes - self.before_cubes

    @property
    def literal_delta(self) -> int:
        return self.after_literals - self.before_literals


def _all_mask(n_vars: int) -> int:
    return (1 << int(n_vars)) - 1


def _var_bit(n_vars: int, pattern_index: int) -> int:
    """Map left-to-right PLA variable index to the internal bit position."""
    return int(n_vars) - 1 - int(pattern_index)


@dataclass(frozen=True, order=True)
class Cube:
    """Ternary product cube.

    A variable can be positive, negative, or absent. The leftmost PLA character
    is stored as the highest internal bit, so pattern strings and integer truth
    table assignments line up naturally.
    """

    n_vars: int
    pos: int = 0
    neg: int = 0

    def __post_init__(self) -> None:
        if self.n_vars < 0:
            raise ValueError("n_vars must be non-negative.")
        mask = _all_mask(self.n_vars)
        if self.pos & self.neg:
            raise ValueError("A variable cannot be both positive and negative.")
        if self.pos & ~mask or self.neg & ~mask:
            raise ValueError("Cube masks exceed n_vars.")

    @classmethod
    def one(cls, n_vars: int) -> "Cube":
        return cls(n_vars=int(n_vars), pos=0, neg=0)

    @classmethod
    def from_pattern(cls, pattern: str) -> "Cube":
        pattern = pattern.strip()
        n_vars = len(pattern)
        pos = 0
        neg = 0
        for index, char in enumerate(pattern):
            bit = _var_bit(n_vars, index)
            if char == "1":
                pos |= 1 << bit
            elif char == "0":
                neg |= 1 << bit
            elif char == "-":
                continue
            else:
                raise ValueError(f"Unsupported cube character {char!r}.")
        return cls(n_vars=n_vars, pos=pos, neg=neg)

    def to_pattern(self) -> str:
        chars = []
        for index in range(self.n_vars):
            bit = _var_bit(self.n_vars, index)
            if self.pos & (1 << bit):
                chars.append("1")
            elif self.neg & (1 << bit):
                chars.append("0")
            else:
                chars.append("-")
        return "".join(chars)

    def n_lits(self) -> int:
        return int((self.pos | self.neg).bit_count())

    def value_at(self, bit: int) -> int | None:
        bit_mask = 1 << int(bit)
        if self.pos & bit_mask:
            return 1
        if self.neg & bit_mask:
            return 0
        return TERNARY_ABSENT

    def with_value(self, bit: int, value: int | None) -> "Cube":
        bit_mask = 1 << int(bit)
        pos = self.pos & ~bit_mask
        neg = self.neg & ~bit_mask
        if value == 1:
            pos |= bit_mask
        elif value == 0:
            neg |= bit_mask
        elif value is TERNARY_ABSENT:
            pass
        else:
            raise ValueError("Cube value must be 0, 1, or None.")
        return Cube(self.n_vars, pos, neg)

    def add_literal(self, bit: int, value: int) -> "Cube":
        if self.value_at(bit) is not TERNARY_ABSENT:
            raise ValueError("Cannot add a literal to an already constrained variable.")
        return self.with_value(bit, int(value))

    def matches(self, assignment: int) -> bool:
        assignment = int(assignment)
        return (assignment & self.pos) == self.pos and (assignment & self.neg) == 0

    def distance(self, other: "Cube", stop_after: int | None = None) -> int:
        self._check_compatible(other)
        distance = 0
        for bit in range(self.n_vars):
            if self.value_at(bit) != other.value_at(bit):
                distance += 1
                if stop_after is not None and distance > stop_after:
                    break
        return distance

    def differing_bits(self, other: "Cube") -> tuple[int, ...]:
        self._check_compatible(other)
        return tuple(bit for bit in range(self.n_vars) if self.value_at(bit) != other.value_at(bit))

    def _check_compatible(self, other: "Cube") -> None:
        if self.n_vars != other.n_vars:
            raise ValueError("Cube variable counts differ.")


class ESOPCover:
    """Reduced XOR cover: adding the same cube twice cancels it."""

    def __init__(self, n_vars: int, cubes: list[Cube] | tuple[Cube, ...] | None = None):
        self.n_vars = int(n_vars)
        self.cubes: set[Cube] = set()
        if cubes:
            for cube in cubes:
                self.add_cube(cube)

    @classmethod
    def from_patterns(cls, patterns: list[str] | tuple[str, ...]) -> "ESOPCover":
        if not patterns:
            raise ValueError("Need at least one pattern to infer n_vars.")
        cover = cls(len(patterns[0]))
        for pattern in patterns:
            cover.add_cube(Cube.from_pattern(pattern))
        return cover

    def copy(self) -> "ESOPCover":
        return ESOPCover(self.n_vars, tuple(self.cubes))

    def add_cube(self, cube: Cube) -> None:
        if cube.n_vars != self.n_vars:
            raise ValueError("Cube variable count does not match cover.")
        if cube in self.cubes:
            self.cubes.remove(cube)
        else:
            self.cubes.add(cube)

    def add_cubes(self, cubes) -> None:
        for cube in cubes:
            self.add_cube(cube)

    def n_cubes(self) -> int:
        return len(self.cubes)

    def literal_count(self) -> int:
        return sum(cube.n_lits() for cube in self.cubes)

    def qcost(self) -> int:
        """Simple quantum-cost proxy for reporting, not from rm01_heu.pdf."""
        return sum(1 << max(cube.n_lits() - 1, 0) for cube in self.cubes)

    def stats(self) -> tuple[int, int]:
        return self.n_cubes(), self.literal_count()

    def signature(self) -> tuple[str, ...]:
        return tuple(sorted(cube.to_pattern() for cube in self.cubes))

    def evaluate_assignment(self, assignment: int) -> int:
        value = 0
        for cube in self.cubes:
            if cube.matches(assignment):
                value ^= 1
        return value

    def truth_table(self) -> tuple[int, ...]:
        return tuple(self.evaluate_assignment(i) for i in range(1 << self.n_vars))

    def equivalent(self, other: "ESOPCover") -> bool:
        if self.n_vars != other.n_vars:
            return False
        return self.truth_table() == other.truth_table()

    def replace_pair(self, left: Cube, right: Cube, replacement: list[Cube]) -> "ESOPCover":
        out = self.copy()
        out.add_cube(left)
        out.add_cube(right)
        out.add_cubes(replacement)
        return out


def distance1_link_cube(left: Cube, right: Cube) -> Cube:
    """Return the single cube equivalent to XORing a distance-1 cube pair."""
    diff = left.differing_bits(right)
    if len(diff) != 1:
        raise ValueError("distance1_link_cube expects cubes at distance 1.")
    bit = diff[0]
    left_value = left.value_at(bit)
    right_value = right.value_at(bit)
    result = left
    if left_value is TERNARY_ABSENT or right_value is TERNARY_ABSENT:
        literal = right_value if left_value is TERNARY_ABSENT else left_value
        link_value = 1 - int(literal)
    else:
        link_value = TERNARY_ABSENT
    return result.with_value(bit, link_value)


def reduce_cube_list(cubes: list[Cube] | tuple[Cube, ...]) -> list[Cube]:
    if not cubes:
        return []
    cover = ESOPCover(cubes[0].n_vars, list(cubes))
    return sorted(cover.cubes)


def exorlink_group(left: Cube, right: Cube, order: tuple[int, ...]) -> list[Cube]:
    """Generate one ExorLink group by following a path from left to right."""
    diff = left.differing_bits(right)
    if set(order) != set(diff):
        raise ValueError("ExorLink order must be a permutation of differing bits.")
    current = left
    group: list[Cube] = []
    for bit in order:
        nxt = current.with_value(bit, right.value_at(bit))
        group.append(distance1_link_cube(current, nxt))
        current = nxt
    if current != right:
        raise AssertionError("ExorLink path did not end at the right cube.")
    return reduce_cube_list(group)


def exorlink_groups(left: Cube, right: Cube, max_distance: int = 4) -> list[list[Cube]]:
    """Return all distance-k ExorLink groups for k <= max_distance."""
    distance = left.distance(right, stop_after=max_distance)
    if distance < 1 or distance > max_distance:
        return []
    groups = []
    seen = set()
    for order in itertools.permutations(left.differing_bits(right)):
        group = exorlink_group(left, right, order)
        key = tuple(cube.to_pattern() for cube in group)
        if key not in seen:
            seen.add(key)
            groups.append(group)
    return groups


def _truth_from_sop_rows(n_vars: int, rows: list[tuple[str, str]], output_index: int) -> tuple[int, ...]:
    truth = [0] * (1 << n_vars)
    for in_pattern, out_pattern in rows:
        if output_index >= len(out_pattern) or out_pattern[output_index] != "1":
            continue
        cube = Cube.from_pattern(in_pattern)
        for assignment in range(1 << n_vars):
            if cube.matches(assignment):
                truth[assignment] = 1
    return tuple(truth)


def _add_literal_to_cover(cubes: list[Cube], bit: int, value: int) -> list[Cube]:
    return [cube.add_literal(bit, value) for cube in cubes]


def exact_pseudokro_cover_from_truth(truth: tuple[int, ...], n_vars: int) -> ESOPCover:
    """Generate an exact small Pseudo-Kronecker-style ESOP starting cover.

    The implementation tries Shannon, positive Davio, and negative Davio
    recursively and chooses the cover with the fewest cubes, breaking ties by
    literal count. This follows the starting-cover intent in Section 4 without
    requiring a BDD package.
    """
    if len(truth) != (1 << n_vars):
        raise ValueError("truth table length does not match n_vars.")

    cache: dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[Cube, ...]] = {}
    variables = tuple(range(n_vars - 1, -1, -1))

    def best_cover(table: tuple[int, ...], vars_left: tuple[int, ...]) -> tuple[Cube, ...]:
        key = (table, vars_left)
        if key in cache:
            return cache[key]
        if not any(table):
            cache[key] = tuple()
            return tuple()
        if all(table):
            cache[key] = (Cube.one(n_vars),)
            return cache[key]
        if not vars_left:
            raise AssertionError("Non-constant truth table with no variables left.")

        top = vars_left[0]
        rest = vars_left[1:]
        half = len(table) // 2
        f0 = table[:half]
        f1 = table[half:]
        fxor = tuple(a ^ b for a, b in zip(f0, f1))

        c0 = list(best_cover(f0, rest))
        c1 = list(best_cover(f1, rest))
        cx = list(best_cover(fxor, rest))

        candidates = [
            ("pDavio", c0 + _add_literal_to_cover(cx, top, 1)),
            ("nDavio", c1 + _add_literal_to_cover(cx, top, 0)),
            ("Shannon", _add_literal_to_cover(c0, top, 0) + _add_literal_to_cover(c1, top, 1)),
        ]

        reduced = []
        expansion_rank = {"pDavio": 0, "nDavio": 1, "Shannon": 2}
        for expansion, candidate in candidates:
            cover = ESOPCover(n_vars, candidate)
            cubes = tuple(sorted(cover.cubes))
            reduced.append((expansion, cubes))
        _best_expansion, best = min(
            reduced,
            key=lambda item: (
                len(item[1]),
                sum(c.n_lits() for c in item[1]),
                expansion_rank[item[0]],
                [c.to_pattern() for c in item[1]],
            ),
        )
        cache[key] = best
        return best

    return ESOPCover(n_vars, list(best_cover(tuple(int(v) & 1 for v in truth), variables)))


def parse_pla_text(text: str, max_truth_vars: int = 16) -> tuple[list[ESOPCover], dict]:
    n_vars = None
    n_outputs = None
    pla_type = "sop"
    rows: list[tuple[str, str]] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("."):
            parts = line.split()
            directive = parts[0].lower()
            if directive == ".i":
                n_vars = int(parts[1])
            elif directive == ".o":
                n_outputs = int(parts[1])
            elif directive == ".type" and len(parts) > 1:
                pla_type = parts[1].lower()
            elif directive == ".e":
                break
            continue
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"Invalid PLA product row: {raw_line!r}")
        rows.append((parts[0], parts[1]))

    if n_vars is None:
        if rows:
            n_vars = len(rows[0][0])
        else:
            raise ValueError("PLA is missing .i and has no rows.")
    if n_outputs is None:
        if rows:
            n_outputs = len(rows[0][1])
        else:
            n_outputs = 1

    covers = [ESOPCover(n_vars) for _ in range(n_outputs)]
    if pla_type == "esop":
        for in_pattern, out_pattern in rows:
            if len(in_pattern) != n_vars:
                raise ValueError("PLA input pattern has wrong width.")
            cube = Cube.from_pattern(in_pattern)
            for output_index, out_char in enumerate(out_pattern[:n_outputs]):
                if out_char == "1":
                    covers[output_index].add_cube(cube)
                elif out_char in {"0", "-"}:
                    continue
                else:
                    raise ValueError(f"Unsupported PLA output character {out_char!r}.")
    else:
        if n_vars > max_truth_vars:
            raise ValueError(
                f"SOP PLA has {n_vars} inputs; exact truth-table starting cover is limited to "
                f"{max_truth_vars}. Use .type esop or raise --max-truth-vars."
            )
        for output_index in range(n_outputs):
            truth = _truth_from_sop_rows(n_vars, rows, output_index)
            covers[output_index] = exact_pseudokro_cover_from_truth(truth, n_vars)

    metadata = {"n_vars": n_vars, "n_outputs": n_outputs, "type": pla_type, "rows": len(rows)}
    return covers, metadata


def parse_pla(path: str | Path, max_truth_vars: int = 16) -> tuple[list[ESOPCover], dict]:
    return parse_pla_text(Path(path).read_text(), max_truth_vars=max_truth_vars)


def write_esop_pla(covers: list[ESOPCover]) -> str:
    if not covers:
        raise ValueError("Need at least one cover.")
    n_vars = covers[0].n_vars
    if any(cover.n_vars != n_vars for cover in covers):
        raise ValueError("All covers must have the same number of variables.")

    cube_outputs: dict[Cube, list[str]] = {}
    for output_index, cover in enumerate(covers):
        for cube in cover.cubes:
            cube_outputs.setdefault(cube, ["0"] * len(covers))[output_index] = "1"

    lines = [
        f".i {n_vars}",
        f".o {len(covers)}",
        f".p {len(cube_outputs)}",
        ".type esop",
    ]
    for cube in sorted(cube_outputs, key=lambda item: item.to_pattern()):
        lines.append(f"{cube.to_pattern()} {''.join(cube_outputs[cube])}")
    lines.append(".e")
    return "\n".join(lines) + "\n"


def _make_pass_stats(
    stage: str,
    strategy: str,
    distance: int,
    attempts: int,
    accepted: int,
    before: ESOPCover,
    after: ESOPCover,
) -> PassStats:
    return PassStats(
        stage=stage,
        strategy=strategy,
        distance=distance,
        attempts=attempts,
        accepted=accepted,
        before_cubes=before.n_cubes(),
        after_cubes=after.n_cubes(),
        before_literals=before.literal_count(),
        after_literals=after.literal_count(),
    )


def _assert_same_truth(cover: ESOPCover, truth: tuple[int, ...] | None, context: str) -> None:
    if truth is not None and cover.truth_table() != truth:
        raise AssertionError(f"{context} changed the represented Boolean function.")


def _distance01_cleanup(
    cover: ESOPCover,
    verify_truth: tuple[int, ...] | None,
    stage: str,
) -> tuple[ESOPCover, list[PassStats]]:
    """Apply the distance-0/1 reductions described in Propositions 1 and 2.

    Distance-0 reduction is implicit in ESOPCover.add_cube(). Distance-1
    reduction is applied until the cover has no distance-1 cube pair. This is
    the reduction that lets an ExorLink reshaping become a cube-count
    improvement after generated cubes are compared with the current cover.
    """
    current = cover.copy()
    trace: list[PassStats] = []
    while True:
        cubes = sorted(current.cubes)
        attempts = 0
        next_cover = None
        for left_index, left in enumerate(cubes):
            for right in cubes[left_index + 1 :]:
                attempts += 1
                if left.distance(right, stop_after=1) != 1:
                    continue
                linked = distance1_link_cube(left, right)
                candidate = current.replace_pair(left, right, [linked])
                _assert_same_truth(candidate, verify_truth, "distance-1 cleanup")
                next_cover = candidate
                break
            if next_cover is not None:
                break
        if next_cover is None:
            return current, trace
        trace.append(_make_pass_stats(stage, "distance01", 1, attempts, 1, current, next_cover))
        current = next_cover


def _candidate_accepted(
    current: ESOPCover,
    candidate: ESOPCover,
    distance: int,
    strategy: str,
    seen_signatures: set[tuple[str, ...]] | None,
) -> bool:
    cur_cubes, cur_lits = current.stats()
    cand_cubes, cand_lits = candidate.stats()
    sig = candidate.signature()
    if sig == current.signature():
        return False
    if seen_signatures is not None and sig in seen_signatures:
        return False
    if strategy == "aggressive":
        if distance == 2:
            return cand_cubes < cur_cubes
        if distance in {3, 4}:
            return cand_cubes <= cur_cubes
        return False
    if strategy == "last_gasp":
        return distance == 4 and cand_cubes <= cur_cubes
    if strategy == "refinement":
        return distance in {2, 3} and cand_cubes <= cur_cubes and cand_lits < cur_lits
    raise ValueError(f"Unknown strategy {strategy!r}.")


def _try_one_exorlink_pass(
    cover: ESOPCover,
    distance: int,
    strategy: str,
    stage: str,
    seen_signatures: set[tuple[str, ...]] | None,
    verify_truth: tuple[int, ...] | None,
) -> tuple[ESOPCover, bool, PassStats, list[PassStats]]:
    """Backtrack through ExorLink candidates until the first successful group."""
    cubes = sorted(cover.cubes)
    attempts = 0
    for left_index, left in enumerate(cubes):
        for right in cubes[left_index + 1 :]:
            pair_distance = left.distance(right, stop_after=distance)
            if pair_distance != distance:
                continue
            for group in exorlink_groups(left, right, max_distance=distance):
                attempts += 1
                candidate = cover.replace_pair(left, right, group)
                candidate, cleanup_trace = _distance01_cleanup(
                    candidate,
                    verify_truth,
                    stage=f"{stage}.distance01",
                )
                _assert_same_truth(candidate, verify_truth, "ExorLink candidate")
                if not _candidate_accepted(cover, candidate, distance, strategy, seen_signatures):
                    continue
                if seen_signatures is not None:
                    seen_signatures.add(candidate.signature())
                stats = _make_pass_stats(stage, strategy, distance, attempts, 1, cover, candidate)
                return candidate, True, stats, cleanup_trace
    stats = _make_pass_stats(stage, strategy, distance, attempts, 0, cover, cover)
    return cover, False, stats, []


def minimize_esop_with_trace(
    cover: ESOPCover,
    quality: int = 5,
    max_dist: int = 4,
    verify: bool | None = None,
    max_steps: int = 10000,
    last_gasp_rounds: int = 4,
) -> tuple[ESOPCover, list[PassStats]]:
    """Run the EXORCISM-4 minimization loop and return a paper-level trace."""
    if max_dist < 1 or max_dist > 4:
        raise ValueError("max_dist must be in [1,4].")
    if quality < 0:
        raise ValueError("quality must be non-negative.")
    if last_gasp_rounds < 0:
        raise ValueError("last_gasp_rounds must be non-negative.")
    if verify is None:
        verify = cover.n_vars <= 12

    original_truth = cover.truth_table() if verify else None
    current, trace = _distance01_cleanup(cover, original_truth, stage="starting_cover")
    steps = 0

    def apply_until_fixed(distance: int, strategy: str, stage: str, seen_signatures: set[tuple[str, ...]]) -> bool:
        nonlocal current, steps
        changed_any = False
        while steps < max_steps:
            candidate, changed, stats, _cleanup_trace = _try_one_exorlink_pass(
                current,
                distance=distance,
                strategy=strategy,
                stage=stage,
                seen_signatures=seen_signatures,
                verify_truth=original_truth,
            )
            if stats.attempts or stats.accepted:
                trace.append(stats)
            if not changed:
                break
            current = candidate
            changed_any = True
            steps += 1
        return changed_any

    for loop_index in range(int(quality)):
        seen_signatures = {current.signature()}
        while steps < max_steps:
            loop_start_signature = current.signature()
            while steps < max_steps:
                aggressive_changed = False
                for distance in range(2, max_dist + 1):
                    aggressive_changed |= apply_until_fixed(
                        distance,
                        "aggressive",
                        stage=f"q{loop_index + 1}.aggressive",
                        seen_signatures=seen_signatures,
                    )
                if not aggressive_changed:
                    break

            if max_dist >= 4:
                for _round_index in range(int(last_gasp_rounds)):
                    candidate, changed, stats, _cleanup_trace = _try_one_exorlink_pass(
                        current,
                        distance=4,
                        strategy="last_gasp",
                        stage=f"q{loop_index + 1}.last_gasp",
                        seen_signatures=seen_signatures,
                        verify_truth=original_truth,
                    )
                    if stats.attempts or stats.accepted:
                        trace.append(stats)
                    if not changed:
                        break
                    current = candidate
                    steps += 1
                    if steps >= max_steps:
                        break

            if current.signature() == loop_start_signature:
                break

    seen_signatures = {current.signature()}
    for distance in (2, 3):
        if distance <= max_dist:
            apply_until_fixed(
                distance,
                "refinement",
                stage="refinement",
                seen_signatures=seen_signatures,
            )

    _assert_same_truth(current, original_truth, "minimized ESOP")
    return current, trace


def minimize_esop(
    cover: ESOPCover,
    quality: int = 5,
    max_dist: int = 4,
    verify: bool | None = None,
    max_steps: int = 10000,
    last_gasp_rounds: int = 4,
) -> ESOPCover:
    """Run the EXORCISM-4 minimization loop."""
    current, _trace = minimize_esop_with_trace(
        cover,
        quality=quality,
        max_dist=max_dist,
        verify=verify,
        max_steps=max_steps,
        last_gasp_rounds=last_gasp_rounds,
    )
    return current


def minimize_covers(
    covers: list[ESOPCover],
    quality: int = 5,
    max_dist: int = 4,
    last_gasp_rounds: int = 4,
) -> list[ESOPCover]:
    return [
        minimize_esop(cover, quality=quality, max_dist=max_dist, last_gasp_rounds=last_gasp_rounds)
        for cover in covers
    ]


def cover_from_truth_function(n_vars: int, fn) -> ESOPCover:
    truth = tuple(int(fn(value)) & 1 for value in range(1 << n_vars))
    return exact_pseudokro_cover_from_truth(truth, n_vars)


def random_truth_cover(n_vars: int, seed: int) -> ESOPCover:
    rng = random.Random(seed)
    truth = tuple(rng.randrange(2) for _ in range(1 << n_vars))
    return exact_pseudokro_cover_from_truth(truth, n_vars)


def assert_exorlink_groups_equivalent(n_vars: int = 4) -> int:
    patterns = []
    for values in itertools.product("01-", repeat=n_vars):
        patterns.append("".join(values))
    checked = 0
    for left_pattern, right_pattern in itertools.combinations(patterns, 2):
        left = Cube.from_pattern(left_pattern)
        right = Cube.from_pattern(right_pattern)
        distance = left.distance(right)
        if not (1 <= distance <= 4):
            continue
        base = ESOPCover(n_vars, [left, right])
        for group in exorlink_groups(left, right, max_distance=4):
            candidate = ESOPCover(n_vars, group)
            if not base.equivalent(candidate):
                raise AssertionError(
                    f"ExorLink changed function: {left_pattern}, {right_pattern} -> "
                    f"{[cube.to_pattern() for cube in group]}"
                )
            checked += 1
    return checked


def assert_trace_obeys_paper_rules(trace: list[PassStats]) -> None:
    for item in trace:
        if item.accepted == 0:
            if item.before_cubes != item.after_cubes or item.before_literals != item.after_literals:
                raise AssertionError(f"Rejected trace item changed cover stats: {item}")
            continue

        if item.strategy == "distance01":
            if item.distance != 1 or item.after_cubes >= item.before_cubes:
                raise AssertionError(f"Invalid distance-1 cleanup trace item: {item}")
        elif item.strategy == "aggressive":
            if item.distance == 2 and item.after_cubes >= item.before_cubes:
                raise AssertionError(f"Invalid aggressive distance-2 acceptance: {item}")
            if item.distance in {3, 4} and item.after_cubes > item.before_cubes:
                raise AssertionError(f"Invalid aggressive distance-{item.distance} acceptance: {item}")
        elif item.strategy == "last_gasp":
            if item.distance != 4 or item.after_cubes > item.before_cubes:
                raise AssertionError(f"Invalid last-gasp acceptance: {item}")
        elif item.strategy == "refinement":
            if item.distance not in {2, 3}:
                raise AssertionError(f"Invalid refinement distance: {item}")
            if item.after_cubes > item.before_cubes or item.after_literals >= item.before_literals:
                raise AssertionError(f"Invalid refinement acceptance: {item}")
        else:
            raise AssertionError(f"Unexpected trace strategy: {item}")


def assert_pseudokro_all_truth_tables(max_vars: int = 3) -> int:
    checked = 0
    for n_vars in range(1, max_vars + 1):
        table_size = 1 << n_vars
        for function_bits in range(1 << table_size):
            truth = tuple((function_bits >> assignment) & 1 for assignment in range(table_size))
            cover = exact_pseudokro_cover_from_truth(truth, n_vars)
            if cover.truth_table() != truth:
                raise AssertionError(f"Pseudo-Kronecker cover mismatch for n={n_vars}, f={function_bits}")
            checked += 1
    return checked


def assert_minimizer_preserves_all_truth_tables(max_vars: int = 3) -> int:
    checked = 0
    for n_vars in range(1, max_vars + 1):
        table_size = 1 << n_vars
        for function_bits in range(1 << table_size):
            truth = tuple((function_bits >> assignment) & 1 for assignment in range(table_size))
            cover = exact_pseudokro_cover_from_truth(truth, n_vars)
            minimized, trace = minimize_esop_with_trace(cover, quality=2, max_dist=4, verify=True)
            if minimized.truth_table() != truth:
                raise AssertionError(f"Minimizer changed function for n={n_vars}, f={function_bits}")
            assert_trace_obeys_paper_rules(trace)
            checked += 1
    return checked


def run_self_tests() -> dict:
    cancellation = ESOPCover.from_patterns(["10-", "10-"])
    assert cancellation.n_cubes() == 0

    left = Cube.from_pattern("10-")
    right = Cube.from_pattern("11-")
    linked = distance1_link_cube(left, right)
    assert linked.to_pattern() == "1--"
    assert ESOPCover(3, [left, right]).equivalent(ESOPCover(3, [linked]))

    checked_groups = assert_exorlink_groups_equivalent(n_vars=4)
    checked_pseudokro = assert_pseudokro_all_truth_tables(max_vars=3)
    checked_exhaustive_min = assert_minimizer_preserves_all_truth_tables(max_vars=3)

    for n_vars in range(1, 5):
        for seed in range(12):
            rng = random.Random(1000 + seed + 100 * n_vars)
            truth = tuple(rng.randrange(2) for _ in range(1 << n_vars))
            original = exact_pseudokro_cover_from_truth(truth, n_vars)
            assert original.truth_table() == truth
            minimized, trace = minimize_esop_with_trace(original, quality=3, max_dist=4, verify=True)
            assert original.equivalent(minimized)
            assert_trace_obeys_paper_rules(trace)
            assert minimized.n_cubes() <= original.n_cubes() or minimized.literal_count() <= original.literal_count()

    # Example often used to illustrate ESOP minimization:
    # A'C' xor A'BC'D' xor A'B'C'D xor AC' xor AB -> B'C' xor A'C'D xor ABC.
    start = ESOPCover.from_patterns(["0-0-", "0100", "0001", "1-0-", "11--"])
    known = ESOPCover.from_patterns(["-00-", "0-01", "111-"])
    assert start.equivalent(known)
    minimized, trace = minimize_esop_with_trace(start, quality=8, max_dist=4, verify=True)
    assert minimized.equivalent(start)
    assert_trace_obeys_paper_rules(trace)
    assert minimized.n_cubes() <= 3

    pla = """
.i 3
.o 2
.type esop
1-- 10
-1- 10
--1 11
.e
"""
    covers, meta = parse_pla_text(pla)
    assert meta["n_vars"] == 3 and meta["n_outputs"] == 2
    assert covers[0].truth_table() == ESOPCover.from_patterns(["1--", "-1-", "--1"]).truth_table()
    assert covers[1].truth_table() == ESOPCover.from_patterns(["--1"]).truth_table()
    round_trip_covers, _ = parse_pla_text(write_esop_pla(covers))
    assert all(a.equivalent(b) for a, b in zip(covers, round_trip_covers))

    return {
        "exorlink_groups_checked": checked_groups,
        "all_small_pseudokro_truth_tables_checked": checked_pseudokro,
        "all_small_minimizer_truth_tables_checked": checked_exhaustive_min,
        "small_random_functions_checked": 48,
        "paper_style_example_start_cubes": start.n_cubes(),
        "paper_style_example_minimized_cubes": minimized.n_cubes(),
    }


def _stats_line(label: str, before: ESOPCover, after: ESOPCover, elapsed: float) -> str:
    return (
        f"{label:18s} "
        f"cubes {before.n_cubes():4d}->{after.n_cubes():4d} "
        f"lits {before.literal_count():4d}->{after.literal_count():4d} "
        f"time {elapsed:.4f}s"
    )


def _trace_summary_lines(trace: list[PassStats]) -> list[str]:
    totals: dict[tuple[str, int], dict[str, int]] = {}
    for item in trace:
        key = (item.strategy, item.distance)
        bucket = totals.setdefault(key, {"attempts": 0, "accepted": 0, "cube_delta": 0, "literal_delta": 0})
        bucket["attempts"] += item.attempts
        bucket["accepted"] += item.accepted
        bucket["cube_delta"] += item.cube_delta
        bucket["literal_delta"] += item.literal_delta

    order = {
        ("distance01", 1): 0,
        ("aggressive", 2): 1,
        ("aggressive", 3): 2,
        ("aggressive", 4): 3,
        ("last_gasp", 4): 4,
        ("refinement", 2): 5,
        ("refinement", 3): 6,
    }
    lines = []
    for key in sorted(totals, key=lambda item: order.get(item, 99)):
        strategy, distance = key
        bucket = totals[key]
        lines.append(
            f"{strategy:11s} d={distance}: attempts={bucket['attempts']} "
            f"accepted={bucket['accepted']} cube_delta={bucket['cube_delta']} "
            f"lit_delta={bucket['literal_delta']}"
        )
    return lines


def run_paper_demo(show_trace: bool = False) -> dict:
    print("EXORCISM-4 algorithmic local verification for rm01_heu.pdf")
    print("Implemented locally: Fig. 2 flow, ExorLink distance 1-4, Table 1 acceptance logic.")
    print("Not locally reproducible without external files/tools: Table 2 alu4.pla, Tables 3-4 tool comparisons.")
    print()

    self_test = run_self_tests()
    print("Self-test passed:", self_test)
    print()

    benchmarks: list[tuple[str, ESOPCover]] = []
    benchmarks.append(("xor4", cover_from_truth_function(4, lambda x: (x.bit_count() & 1))))
    benchmarks.append(("majority5", cover_from_truth_function(5, lambda x: int(x.bit_count() >= 3))))
    benchmarks.append(("random5-s7", random_truth_cover(5, seed=7)))
    benchmarks.append(("random6-s11", random_truth_cover(6, seed=11)))

    results = {}
    for name, cover in benchmarks:
        start = time.perf_counter()
        minimized, trace = minimize_esop_with_trace(cover, quality=5, max_dist=4, verify=True)
        elapsed = time.perf_counter() - start
        assert cover.equivalent(minimized)
        print(_stats_line(name, cover, minimized, elapsed))
        if show_trace:
            for line in _trace_summary_lines(trace):
                print(f"  {line}")
        results[name] = {
            "before_cubes": cover.n_cubes(),
            "after_cubes": minimized.n_cubes(),
            "before_literals": cover.literal_count(),
            "after_literals": minimized.literal_count(),
            "seconds": elapsed,
            "trace_summary": _trace_summary_lines(trace),
        }

    print()
    print("Quality sweep on random6-s11, analogous to the paper's alu4.pla Table 2 trend:")
    base = random_truth_cover(6, seed=11)
    sweep = []
    for quality in range(0, 6):
        start = time.perf_counter()
        minimized = minimize_esop(base, quality=quality, max_dist=4, verify=True)
        elapsed = time.perf_counter() - start
        print(
            f"  q={quality}: cubes={minimized.n_cubes()} "
            f"lits={minimized.literal_count()} time={elapsed:.4f}s"
        )
        sweep.append((quality, minimized.n_cubes(), minimized.literal_count(), elapsed))

    return {"self_test": self_test, "benchmarks": results, "quality_sweep": sweep}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Run embedded correctness tests.")
    parser.add_argument("--demo-paper", action="store_true", help="Run feasible paper-style local experiments.")
    parser.add_argument("--pla", help="Input PLA/ESOP PLA file.")
    parser.add_argument("--out", help="Write minimized ESOP PLA to this file.")
    parser.add_argument("--quality", type=int, default=5, help="Number of minimization quality loops.")
    parser.add_argument("--max-dist", type=int, default=4, choices=[1, 2, 3, 4], help="Maximum ExorLink distance.")
    parser.add_argument("--last-gasp-rounds", type=int, default=4, help="Distance-4 last-gasp attempts per quality loop.")
    parser.add_argument("--max-steps", type=int, default=10000, help="Maximum accepted transformations.")
    parser.add_argument("--max-truth-vars", type=int, default=16, help="Exact SOP truth-table conversion limit.")
    parser.add_argument("--trace", action="store_true", help="Print an algorithm-stage trace summary.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    did_something = False

    if args.self_test:
        print("Self-test passed:", run_self_tests())
        did_something = True

    if args.demo_paper:
        run_paper_demo(show_trace=args.trace)
        did_something = True

    if args.pla:
        covers, metadata = parse_pla(args.pla, max_truth_vars=args.max_truth_vars)
        print(f"Read PLA: {metadata}")
        minimized = []
        for index, cover in enumerate(covers):
            start = time.perf_counter()
            out, trace = minimize_esop_with_trace(
                cover,
                quality=args.quality,
                max_dist=args.max_dist,
                max_steps=args.max_steps,
                last_gasp_rounds=args.last_gasp_rounds,
            )
            elapsed = time.perf_counter() - start
            if not cover.equivalent(out):
                raise AssertionError(f"Output {index} minimization changed the function.")
            print(_stats_line(f"output[{index}]", cover, out, elapsed))
            if args.trace:
                for line in _trace_summary_lines(trace):
                    print(f"  {line}")
            minimized.append(out)
        if args.out:
            Path(args.out).write_text(write_esop_pla(minimized))
            print(f"Wrote minimized ESOP PLA: {args.out}")
        did_something = True

    if not did_something:
        print("Nothing to do. Use --self-test, --demo-paper, or --pla input.pla.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
