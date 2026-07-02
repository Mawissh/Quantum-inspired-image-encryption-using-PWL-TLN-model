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
   
   def evaluate_assignment(self, assignment: int) -> int:
        value = 0
        for cube in self.cubes:
            if cube.matches(assignment):
                value ^= 1
        return value

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
if __name__ == "__main__":
    raise SystemExit(main())
