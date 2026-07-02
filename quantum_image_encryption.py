"""Implement Sections 1.1 through 2.4.3 of the ELSA draft in plain Python.

This file replaces the exploratory notebook code. It contains:
- NEQR encoding/decoding from Section 1.1.
- Improved generalized quantum Arnold transform helpers.
- TLN model implementation from Section 2.1.
- ESOP Table 1 S-box substitution, key generation, encryption, and small
  quantum-circuit checks through Section 2.4.2.
  Run:
    python quantum_image_encryption.py

The script writes Section 2.2 figure reproductions as SVG files under
``figures_2_2/`` by default.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import hmac
import json
import secrets
from dataclasses import dataclass
from itertools import permutations
from math import log2
from pathlib import Path

import numpy as np

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except Exception:  # pragma: no cover - keeps non-cloud workflows usable.
    AESGCM = None

try:
    from pqcrypto.kem import ml_kem_768
except Exception:  # pragma: no cover - keeps non-cloud workflows usable.
    ml_kem_768 = None

try:
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit.circuit import Gate
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import Statevector
    import qiskit.qasm2 as qiskit_qasm2
except Exception:  # pragma: no cover - keeps TLN-only work usable if qiskit is absent.
    ClassicalRegister = None
    Gate = None
    QuantumCircuit = None
    QuantumRegister = None
    Statevector = None
    UnitaryGate = None
    qiskit_qasm2 = None

try:
    from exorcism4_minimizer import exact_pseudokro_cover_from_truth, minimize_esop_with_trace
except Exception:  # pragma: no cover - keeps the core paper implementation usable standalone.
    exact_pseudokro_cover_from_truth = None
    minimize_esop_with_trace = None
def _require_qiskit() -> None:
    if QuantumCircuit is None or QuantumRegister is None or Statevector is None:
        raise RuntimeError("Qiskit is required for the quantum-circuit functions.")


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _bits_msb(value: int, width: int) -> list[int]:
    """Return ``width`` bits in most-significant-bit first order."""
    return [(int(value) >> shift) & 1 for shift in range(width - 1, -1, -1)]


def _int_from_msb_bits(bits) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _validate_neqr_image(image, q: int) -> tuple[np.ndarray, int]:
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("NEQR expects a 2D grayscale image array.")

    height, width = image.shape
    if height != width:
        raise ValueError("NEQR section 1.1 assumes a square 2^n x 2^n image.")
    if not _is_power_of_two(height):
        raise ValueError("Image side length must be a power of two.")
    if q < 1:
        raise ValueError("q must be at least 1.")
    if np.min(image) < 0 or np.max(image) >= 2**q:
        raise ValueError(f"Pixel values must lie in [0, {2**q - 1}] for q={q}.")

    return image.astype(int), int(log2(height))


def build_neqr_circuit(image, q: int = 8, measure: bool = False, name: str = "NEQR"):
    """Build the NEQR circuit from Section 1.1.

    For an image of size ``2^n x 2^n`` with q-bit pixel values, the circuit
    prepares

        |C> = 1/2^n sum_y sum_x |f(y,x)> |y> |x>

    Register convention:
        f[0:q] stores the pixel intensity bits, MSB first.
        y[0:n] stores the row coordinate bits, MSB first.
        x[0:n] stores the column coordinate bits, MSB first.
    """
    _require_qiskit()
    image, n = _validate_neqr_image(image, q)
    side = image.shape[0]

    intensity = QuantumRegister(q, "f")
    y = QuantumRegister(n, "y")
    x = QuantumRegister(n, "x")

    if measure:
        classical = ClassicalRegister(q + 2 * n, "c")
        qc = QuantumCircuit(intensity, y, x, classical, name=name)
    else:
        classical = None
        qc = QuantumCircuit(intensity, y, x, name=name)

    qc.h(y)
    qc.h(x)
    address_controls = list(y) + list(x)

    for y_value in range(side):
        y_bits = _bits_msb(y_value, n)
        for x_value in range(side):
            x_bits = _bits_msb(x_value, n)
            pixel_bits = _bits_msb(int(image[y_value, x_value]), q)

            for i, bit in enumerate(y_bits):
                if bit == 0:
                    qc.x(y[i])
            for i, bit in enumerate(x_bits):
                if bit == 0:
                    qc.x(x[i])

            for bit_index, bit in enumerate(pixel_bits):
                if bit == 1:
                    qc.mcx(address_controls, intensity[bit_index])

            for i, bit in reversed(list(enumerate(x_bits))):
                if bit == 0:
                    qc.x(x[i])
            for i, bit in reversed(list(enumerate(y_bits))):
                if bit == 0:
                    qc.x(y[i])

    if measure:
        offset = 0
        for i in range(q):
            qc.measure(intensity[i], classical[offset + i])
        offset += q
        for i in range(n):
            qc.measure(y[i], classical[offset + i])
        offset += n
        for i in range(n):
            qc.measure(x[i], classical[offset + i])

    metadata = {
        "q": q,
        "n": n,
        "side": side,
        "intensity": intensity,
        "y": y,
        "x": x,
        "classical": classical,
    }
    return qc, metadata
def decode_neqr_statevector(qc, q: int, image_shape, atol: float = 1e-9):
    """Decode a small NEQR circuit exactly from its statevector."""
    _require_qiskit()
    qc_no_measure = qc.remove_final_measurements(inplace=False)
    state = Statevector.from_instruction(qc_no_measure)

    side = image_shape[0]
    n = int(log2(side))
    recovered = np.zeros(image_shape, dtype=int)
    probabilities = np.zeros(image_shape, dtype=float)
    expected_probability = 1.0 / (side * side)

    for basis_index, amplitude in enumerate(state.data):
        probability = abs(amplitude) ** 2
        if probability <= atol:
            continue

        qubit_bits = [(basis_index >> position) & 1 for position in range(qc_no_measure.num_qubits)]
        f_bits = qubit_bits[0:q]
        y_bits = qubit_bits[q : q + n]
        x_bits = qubit_bits[q + n : q + 2 * n]

        intensity_value = _int_from_msb_bits(f_bits)
        y_value = _int_from_msb_bits(y_bits)
        x_value = _int_from_msb_bits(x_bits)

        recovered[y_value, x_value] = intensity_value
        probabilities[y_value, x_value] += probability

    if not np.allclose(probabilities, expected_probability, atol=1e-8):
        raise AssertionError(f"Address probabilities are not uniform:\n{probabilities}")

    return recovered, probabilities
  def madd_gate(n: int, coeff: int = 1, label: str | None = None):
    """Functional modular-adder primitive used by IGQAT.

    The action is:

        |source>|target> -> |source>|target + coeff*source mod 2^n>

    The video-encryption paper implements this with the MADD circuit built from
    the PG1/TR2 modular adder in Li et al. This dense unitary version is for
    small exact verification.
    """
    _require_qiskit()
    if UnitaryGate is None:
        raise RuntimeError("Qiskit UnitaryGate is required for MADD.")
    if n < 1:
        raise ValueError("n must be at least 1.")

    modulus = 2**n
    coeff = int(coeff) % modulus
    size = 2 ** (2 * n)
    matrix = np.zeros((size, size), dtype=complex)

    for input_index in range(size):
        bits = [(input_index >> position) & 1 for position in range(2 * n)]
        source_bits = bits[:n]
        target_bits = bits[n:]

        source_value = _int_from_msb_bits(source_bits)
        target_value = _int_from_msb_bits(target_bits)
        new_target_value = (target_value + coeff * source_value) % modulus

        output_bits = source_bits + _bits_msb(new_target_value, n)
        output_index = sum(bit << position for position, bit in enumerate(output_bits))
        matrix[output_index, input_index] = 1.0

    gate_label = label or (f"MADD_{n}" if coeff == 1 else f"MADD_{coeff}_{n}")
    return UnitaryGate(matrix, label=gate_label)
def classical_igqat_map(y_value: int, x_value: int, n: int, a: int = 1, b: int = 1, iterations: int = 1):
    """Classical coordinate map for the MADD-based iterative IGQAT."""
    modulus = 2**n
    y_new = int(y_value)
    x_new = int(x_value)

    for _ in range(int(iterations)):
        x_new = (x_new + int(a) * y_new) % modulus
        y_new = (y_new + int(b) * x_new) % modulus

    return y_new, x_new


def classical_igqat_direct_map(
    y_value: int,
    x_value: int,
    n: int,
    a: int = 1,
    b: int = 1,
    iterations: int = 1,
):
    """Same coordinate map computed by A^h mod 2^n."""
    a_h = igqat_direct_matrix(n, a=a, b=b, iterations=iterations)
    modulus = 2**n
    x_new = (a_h[0, 0] * int(x_value) + a_h[0, 1] * int(y_value)) % modulus
    y_new = (a_h[1, 0] * int(x_value) + a_h[1, 1] * int(y_value)) % modulus
    return y_new, x_new
def build_igqat_circuit(
    n: int,
    a: int = 1,
    b: int = 1,
    iterations: int = 1,
    name: str = "IGQAT",
    gate_level: bool = False,
):
    """Build a standalone improved generalized quantum Arnold transform circuit."""
    _require_qiskit()
    y = QuantumRegister(n, "y")
    x = QuantumRegister(n, "x")
    qc = QuantumCircuit(y, x, name=name)

    if gate_level:
        if n != 1:
            raise NotImplementedError("Gate-level IGQAT MADD is currently implemented for n=1 verification circuits.")
        for _ in range(int(iterations)):
            if int(a) & 1:
                qc.cx(y[0], x[0])
            if int(b) & 1:
                qc.cx(x[0], y[0])
        return qc

    madd_a = madd_gate(n, a, label=f"MADD(a={a})")
    madd_b = madd_gate(n, b, label=f"MADD(b={b})")

    for _ in range(int(iterations)):
        qc.append(madd_a, list(y) + list(x))
        qc.append(madd_b, list(x) + list(y))

    return qc


def append_igqat_to_neqr(qc, neqr_info, a: int = 1, b: int = 1, iterations: int = 1, gate_level: bool = False):
    """Append IGQAT to the coordinate registers of an existing NEQR circuit."""
    n = neqr_info["n"]
    out = qc.copy()
    if gate_level:
        if n != 1:
            raise NotImplementedError("Gate-level IGQAT MADD is currently implemented for n=1 verification circuits.")
        for _ in range(int(iterations)):
            if int(a) & 1:
                out.cx(neqr_info["y"][0], neqr_info["x"][0])
            if int(b) & 1:
                out.cx(neqr_info["x"][0], neqr_info["y"][0])
        return out

    madd_a = madd_gate(n, a, label=f"MADD(a={a})")
    madd_b = madd_gate(n, b, label=f"MADD(b={b})")

    for _ in range(int(iterations)):
        out.append(madd_a, list(neqr_info["y"]) + list(neqr_info["x"]))
        out.append(madd_b, list(neqr_info["x"]) + list(neqr_info["y"]))

    return out
def create_tln_parameters(
    a1: float = 0.1,
    a2: float = 0.1,
    b11: float = 0.1,
    b12: float = 0.5,
    b21: float = -1.0,
    b22: float = 2.0,
    i1: float = 1.1,
    i2: float = 2.0,
    c: float = 0.1,
    d: float = 4.0,
    alpha: float = 3.3,
    beta: float = 0.3,
) -> dict[str, float]:
    """Return the PWL-TLN parameters used in Section 2.1."""
    return {
        "a1": float(a1),
        "a2": float(a2),
        "b11": float(b11),
        "b12": float(b12),
        "b21": float(b21),
        "b22": float(b22),
        "I1": float(i1),
        "I2": float(i2),
        "c": float(c),
        "d": float(d),
        "alpha": float(alpha),
        "beta": float(beta),
    }
def tln_rhs(state, params) -> np.ndarray:
    """Continuous-time two-neuron Tabu learning neuron model.

    State order is [x1, x2, v1, v2]. The second equation uses -a2*x2, which is
    the standard state damping term for neuron 2.
    """
    x1, x2, v1, v2 = np.asarray(state, dtype=float)
    f1 = tln_pwl_activation(x1, params)
    f2 = tln_pwl_activation(x2, params)

    dx1 = -params["a1"] * x1 + params["b11"] * f1 + params["b12"] * f2 + v1 + params["I1"]
    dx2 = -params["a2"] * x2 + params["b21"] * f1 + params["b22"] * f2 + v2 + params["I2"]
    dv1 = -params["c"] * v1 - params["d"] * f1
    dv2 = -params["c"] * v2 - params["d"] * f2

    return np.array([dx1, dx2, dv1, dv2], dtype=float)
def tln_divergence(state, params) -> float:
    """Volume contraction rate Psi = trace(J)."""
    return float(np.trace(tln_jacobian(state, params)))
def tln_unit_interval_sequence(trajectory) -> np.ndarray:
    """Map a TLN trajectory to [0,1) by taking the fractional part of abs(state)."""
    return np.mod(np.abs(np.asarray(trajectory, dtype=float)), 1.0)
def section_2_2_table_params(b11: float = 0.1, d: float = 0.4) -> dict[str, float]:
    """Parameter set used by Tables 1 and 2 in Section 2.2."""
    return create_tln_parameters(
        a1=1.0,
        a2=1.0,
        b11=b11,
        b12=0.2,
        b21=0.2,
        b22=0.5,
        i1=0.0,
        i2=0.0,
        c=1.0,
        d=d,
        alpha=3.3,
        beta=0.3,
    )


def tln_origin_eigenvalues(params) -> np.ndarray:
    """Eigenvalues of the Jacobian at the trivial equilibrium (0,0,0,0)."""
    return np.linalg.eigvals(tln_jacobian(np.zeros(4), params))
def estimate_max_lyapunov_tln(
    initial_state,
    params,
    steps: int = 2000,
    dt: float = 0.005,
    burn_in: int = 200,
    delta0: float = 1e-8,
    renormalize_every: int = 10,
    seed: int = 7,
) -> float:
    """Benettin-style maximum Lyapunov estimate for Section 2.2 plots."""
    rng = np.random.default_rng(seed)
    state = np.asarray(initial_state, dtype=float)

    for _ in range(burn_in):
        state = rk4_step(tln_rhs, state, dt, params)

    direction = rng.normal(size=4)
    direction /= np.linalg.norm(direction)
    perturbed = state + delta0 * direction

    log_growth = 0.0
    elapsed = 0.0
    chunks = max(1, steps // renormalize_every)

    for _ in range(chunks):
        for _ in range(renormalize_every):
            state = rk4_step(tln_rhs, state, dt, params)
            perturbed = rk4_step(tln_rhs, perturbed, dt, params)

        separation = perturbed - state
        distance = np.linalg.norm(separation)
        if distance == 0 or not np.isfinite(distance):
            return np.nan

        log_growth += np.log(distance / delta0)
        elapsed += renormalize_every * dt
        perturbed = state + delta0 * separation / distance

    return float(log_growth / elapsed)
def one_parameter_bifurcation_scan(
    parameter_name: str,
    values,
    base_params=None,
    initial_state=(-0.54, -0.1, 0.0, 0.0),
    steps: int = 2000,
    dt: float = 0.005,
    burn_in: int = 500,
    lyapunov_steps: int = 1000,
    max_points_per_parameter: int = 64,
    continuation: bool = True,
) -> list[dict]:
    """Generate one-parameter bifurcation data and a matching max-Lyapunov trace."""
    if base_params is None:
        base_params = create_tln_parameters()

    records = []
    current_state = np.asarray(initial_state, dtype=float)

    for value in np.asarray(values, dtype=float):
        params = dict(base_params)
        params[parameter_name] = float(value)
        _, trajectory = simulate_tln(current_state, params=params, steps=steps, dt=dt, burn_in=burn_in)
        maxima = local_maxima(trajectory[:, 0])[-max_points_per_parameter:]
        max_lyapunov = estimate_max_lyapunov_tln(
            current_state,
            params,
            steps=lyapunov_steps,
            dt=dt,
            burn_in=burn_in,
        )
        records.append(
            {
                "parameter": float(value),
                "maxima_x1": maxima,
                "max_lyapunov": max_lyapunov,
                "final_state": trajectory[-1].copy(),
            }
        )

        if continuation:
            current_state = trajectory[-1]

    return records
IRREDUCIBLE_POLYS_8 = (
    0x11B, 0x11D, 0x12B, 0x12D, 0x139, 0x13F, 0x14D, 0x15F,
    0x163, 0x165, 0x169, 0x171, 0x177, 0x17B, 0x187, 0x18B,
    0x18D, 0x19F, 0x1A3, 0x1A9, 0x1B1, 0x1BD, 0x1C3, 0x1CF,
    0x1D7, 0x1DD, 0x1E7, 0x1F3, 0x1F5,
)
@dataclass
class Parameters:
    """Table 3 parameter settings for the Section 2.3 S-box search."""

    population_size: int = 120
    generations: int = 300
    stagnation_threshold: int = 35
    jumping_rate: float = 0.08
    mutation_probability: float = 0.04
    crossover_probability: float = 0.90
    tournament_size: int = 3
    elite_fraction: float = 0.05
    moora_rate: float = 0.05
    weight_floor: float = 0.02
    n_bits: int = 8
    n_polynomials: int = 29


@dataclass
class SBoxChromosome:
    """Continuous and stored binary chromosome C=(X,A,b,p)."""

    X: np.ndarray
    A: np.ndarray
    b: np.ndarray
    p_idx: int
    fitness: float = -1e9
    moora_score: float = -1e9
    sbox: np.ndarray | None = None
    metrics: dict | None = None
INVERSE_TABLE_CACHE: dict[tuple[int, int], np.ndarray] = {}
def gf_multiply(a: int, b: int, poly: int, n_bits: int = 8) -> int:
    """Multiply in GF(2^n) under the given irreducible polynomial."""
    result = 0
    mask = (1 << n_bits) - 1
    carry = 1 << n_bits
    a &= mask
    b &= mask

    while b:
        if b & 1:
            result ^= a
        b >>= 1
        a <<= 1
        if a & carry:
            a ^= poly
        a &= mask

    return result & mask


def build_inverse_table(poly: int, n_bits: int = 8) -> np.ndarray:
    """Build the GF(2^n) multiplicative inverse table with inverse(0)=0."""
    key = (int(poly), int(n_bits))
    if key in _INVERSE_TABLE_CACHE:
        return _INVERSE_TABLE_CACHE[key].copy()

    size = 1 << n_bits
    inverse = np.zeros(size, dtype=np.uint16)
    for x in range(1, size):
        for y in range(1, size):
            if gf_multiply(x, y, poly, n_bits=n_bits) == 1:
                inverse[x] = y
                break

    _INVERSE_TABLE_CACHE[key] = inverse.copy()
    return inverse


def gf2_rank(matrix) -> int:
    """Rank over GF(2)."""
    A = np.asarray(matrix, dtype=np.uint8).copy() & 1
    rows, cols = A.shape
    rank = 0

    for col in range(cols):
        pivot = None
        for row in range(rank, rows):
            if A[row, col]:
                pivot = row
                break
        if pivot is None:
            continue
        if pivot != rank:
            A[[rank, pivot]] = A[[pivot, rank]]
        for row in range(rows):
            if row != rank and A[row, col]:
                A[row] ^= A[rank]
        rank += 1
        if rank == rows:
            break

    return rank


def gf2_is_invertible(matrix) -> bool:
    matrix = np.asarray(matrix, dtype=np.uint8)
    return matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1] and gf2_rank(matrix) == matrix.shape[0]
def make_sbox_chromosome(
    X,
    p_idx: int,
    rng=None,
    n_bits: int = 8,
    repair: bool = True,
    stochastic: bool = True,
) -> SBoxChromosome:
    """Create C=(X,A,b,p) and store one binarized affine pair for its lifetime."""
    X = np.clip(np.asarray(X, dtype=float).ravel(), 0.0, 1.0)
    expected = n_bits * n_bits + n_bits
    if len(X) != expected:
        raise ValueError(f"Expected X length {expected}, got {len(X)}.")
    if stochastic:
        if rng is None:
            raise ValueError("stochastic chromosome creation requires rng.")
        bits = stochastic_binarize(X, rng)
    else:
        bits = deterministic_binarize(X)
    A, b = decode_sbox_chromosome_bits(bits, n_bits=n_bits)
    if repair and not gf2_is_invertible(A):
        A = gf2_repair_matrix(A)
        X = encode_sbox_chromosome_bits(A, b).astype(float)
    return SBoxChromosome(X=X, A=A, b=b, p_idx=int(p_idx) % len(IRREDUCIBLE_POLYS_8))
def apply_affine_to_int(value: int, A, b, n_bits: int = 8) -> int:
    """Apply y=A*x+b over GF(2), using LSB-first bit vectors."""
    x_vec = _bits_lsb(value, n_bits)
    y_vec = ((np.asarray(A, dtype=np.uint8) @ x_vec) & 1) ^ (np.asarray(b, dtype=np.uint8) & 1)
    return _int_from_lsb_bits(y_vec)


def sbox_from_affine_inverse(A, b, p_idx: int, n_bits: int = 8) -> np.ndarray:
    """Construct S(alpha)=A*alpha^{-1}+b over GF(2^n), Eq. (20)."""
    poly = IRREDUCIBLE_POLYS_8[int(p_idx) % len(IRREDUCIBLE_POLYS_8)]
    inverse = build_inverse_table(poly, n_bits=n_bits)
    size = 1 << n_bits
    sbox = np.zeros(size, dtype=np.uint16)
    for x in range(size):
        sbox[x] = apply_affine_to_int(int(inverse[x]), A, b, n_bits=n_bits)
    return sbox
def validate_sbox(sbox, n_bits: int = 8) -> None:
    sbox = np.asarray(sbox, dtype=int)
    expected = list(range(1 << n_bits))
    if len(sbox) != (1 << n_bits):
        raise ValueError("S-box has the wrong length.")
    if sorted(sbox.tolist()) != expected:
        raise ValueError("S-box is not bijective.")
def recover_affine_from_inverse_sbox(sbox, p_idx: int, n_bits: int = 8):
    """Recover (A,b) if the S-box has the paper's inverse-affine form."""
    sbox = np.asarray(sbox, dtype=np.uint16)
    b_int = int(sbox[0])
    b = _bits_lsb(b_int, n_bits)
    poly = IRREDUCIBLE_POLYS_8[int(p_idx) % len(IRREDUCIBLE_POLYS_8)]
    inverse = build_inverse_table(poly, n_bits=n_bits)
    A = np.zeros((n_bits, n_bits), dtype=np.uint8)

    for col in range(n_bits):
        preimage = int(inverse[1 << col])
        A[:, col] = _bits_lsb(int(sbox[preimage]) ^ b_int, n_bits)

    rebuilt = sbox_from_affine_inverse(A, b, p_idx, n_bits=n_bits)
    if not np.array_equal(rebuilt, sbox):
        return None
    return A, b
def component_nonlinearity_and_lap(sbox, n_bits: int = 8) -> tuple[int, float]:
    """Minimum component nonlinearity and linear-approximation bias."""
    sbox = np.asarray(sbox, dtype=np.uint16)
    size = 1 << n_bits
    max_abs = 0

    for output_mask in range(1, size):
        values = np.array([1 if _parity(output_mask & int(sbox[x])) == 0 else -1 for x in range(size)], dtype=int)
        walsh = _fwht(values)
        max_abs = max(max_abs, int(np.max(np.abs(walsh))))

    nonlinearity = size // 2 - max_abs // 2
    lap = max_abs / (2.0 * size)
    return int(nonlinearity), float(lap)


def sac_offset(sbox, n_bits: int = 8) -> float:
    """Average absolute SAC deviation from 0.5."""
    sbox = np.asarray(sbox, dtype=np.uint16)
    size = 1 << n_bits
    deviations = []
    for input_bit in range(n_bits):
        delta = 1 << input_bit
        for output_bit in range(n_bits):
            flips = sum(((int(sbox[x]) ^ int(sbox[x ^ delta])) >> output_bit) & 1 for x in range(size))
            deviations.append(abs(flips / size - 0.5))
    return float(np.mean(deviations))
def bic_metrics(sbox, n_bits: int = 8) -> tuple[int, float]:
    """BIC nonlinearity and BIC-SAC deviation for pairwise output-bit XORs."""
    sbox = np.asarray(sbox, dtype=np.uint16)
    size = 1 << n_bits
    best_nl = size
    deviations = []

    for bit_i in range(n_bits):
        for bit_j in range(bit_i + 1, n_bits):
            output_mask = (1 << bit_i) | (1 << bit_j)
            values = np.array([1 if _parity(output_mask & int(sbox[x])) == 0 else -1 for x in range(size)], dtype=int)
            walsh = _fwht(values)
            max_abs = int(np.max(np.abs(walsh)))
            best_nl = min(best_nl, size // 2 - max_abs // 2)

            for input_bit in range(n_bits):
                delta = 1 << input_bit
                flips = sum(
                    _parity((int(sbox[x]) ^ int(sbox[x ^ delta])) & output_mask)
                    for x in range(size)
                )
                deviations.append(abs(flips / size - 0.5))

    return int(best_nl), float(np.mean(deviations))
def sbox_cycle_stats(sbox) -> tuple[int, int, list[int]]:
    sbox = np.asarray(sbox, dtype=np.uint16)
    seen = np.zeros(len(sbox), dtype=bool)
    lengths = []

    for start in range(len(sbox)):
        if seen[start]:
            continue
        cur = start
        length = 0
        while not seen[cur]:
            seen[cur] = True
            cur = int(sbox[cur])
            length += 1
        lengths.append(length)

    return int(min(lengths)), int(len(lengths)), lengths


def evaluate_sbox_metrics(sbox, n_bits: int = 8, full: bool = True) -> dict:
    """Compute the Section 2.3 S-box metrics used by fitness/MOORA."""
    sbox = np.asarray(sbox, dtype=np.uint16)
    validate_sbox(sbox, n_bits=n_bits)
    size = 1 << n_bits
    fixed_points = int(np.sum(sbox == np.arange(size)))
    reverse_fixed_points = int(np.sum((sbox ^ np.arange(size, dtype=np.uint16)) == (size - 1)))
    min_cycle, cycle_count, cycle_lengths = sbox_cycle_stats(sbox)

    metrics = {
        "bijective": True,
        "fixed_points": fixed_points,
        "reverse_fixed_points": reverse_fixed_points,
        "min_cycle": min_cycle,
        "cycle_count": cycle_count,
        "cycle_lengths": cycle_lengths,
    }

    if full:
        nonlinearity, lap = component_nonlinearity_and_lap(sbox, n_bits=n_bits)
        bic_nl, bic_sac = bic_metrics(sbox, n_bits=n_bits)
        max_diff_count, dap = differential_approximation_probability(sbox, n_bits=n_bits)
        metrics.update(
            {
                "nonlinearity": nonlinearity,
                "sac_offset": sac_offset(sbox, n_bits=n_bits),
                "bic_nonlinearity": bic_nl,
                "bic_sac_deviation": bic_sac,
                "max_differential_count": max_diff_count,
                "dap": dap,
                "lap": lap,
            }
        )

    return metrics


def tqcoblah_fitness(metrics: dict, n_bits: int = 8) -> float:
    """Three-stage fitness evaluation from Algorithm 2."""
    if not metrics.get("bijective", False):
        return -1e9

    fixed_points = metrics["fixed_points"]
    reverse_fixed_points = metrics["reverse_fixed_points"]
    min_cycle = metrics["min_cycle"]
    if fixed_points > 0 or reverse_fixed_points > 0 or min_cycle <= n_bits // 2:
        if min_cycle == 1:
            cycle_term = 2 ** (n_bits + 4)
        elif min_cycle <= 2:
            cycle_term = n_bits**4
        elif min_cycle <= n_bits // 2:
            cycle_term = 20 * n_bits
        else:
            cycle_term = 0
        penalty = n_bits**2 * fixed_points + n_bits**2 * reverse_fixed_points + cycle_term
        return float(-penalty - 1000)

    nonlinearity = metrics["nonlinearity"]
    sac = metrics["sac_offset"]
    if nonlinearity < 2 ** (n_bits - 2):
        return float(1000 * nonlinearity - 10 * sac)

    return float(
        -10000 * sac
        - 2000 * metrics["bic_sac_deviation"]
        + 10 * metrics["bic_nonlinearity"]
        + 2 * metrics["cycle_count"]
        + 0.5 * metrics["min_cycle"]
    )


def evaluate_chromosome(chromosome: SBoxChromosome, n_bits: int = 8, full_metrics: bool = True) -> SBoxChromosome:
    if not gf2_is_invertible(chromosome.A):
        chromosome.fitness = -1e9
        chromosome.metrics = {"bijective": False}
        return chromosome

    sbox = sbox_from_chromosome(chromosome, n_bits=n_bits)
    chromosome.sbox = sbox
    try:
        metrics = evaluate_sbox_metrics(sbox, n_bits=n_bits, full=full_metrics)
    except ValueError:
        chromosome.fitness = -1e9
        chromosome.metrics = {"bijective": False}
        return chromosome

    chromosome.metrics = metrics
    chromosome.fitness = tqcoblah_fitness(metrics, n_bits=n_bits)
    return chromosome


def moora_scores(chromosomes: list[SBoxChromosome], weights: dict[str, float]) -> np.ndarray:
    """Adaptive MOORA score from Eq. (23)-(24)."""
    metrics = ["nonlinearity", "sac_offset", "bic_nonlinearity", "bic_sac_deviation", "dap", "lap"]
    beneficial = {"nonlinearity", "bic_nonlinearity"}
    values = np.array([[float(ch.metrics[name]) for name in metrics] for ch in chromosomes], dtype=float)
    denominators = np.sqrt(np.sum(values * values, axis=0))
    denominators[denominators == 0] = 1.0
    normalized = values / denominators
    scores = np.zeros(len(chromosomes), dtype=float)

    for index, name in enumerate(metrics):
        sign = 1.0 if name in beneficial else -1.0
        scores += sign * weights[name] * normalized[:, index]

    return scores

def select_top_chromosomes(chromosomes: list[SBoxChromosome], count: int) -> list[SBoxChromosome]:
    return sorted(chromosomes, key=lambda ch: (ch.moora_score, ch.fitness), reverse=True)[:count]


def initialize_tqcoblah_population(params: TQCOBLAHParameters, seed: int = 7) -> list[SBoxChromosome]:
    """Algorithm 1: TLN/QOBL population initialization."""
    rng = np.random.default_rng(seed)
    n_bits = params.n_bits
    chromosome_len = n_bits * n_bits + n_bits
    needed = params.population_size * (chromosome_len + 1)
    values = tln_chaotic_values(needed, seed=seed)
    population = []
    cursor = 0

    for _ in range(params.population_size):
        X = values[cursor : cursor + chromosome_len]
        cursor += chromosome_len
        p_idx = int(values[cursor] * params.n_polynomials) % params.n_polynomials
        cursor += 1
        population.append(make_sbox_chromosome(X, p_idx, rng=rng, n_bits=n_bits, stochastic=True))

        qX = qobl_vector(X, rng)
        q_poly_choices = [idx for idx in range(params.n_polynomials) if idx != p_idx]
        q_p_idx = int(rng.choice(q_poly_choices))
        population.append(make_sbox_chromosome(qX, q_p_idx, rng=rng, n_bits=n_bits, stochastic=True))

    for chromosome in population:
        evaluate_chromosome(chromosome, n_bits=n_bits, full_metrics=True)

    weights = {name: 1.0 / 6.0 for name in ["nonlinearity", "sac_offset", "bic_nonlinearity", "bic_sac_deviation", "dap", "lap"]}
    scores = moora_scores(population, weights)
    for chromosome, score in zip(population, scores):
        chromosome.moora_score = float(score)
    return select_top_chromosomes(population, params.population_size)
def run_tqcoblah_sbox(seed: int = 7, params: TQCOBLAHParameters | None = None) -> tuple[np.ndarray, SBoxChromosome]:
    """Algorithm 3: modified GA with adaptive MOORA and OBL generation jumping."""
    if params is None:
        params = TQCOBLAHParameters()
    rng = np.random.default_rng(seed)
    population = initialize_tqcoblah_population(params, seed=seed)
    weights = {name: 1.0 / 6.0 for name in ["nonlinearity", "sac_offset", "bic_nonlinearity", "bic_sac_deviation", "dap", "lap"]}
    best = max(population, key=lambda ch: ch.fitness)
    best_score = -np.inf
    stagnation = 0
    mode = "QOBL"

    for _ in range(params.generations):
        weights = update_moora_weights(population, weights, rate=params.moora_rate, floor=params.weight_floor)
        scores = moora_scores(population, weights)
        for chromosome, score in zip(population, scores):
            chromosome.moora_score = float(score)

        current = max(population, key=lambda ch: ch.moora_score)
        if current.moora_score > best_score:
            best = current
            best_score = current.moora_score
            stagnation = 0
            mode = "QOBL"
        else:
            stagnation += 1
            if stagnation >= params.stagnation_threshold:
                mode = "COOBL"
                stagnation = 0

        elite_count = max(2, int(np.floor(params.elite_fraction * params.population_size)))
        next_population = select_top_chromosomes(population, elite_count)
        while len(next_population) < params.population_size:
            parent_a = tournament_select(population, rng, params.tournament_size)
            parent_b = tournament_select(population, rng, params.tournament_size)
            child = crossover_chromosomes(parent_a, parent_b, rng, params)
            child = mutate_chromosome(child, rng, params)
            evaluate_chromosome(child, n_bits=params.n_bits, full_metrics=True)
            next_population.append(child)

        population = next_population
        if rng.random() < params.jumping_rate:
            population = apply_generation_jumping(population, best, mode, rng, params, weights)

    best = max(population + [best], key=lambda ch: ch.fitness)
    if best.sbox is None:
        best.sbox = sbox_from_chromosome(best, n_bits=params.n_bits)
    return best.sbox.copy(), best
DEFAULT_TLN_CHAOTIC_BASIN_BOUNDS = np.array(
    [
        [-0.64, -0.44],
        [-0.20, 0.00],
        [-0.05, 0.05],
        [-0.05, 0.05],
    ],
    dtype=float,
)


@dataclass(frozen=True)
class Section241KeyMaterial:
    """All values produced by the Section 2.4.1 key-generation procedure."""

    salt: bytes
    nonce: bytes
    master_key: bytes
    k_tln: bytes
    k_auth: bytes
    stream: bytes
    normalized_values: np.ndarray
    initial_conditions: np.ndarray
    basin_bounds: np.ndarray


def _ensure_bytes(value, name: str) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    raise TypeError(f"{name} must be bytes or str.")


def _check_fixed_length(value: bytes, length: int, name: str) -> bytes:
    value = bytes(value)
    if len(value) != length:
        raise ValueError(f"{name} must be exactly {length} bytes, got {len(value)}.")
    return value


def derive_master_key_pbkdf2_sha256(master_key, salt: bytes, iterations: int = 200_000, length: int = 32) -> bytes:
    """Eq. (28): PBKDF2-HMAC-SHA256(Mk, P, 200000, 256 bits)."""
    master_key = _ensure_bytes(master_key, "master_key")
    salt = _check_fixed_length(salt, 16, "salt")
    if iterations < 1:
        raise ValueError("PBKDF2 iterations must be positive.")
    if length < 1:
        raise ValueError("derived key length must be positive.")
    return hashlib.pbkdf2_hmac("sha256", master_key, salt, iterations, dklen=length)
def uint32_be_segments(stream: bytes, count: int) -> np.ndarray:
    """Split stream into unsigned big-endian 32-bit segments."""
    stream = bytes(stream)
    needed = 4 * int(count)
    if len(stream) < needed:
        raise ValueError(f"Need at least {needed} bytes for {count} uint32 segments.")
    return np.array([int.from_bytes(stream[4 * i : 4 * i + 4], "big") for i in range(count)], dtype=np.uint64)
@dataclass(frozen=True)
class Section242Metadata:
    """Public decryption metadata for the Section 2.4.2 image transform."""

    salt: bytes
    nonce: bytes
    arnold_iterations: int
    arnold_r: int
    arnold_z: int
    block_power: int
    q: int = 8
    diffusion_warmup: int = 1000
    diffusion_stride: int = 11


def validate_uint8_square_image(image, require_power_of_two: bool = True) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("Section 2.4.2 expects a 2D grayscale image.")
    if image.shape[0] != image.shape[1]:
        raise ValueError("Section 2.4.2 expects a square image.")
    if image.dtype != np.uint8:
        raise ValueError("Section 2.4.2 expects uint8 pixels.")
    if require_power_of_two and not _is_power_of_two(int(image.shape[0])):
        raise ValueError("Image side length must be a power of two.")
    return image


def inverse_classical_igqat_image(image, a: int = 1, b: int = 1, iterations: int = 1) -> np.ndarray:
    """Inverse of ``classical_igqat_image`` by reversing the coordinate permutation."""
    image = np.asarray(image)
    if image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError("IGQAT inverse expects a square 2D image.")
    if not _is_power_of_two(image.shape[0]):
        raise ValueError("IGQAT inverse expects side length 2^n.")

    side = image.shape[0]
    n = int(log2(side))
    output = np.zeros_like(image)
    for y_value in range(side):
        for x_value in range(side):
            y_new, x_new = classical_igqat_map(y_value, x_value, n, a, b, iterations)
            output[y_value, x_value] = image[y_new, x_new]
    return output


def gf2_inverse_matrix(matrix) -> np.ndarray:
    """Invert a square binary matrix over GF(2)."""
    matrix = np.asarray(matrix, dtype=np.uint8) & 1
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("GF(2) inverse expects a square matrix.")
    n = matrix.shape[0]
    augmented = np.concatenate([matrix.copy(), np.eye(n, dtype=np.uint8)], axis=1)

    row = 0
    for col in range(n):
        pivot = None
        for candidate in range(row, n):
            if augmented[candidate, col]:
                pivot = candidate
                break
        if pivot is None:
            raise ValueError("Matrix is singular over GF(2).")
        if pivot != row:
            augmented[[row, pivot]] = augmented[[pivot, row]]
        for other in range(n):
            if other != row and augmented[other, col]:
                augmented[other] ^= augmented[row]
        row += 1

    return augmented[:, n:].astype(np.uint8)


def gf2_matrix_power(matrix, exponent: int) -> np.ndarray:
    """Power of a binary matrix over GF(2)."""
    matrix = np.asarray(matrix, dtype=np.uint8) & 1
    result = np.eye(matrix.shape[0], dtype=np.uint8)
    base = matrix.copy()
    exp = int(exponent)
    while exp > 0:
        if exp & 1:
            result = (result @ base) & 1
        base = (base @ base) & 1
        exp >>= 1
    return result


def csgc_matrix(q: int = 8) -> np.ndarray:
    """CSGC circulant matrix from Eq. (36)-(37)."""
    if q < 2:
        raise ValueError("q must be at least 2.")
    ones_count = q // 2 + 1
    start_offset = q - ones_count
    matrix = np.zeros((q, q), dtype=np.uint8)
    for row in range(q):
        for offset in range(start_offset, q):
            matrix[row, (row + offset) % q] = 1
    return matrix
def csgc_pre_diffusion(image, q: int = 8) -> np.ndarray:
    """Apply the CSGC pre-diffusion matrix Mc to the intensity bits."""
    return byte_linear_transform(image, csgc_matrix(q), q=q)


def inverse_csgc_pre_diffusion(image, q: int = 8) -> np.ndarray:
    """Reverse CSGC pre-diffusion using the GF(2) inverse of the printed Mc."""
    matrix = csgc_matrix(q)
    return byte_linear_transform(image, gf2_inverse_matrix(matrix), q=q)
def generate_tln_diffusion_key_matrix(
    image_shape,
    key_material: Section241KeyMaterial,
    q: int = 8,
    warmup: int = 1000,
    stride: int = 11,
    dt: float = 0.005,
) -> np.ndarray:
    """Generate Eq. (41) TLN pseudorandom bytes for all pixels."""
    height, width = image_shape
    count = int(height) * int(width)
    if q != 8:
        raise NotImplementedError("Section 2.4.2 currently targets q=8 grayscale images.")
    if count < 1:
        raise ValueError("image_shape must contain at least one pixel.")
    if warmup < 0 or stride < 1:
        raise ValueError("warmup must be non-negative and stride must be positive.")

    params = tln_diffusion_parameters()
    state = np.asarray(key_material.initial_conditions, dtype=float).copy()
    samples = np.zeros((count, 4), dtype=float)

    for _ in range(warmup):
        state = rk4_step(tln_rhs, state, dt, params)
    for index in range(count):
        for _ in range(stride):
            state = rk4_step(tln_rhs, state, dt, params)
        samples[index] = np.mod(np.abs(state), 1.0)

    mixed = samples[:, 0] * samples[:, 1] + samples[:, 2] * samples[:, 3]
    mixed += samples[:, 0] * samples[:, 2] + samples[:, 1] * samples[:, 3]
    values = np.mod(np.floor(np.abs(mixed) * 1e14), 2**q).astype(np.uint8)
    return values.reshape((height, width))


def apply_tln_xor_diffusion(csgc_image, key_matrix) -> np.ndarray:
    """Apply Eq. (42) to a CSGC-pre-diffused image."""
    key_matrix = np.asarray(key_matrix, dtype=np.uint8)
    if np.asarray(csgc_image).shape != key_matrix.shape:
        raise ValueError("image and key matrix shapes must match.")
    f_image = inverse_csgc_pre_diffusion(csgc_image, q=8)
    transformed = byte_linear_transform(f_image, tln_xor_diffusion_matrix_q8(), q=8)
    return np.bitwise_xor(transformed, key_matrix).astype(np.uint8)


def inverse_tln_xor_diffusion(diffused_image, key_matrix) -> np.ndarray:
    """Invert Eq. (42), returning the CSGC-pre-diffused image."""
    key_matrix = np.asarray(key_matrix, dtype=np.uint8)
    if np.asarray(diffused_image).shape != key_matrix.shape:
        raise ValueError("image and key matrix shapes must match.")
    unkeyed = np.bitwise_xor(np.asarray(diffused_image, dtype=np.uint8), key_matrix)
    f_image = byte_linear_transform(unkeyed, gf2_inverse_matrix(tln_xor_diffusion_matrix_q8()), q=8)
    return csgc_pre_diffusion(f_image, q=8)
def block_sbox_substitution_section_2_4_2(
    image,
    key_material: Section241KeyMaterial,
    block_power: int = 4,
    inverse: bool = False,
    seed_base: int = 7,
) -> np.ndarray:
    """Apply Eq. (49)-(59) as block-wise S-box substitution."""
    image = validate_uint8_square_image(image)
    side = image.shape[0]
    m = int(log2(side))
    if not (0 <= block_power <= m):
        raise ValueError("block_power must satisfy 0 <= p <= m for side=2^m.")
    block_size = 1 << block_power
    if side % block_size != 0:
        raise ValueError("block size must divide image side length.")

def encrypt_section_2_4_2_image(
    image,
    master_key,
    salt: bytes | None = None,
    nonce: bytes | None = None,
    arnold_iterations: int = 59,
    arnold_r: int = 11,
    arnold_z: int = 19,
    block_power: int = 4,
    key_material: Section241KeyMaterial | None = None,
) -> tuple[np.ndarray, Section242Metadata]:
    """Classical simulation of the Section 2.4.2 quantum encryption pipeline."""
    image = validate_uint8_square_image(image)
    side = image.shape[0]
    if block_power > int(log2(side)):
        raise ValueError("block_power is too large for the image side length.")
    if key_material is None:
        key_material = derive_section_2_4_1_key_material(master_key, salt=salt, nonce=nonce)

    scrambled = classical_igqat_image(image, a=arnold_r, b=arnold_z, iterations=arnold_iterations)
    pre_diffused = csgc_pre_diffusion(scrambled, q=8)
    key_matrix = generate_tln_diffusion_key_matrix(
        image.shape,
        key_material,
        q=8,
        warmup=1000,
        stride=11,
    )
    diffused = apply_tln_xor_diffusion(pre_diffused, key_matrix)
    cipher = block_sbox_substitution_section_2_4_2(diffused, key_material, block_power=block_power, inverse=False)

    metadata = Section242Metadata(
        salt=key_material.salt,
        nonce=key_material.nonce,
        arnold_iterations=int(arnold_iterations),
        arnold_r=int(arnold_r),
        arnold_z=int(arnold_z),
        block_power=int(block_power),
    )
    return cipher.astype(np.uint8), metadata


def decrypt_section_2_4_2_image(cipher, master_key, metadata: Section242Metadata) -> np.ndarray:
    """Reverse the Section 2.4.2 image transform."""
    cipher = validate_uint8_square_image(cipher)
    key_material = derive_section_2_4_1_key_material(master_key, salt=metadata.salt, nonce=metadata.nonce)
    desubstituted = block_sbox_substitution_section_2_4_2(
        cipher,
        key_material,
        block_power=metadata.block_power,
        inverse=True,
    )
    key_matrix = generate_tln_diffusion_key_matrix(
        cipher.shape,
        key_material,
        q=metadata.q,
        warmup=metadata.diffusion_warmup,
        stride=metadata.diffusion_stride,
    )
    csgc_image = inverse_tln_xor_diffusion(desubstituted, key_matrix)
    scrambled = inverse_csgc_pre_diffusion(csgc_image, q=metadata.q)
    plain = inverse_classical_igqat_image(
        scrambled,
        a=metadata.arnold_r,
        b=metadata.arnold_z,
        iterations=metadata.arnold_iterations,
    )
    return plain.astype(np.uint8)


def _neqr_basis_index(intensity_value: int, y_value: int, x_value: int, q: int, n: int) -> int:
    bits = _bits_msb(intensity_value, q) + _bits_msb(y_value, n) + _bits_msb(x_value, n)
    return sum(int(bit) << position for position, bit in enumerate(bits))


def _neqr_basis_values(index: int, q: int, n: int) -> tuple[int, int, int]:
    bits = [(int(index) >> position) & 1 for position in range(q + 2 * n)]
    intensity = _int_from_msb_bits(bits[:q])
    y_value = _int_from_msb_bits(bits[q : q + n])
    x_value = _int_from_msb_bits(bits[q + n : q + 2 * n])
    return intensity, y_value, x_value
def _apply_row_cx(matrix, control: int, target: int) -> None:
    matrix[target] ^= matrix[control]


def _synthesize_linear_cnot_ops(matrix) -> list[tuple[str, int, int]]:
    """Return CNOT/SWAP row operations implementing ``matrix`` over GF(2)."""
    matrix = np.asarray(matrix, dtype=np.uint8).copy() & 1
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("linear CNOT synthesis expects a square binary matrix.")
    if not gf2_is_invertible(matrix):
        raise ValueError("linear CNOT synthesis expects an invertible matrix.")

    n = matrix.shape[0]
    reduce_ops: list[tuple[str, int, int]] = []
    for col in range(n):
        pivot = None
        for row in range(col, n):
            if matrix[row, col]:
                pivot = row
                break
        if pivot is None:
            raise AssertionError("invertible matrix lost pivot during synthesis.")
        if pivot != col:
            matrix[[col, pivot]] = matrix[[pivot, col]]
            reduce_ops.append(("swap", col, pivot))
        for row in range(n):
            if row != col and matrix[row, col]:
                _apply_row_cx(matrix, col, row)
                reduce_ops.append(("cx", col, row))

    if not np.array_equal(matrix, np.eye(n, dtype=np.uint8)):
        raise AssertionError("linear CNOT synthesis failed to reduce matrix.")
    return list(reversed(reduce_ops))
def append_address_conditioned_xor(qc, intensity_qubits, y_qubits, x_qubits, key_matrix, q: int = 8) -> None:
    key_matrix = np.asarray(key_matrix, dtype=np.uint8)
    side = key_matrix.shape[0]
    address_qubits = list(y_qubits) + list(x_qubits)
    for y_value in range(side):
        for x_value in range(side):
            key_value = int(key_matrix[y_value, x_value])
            address_value = (y_value << int(log2(side))) | x_value
            for bit in range(q):
                if (key_value >> bit) & 1:
                    append_controlled_x_for_address(qc, address_qubits, address_value, intensity_qubits[q - 1 - bit])
def _append_cube_controlled_x(qc, source_qubits, target, cube) -> None:
    controls = []
    ctrl_state = []
    q = len(source_qubits)
    for source_index, bit in enumerate(range(q - 1, -1, -1)):
        value = cube.value_at(bit)
        if value is None:
            continue
        controls.append(source_qubits[source_index])
        ctrl_state.append(str(int(value)))

    if not controls:
        qc.x(target)
    elif len(controls) == 1:
        if ctrl_state[0] == "0":
            qc.x(controls[0])
        qc.cx(controls[0], target)
        if ctrl_state[0] == "0":
            qc.x(controls[0])
    else:
        qc.mcx(controls, target, ctrl_state="".join(reversed(ctrl_state)))
def _matrix_from_linear_ops(q: int, ops: list[tuple[str, int, int]]) -> np.ndarray:
    matrix = np.eye(q, dtype=np.uint8)
    for op, first, second in ops:
        if op == "swap":
            matrix[[first, second]] = matrix[[second, first]]
        else:
            _apply_row_cx(matrix, first, second)
    return matrix
PAPER_FRAMEWORK_INPUT_4X4 = np.array(
    [
        [12, 34, 56, 78],
        [90, 123, 145, 167],
        [201, 222, 17, 63],
        [88, 44, 132, 250],
    ],
    dtype=np.uint8,
)

APPENDIX_TOY_SBOX_4X4 = np.array(
    [
        [49, 168, 79, 219],
        [131, 147, 157, 26],
        [143, 242, 64, 78],
        [67, 192, 4, 240],
    ],
    dtype=np.uint8,
)


def paper_madd_component_gate(n: int, coeff: int = 1):
    """Opaque MADD block for paper-scale circuit diagrams."""
    _require_qiskit()
    if Gate is None:
        raise RuntimeError("Qiskit Gate is required for paper MADD components.")
    return Gate(f"MADD_{int(coeff)}_mod_2^{int(n)}", 2 * int(n), [])


def build_igqat_component_circuit(
    n: int,
    a: int = 11,
    b: int = 19,
    iterations: int = 1,
    name: str | None = None,
):
    """Build the paper-level IGQAT circuit with named MADD components."""
    _require_qiskit()
    if n < 1:
        raise ValueError("n must be positive.")
    y = QuantumRegister(n, "y")
    x = QuantumRegister(n, "x")
    qc = QuantumCircuit(y, x, name=name or f"IGQAT_MADD_n{n}")
    madd_a = paper_madd_component_gate(n, a)
    madd_b = paper_madd_component_gate(n, b)
    for _ in range(int(iterations)):
        qc.append(madd_a, list(y) + list(x))
        qc.append(madd_b, list(x) + list(y))
    qc.metadata = {
        "n": int(n),
        "a": int(a),
        "b": int(b),
        "iterations": int(iterations),
        "metrics": igqat_metrics_video_paper(n, iterations=max(1, int(iterations))) if n >= 2 else None,
    }
    return qc


def _byte_map_values(fn, q: int = 8) -> np.ndarray:
    return np.array([int(fn(value)) & ((1 << q) - 1) for value in range(1 << q)], dtype=np.uint16)


def csgc_byte_map_values(q: int = 8) -> np.ndarray:
    return _byte_map_values(
        lambda value: int(csgc_pre_diffusion(np.array([[value]], dtype=np.uint8), q=q)[0, 0]),
        q=q,
    )


def neqr_data_loading_map_values(image, q: int = 8) -> np.ndarray:
    """Truth table for the fixed-image NEQR coordinate-to-intensity loader."""
    image = validate_uint8_square_image(image)
    side = int(image.shape[0])
    n = int(log2(side))
    values = np.zeros(1 << (2 * n), dtype=np.uint16)
    for y_value in range(side):
        for x_value in range(side):
            address = (y_value << n) | x_value
            values[address] = int(image[y_value, x_value]) & ((1 << q) - 1)
    return values


def qat_coordinate_map_values(n: int, a: int = 11, b: int = 19, iterations: int = 2) -> np.ndarray:
    """Truth table for the paper/Qiskit QAT coordinate permutation."""
    side = 1 << int(n)
    values = np.zeros(1 << (2 * int(n)), dtype=np.uint16)
    for y_value in range(side):
        for x_value in range(side):
            y_new, x_new = classical_igqat_map(y_value, x_value, n, a=a, b=b, iterations=iterations)
            address = (y_value << int(n)) | x_value
            values[address] = (int(y_new) << int(n)) | int(x_new)
    return values

def append_boolean_values_to_target_register(
    qc,
    source_qubits,
    target_qubits,
    values,
    n_inputs: int,
    n_outputs: int,
    synthesis: str = "anf",
    quality: int = 2,
    max_dist: int = 4,
    max_steps: int = 1000,
    last_gasp_rounds: int = 1,
) -> dict:
    """Append an out-of-place Boolean map into ``target_qubits`` and return metrics."""
    values = np.asarray(values, dtype=np.uint16)
    synthesis = str(synthesis).lower()
    if synthesis == "anf":
        terms = esop_anf_terms_for_values(values, n_inputs=n_inputs, n_outputs=n_outputs)
        gate_count = append_esop_to_target_register(qc, source_qubits, target_qubits, terms)
        term_count = sum(len(items) for items in terms)
        literal_count = _anf_literal_count(terms)
        for value in range(1 << n_inputs):
            if evaluate_esop_terms(terms, value, n_bits=n_inputs, n_output_bits=n_outputs) != int(values[value]):
                raise AssertionError(f"ANF Boolean map mismatch at input {value}.")
        trace_summary = None
    elif synthesis == "exorcism4":
        terms, traces = esop_exorcism4_terms_for_values(
            values,
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            quality=quality,
            max_dist=max_dist,
            max_steps=max_steps,
            last_gasp_rounds=last_gasp_rounds,
        )
        gate_count = append_esop_cube_terms_to_target_register(qc, source_qubits, target_qubits, terms)
        term_count = sum(len(items) for items in terms)
        literal_count = _cube_literal_count(terms)
        for value in range(1 << n_inputs):
            if evaluate_esop_cube_terms(terms, value, n_bits=n_inputs, n_output_bits=n_outputs) != int(values[value]):
                raise AssertionError(f"EXORCISM-4 Boolean map mismatch at input {value}.")
        trace_summary = _summarize_exorcism_traces(traces)
    else:
        raise ValueError("synthesis must be 'anf' or 'exorcism4'.")

    metrics = {
        "synthesis": synthesis,
        "term_count": int(term_count),
        "literal_count": int(literal_count),
        "gate_count": int(gate_count),
        "n_inputs": int(n_inputs),
        "n_outputs": int(n_outputs),
        "quality": int(quality),
        "max_dist": int(max_dist),
        "max_steps": int(max_steps),
    }
    if trace_summary is not None:
        metrics["trace_summary"] = trace_summary
    return metrics
def _append_resource_row(rows: list[dict], name: str, qc, note: str = "") -> None:
    metrics = _circuit_basic_metrics(qc)
    rows.append(
        {
            "component": name,
            "qubits": metrics["qubits"],
            "depth": metrics["depth"],
            "ops": metrics["ops"],
            "note": note,
        }
    )
def _complete_depth_row(
    operation: str,
    without_qc,
    with_qc,
    without_details: dict | None = None,
    with_details: dict | None = None,
    note: str = "",
) -> dict:
    without_metrics = _circuit_basic_metrics(without_qc)
    with_metrics = _circuit_basic_metrics(with_qc)
    return {
        "operation": operation,
        "without_esop_depth": without_metrics["depth"],
        "with_esop_depth": with_metrics["depth"],
        "without_esop": without_metrics,
        "with_esop": with_metrics,
        "without_details": without_details or {},
        "with_details": with_details or {},
        "note": note,
    }
def _write_demo_markdown(
    outdir: Path,
    matrices: dict[str, np.ndarray],
    resource_rows: list[dict],
    esop_rows: list[dict],
) -> None:
    matrix_lines = ["# Paper Framework 4x4 Matrices", ""]
    for title, matrix in matrices.items():
        matrix_lines.extend([f"## {title}", "", _format_matrix_markdown(matrix), ""])
    (outdir / "matrices.md").write_text("\n".join(matrix_lines))

    resource_lines = [
        "# Qiskit Circuit Resource Table",
        "",
        "| Component | Qubits | Depth | Operations | Note |",
        "|---|---:|---:|---|---|",
    ]
    for row in resource_rows:
        resource_lines.append(
            f"| {row['component']} | {row['qubits']} | {row['depth']} | "
            f"{_ops_summary(row['ops'])} | {row['note']} |"
        )
    (outdir / "resource_table.md").write_text("\n".join(resource_lines) + "\n")

    esop_lines = [
        "# ESOP Optimization Comparison",
        "",
        "| Operations | Without ESOP | With ESOP |",
        "|---|---|---|",
    ]
    for row in esop_rows:
        esop_lines.append(f"| {row['operation']} | {row['without_esop']} | {row['with_esop']} |")
    (outdir / "esop_comparison_table.md").write_text("\n".join(esop_lines) + "\n")
    csgc_after = csgc_pre_diffusion(arnold_matrix.astype(np.uint8), q=8)
    csgc_stage_qc = QuantumCircuit(QuantumRegister(8, "f"), name="CSGC_stage")
    append_linear_transform_gate_level(csgc_stage_qc, list(csgc_stage_qc.qubits), csgc_matrix(8))
    circuits["csgc_stage"] = _write_circuit_artifacts(csgc_stage_qc, "csgc_stage", circuits_dir, qasm_dir)
    _append_resource_row(resource_rows, "CSGC stage", csgc_stage_qc, "Eq. (36)-(37) CNOT network")

    csgc_full_qc = arnold_qc.copy()
    append_linear_transform_gate_level(csgc_full_qc, list(neqr_info["intensity"]), csgc_matrix(8))
    csgc_recovered, _ = decode_neqr_statevector(csgc_full_qc, q=8, image_shape=image.shape)
    if not np.array_equal(csgc_recovered.astype(np.uint8), csgc_after):
        raise AssertionError("CSGC Qiskit stage does not match the GF(2) matrix.")

    key_matrix = generate_tln_diffusion_key_matrix(image.shape, key_material, q=8, warmup=1000, stride=11)
    tln_after = apply_tln_xor_diffusion(csgc_after, key_matrix)
    tln_f = QuantumRegister(8, "f")
    tln_y = QuantumRegister(2, "y")
    tln_x = QuantumRegister(2, "x")
    tln_stage_qc = QuantumCircuit(tln_f, tln_y, tln_x, name="TLN_diffusion_stage")
    append_linear_transform_gate_level(tln_stage_qc, list(tln_f), _gate_level_tln_xor_linear_matrix_q8())
    append_address_conditioned_xor(tln_stage_qc, list(tln_f), list(tln_y), list(tln_x), key_matrix, q=8)
    circuits["tln_diffusion_stage"] = _write_circuit_artifacts(
        tln_stage_qc,
        "tln_diffusion_stage",
        circuits_dir,
        qasm_dir,
    )
    _append_resource_row(resource_rows, "TLN diffusion stage", tln_stage_qc, "Eq. (42), address-conditioned key XOR")

    tln_full_qc = csgc_full_qc.copy()
    append_linear_transform_gate_level(tln_full_qc, list(neqr_info["intensity"]), _gate_level_tln_xor_linear_matrix_q8())
    append_address_conditioned_xor(
        tln_full_qc,
        list(neqr_info["intensity"]),
        list(neqr_info["y"]),
        list(neqr_info["x"]),
        key_matrix,
        q=8,
    )
    tln_recovered, _ = decode_neqr_statevector(tln_full_qc, q=8, image_shape=image.shape)
    if not np.array_equal(tln_recovered.astype(np.uint8), tln_after):
        raise AssertionError("TLN Qiskit stage does not match Eq. (42).")

    sbox_after = ESOP_INTENSITY_SBOX[tln_after].astype(np.uint8)
    sbox_f = QuantumRegister(8, "f")
    sbox_work = QuantumRegister(8, "sbox_work")
    sbox_stage_qc = QuantumCircuit(sbox_f, sbox_work, name="SBOX_q8_stage")
    sbox_metrics = append_sbox_esop_gate_level(
        sbox_stage_qc,
        list(sbox_f),
        list(sbox_work),
        ESOP_INTENSITY_SBOX,
        synthesis="anf",
    )
    circuits["sbox_q8_stage"] = _write_circuit_artifacts(sbox_stage_qc, "sbox_q8_stage", circuits_dir, qasm_dir)
    _append_resource_row(resource_rows, "q=8 S-box stage", sbox_stage_qc, "Eq. (49)-(59), ANF reversible evaluator")

    toy_input = np.arange(16, dtype=np.uint8).reshape(4, 4)
    toy_output = APPENDIX_TOY_SBOX_4X4.copy()
    toy_values = APPENDIX_TOY_SBOX_4X4.reshape(-1).astype(np.uint16)
    toy_qc, toy_metrics = build_esop_byte_map_circuit(
        toy_values,
        n_inputs=4,
        n_outputs=8,
        synthesis="exorcism4" if use_exorcism4 else "anf",
        quality=2,
        max_dist=4,
        max_steps=1000,
        name="Appendix_toy_SBOX",
    )
    for value in range(16):
        if int(toy_values[value]) != int(APPENDIX_TOY_SBOX_4X4.reshape(-1)[value]):
            raise AssertionError("Unexpected appendix toy S-box flattening mismatch.")
    circuits["appendix_toy_sbox"] = _write_circuit_artifacts(
        toy_qc,
        "appendix_toy_sbox",
        circuits_dir,
        qasm_dir,
    )
    _append_resource_row(resource_rows, "Appendix toy 4x4 S-box", toy_qc, "4-bit index to 8-bit output")

    esop_rows, esop_metrics = _build_esop_comparison_rows(
        outdir,
        use_exorcism4=use_exorcism4,
        circuits=circuits,
        image=image,
        arnold_iterations=arnold_iterations,
        arnold_r=arnold_r,
        arnold_z=arnold_z,
    )
    complete_depth_rows = _build_complete_depth_comparison_rows(
        image,
        key_matrix,
        use_exorcism4=use_exorcism4,
        arnold_iterations=arnold_iterations,
        arnold_r=arnold_r,
        arnold_z=arnold_z,
    )
    matrices = {
        "Input 4x4 matrix": image,
        "NEQR decoded matrix": neqr_matrix.astype(np.uint8),
        f"After improved Arnold MADD scrambling h={arnold_iterations}": arnold_matrix.astype(np.uint8),
        "After CSGC pre-diffusion": csgc_after,
        "TLN key matrix": key_matrix,
        "After TLN diffusion": tln_after,
        "After q=8 S-box substitution": sbox_after,
        "Appendix toy S-box input indices": toy_input,
        "Appendix toy S-box output matrix": toy_output,
    }
    _write_demo_markdown(outdir, matrices, resource_rows, esop_rows)

    metrics = {
        "paper_pdf": "main paper.pdf",
        "arnold": {
            "r": int(arnold_r),
            "z": int(arnold_z),
            "iterations": int(arnold_iterations),
            "demo_n": int(arnold_n),
            "n2_metrics": igqat_metrics_video_paper(2, arnold_iterations),
            "demo_n_metrics": igqat_metrics_video_paper(arnold_n, arnold_iterations) if arnold_n >= 2 else None,
        },
        "neqr_probabilities": neqr_probabilities,
        "matrices": matrices,
        "resource_rows": resource_rows,
        "sbox_metrics": sbox_metrics,
        "toy_sbox_metrics": toy_metrics,
        "esop_comparison": esop_rows,
        "esop_metrics": esop_metrics,
        "complete_depth_comparison": complete_depth_rows,
        "circuits": circuits,
    }
    (outdir / "metrics.json").write_text(json.dumps(_jsonable(metrics), indent=2, sort_keys=True))
    return metrics
@dataclass
class SvgCanvas:
    width: int
    height: int

    def __post_init__(self):
        self.items: list[str] = []

    def rect(self, x, y, w, h, fill="none", stroke="#1f2937", sw=1, opacity=1.0):
        self.items.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity:.3f}"/>'
        )

    def line(self, x1, y1, x2, y2, stroke="#111827", sw=1, opacity=1.0):
        self.items.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{stroke}" stroke-width="{sw}" opacity="{opacity:.3f}"/>'
        )

    def circle(self, cx, cy, r, fill="#111827", opacity=1.0):
        self.items.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{fill}" opacity="{opacity:.3f}"/>')

    def polyline(self, points, stroke="#111827", sw=1, fill="none", opacity=1.0):
        if len(points) < 2:
            return
        data = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.items.append(
            f'<polyline points="{data}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{sw}" opacity="{opacity:.3f}"/>'
        )

    def text(self, x, y, value, size=12, anchor="middle", fill="#111827", weight="400", rotate=None):
        value = html.escape(str(value))
        transform = f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate is not None else ""
        self.items.append(
            f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" font-family="Arial, sans-serif" '
            f'font-size="{size}" font-weight="{weight}" fill="{fill}"{transform}>{value}</text>'
        )

    def save(self, path: Path, title: str):
        body = "\n".join(self.items)
        path.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}">\n'
            f"<title>{html.escape(title)}</title>\n"
            f'<rect x="0" y="0" width="{self.width}" height="{self.height}" fill="#ffffff"/>\n'
            f"{body}\n</svg>\n"
        )
def _draw_line_panel(svg, x, y, w, h, xs, series, labels, title, xlabel, ylabel, colors=None):
    xs = np.asarray(xs, dtype=float)
    if colors is None:
        colors = ["#1d4ed8", "#dc2626", "#059669", "#7c3aed"]
    all_y = np.concatenate([np.asarray(yv, dtype=float) for yv in series if len(yv)])
    xmin, xmax = _finite_minmax(xs, pad=0.0)
    ymin, ymax = _finite_minmax(all_y)
    _draw_axes(svg, x, y, w, h, title, xlabel, ylabel)
    svg.line(x, _scale(0, ymin, ymax, y + h, y), x + w, _scale(0, ymin, ymax, y + h, y), stroke="#d1d5db", sw=1)
def make_figure_4(outdir: Path, quality: str) -> Path:
    steps = 3000 if quality == "quick" else 9000
    burn_in = 600 if quality == "quick" else 1800
    params = create_tln_parameters(d=2.0)
    time, trajectory = simulate_tln([-0.54, -0.1, 0.0, 0.0], params=params, steps=steps, burn_in=burn_in)
    t = time - time[0]

    svg = SvgCanvas(1120, 820)
    svg.text(560, 32, "Figure 4 reproduction: phase portraits and time series for d=2", size=18, weight="700")
    _draw_phase_panel(svg, 70, 80, 300, 220, trajectory[:, 0], trajectory[:, 1], "(a) x1-x2", "x1", "x2")
    _draw_phase_panel(svg, 430, 80, 300, 220, trajectory[:, 0], trajectory[:, 2], "(b) x1-v1", "x1", "v1", "#dc2626")
    _draw_phase_panel(svg, 790, 80, 300, 220, trajectory[:, 1], trajectory[:, 3], "(c) x2-v2", "x2", "v2", "#059669")
    _draw_line_panel(
        svg,
        80,
        410,
        960,
        260,
        t,
        [trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], trajectory[:, 3]],
        ["x1", "x2", "v1", "v2"],
        "(d) time series",
        "time",
        "state",
    )
    path = outdir / "figure_4_phase_portraits_timeseries_d2.svg"
    svg.save(path, "Figure 4 reproduction")
    return path
def make_figure_6(outdir: Path, quality: str) -> Path:
    params = create_tln_parameters(c=0.35, d=3.3)
    steps = 2600 if quality == "quick" else 8000
    burn_in = 600 if quality == "quick" else 1800
    _, chaotic = simulate_tln([4.0, -0.1, 0.0, 0.0], params=params, steps=steps, burn_in=burn_in)
    _, periodic = simulate_tln([1.8, -0.1, 0.0, 0.0], params=params, steps=steps, burn_in=burn_in)

    basin_n = 16 if quality == "quick" else 36
    x1_values = np.linspace(0.8, 4.3, basin_n)
    x2_values = np.linspace(-1.0, 0.8, basin_n)
    v1_values = np.linspace(-1.5, 1.5, basin_n)
    basin_x1_x2 = _basin_grid_x1_x2(x1_values, x2_values, params)
    basin_x1_v1 = _basin_grid_x1_v1(x1_values, v1_values, params)

    chaotic_maxima = local_maxima(chaotic[:, 0])
    periodic_maxima = local_maxima(periodic[:, 0])

    svg = SvgCanvas(1120, 840)
    svg.text(560, 32, "Figure 6 reproduction: coexisting attractors, basins, and frequency summary", size=18, weight="700")
    _draw_phase_panel(svg, 70, 80, 300, 220, chaotic[:, 0], chaotic[:, 1], "(a) chaotic attractor", "x1", "x2", "#dc2626")
    _draw_phase_panel(svg, 420, 80, 300, 220, periodic[:, 0], periodic[:, 1], "(b) periodic attractor", "x1", "x2", "#1d4ed8")
    _draw_heatmap(svg, 770, 80, 280, 220, basin_x1_x2, "(c) basin x1(0)-x2(0)", "x1(0)", "x2(0)", 0, 2)
    _draw_heatmap(svg, 70, 430, 300, 220, basin_x1_v1, "(d) basin x1(0)-v1(0)", "x1(0)", "v1(0)", 0, 2)

    labels = np.array([0, 1])
    counts = np.array([len(chaotic_maxima), len(periodic_maxima)], dtype=float)
    _draw_scatter_panel(svg, 430, 430, 280, 220, labels, counts, "(e) local-maximum counts", "attractor", "count", "#059669", radius=5)
    svg.text(465, 685, "0: chaotic init", size=10, anchor="start")
    svg.text(590, 685, "1: periodic init", size=10, anchor="start")

    path = outdir / "figure_6_coexisting_attractors_basins.svg"
    svg.save(path, "Figure 6 reproduction")
    return path


def make_figure_7(outdir: Path) -> Path:
    svg = SvgCanvas(1120, 520)
    svg.text(560, 32, "Figure 7 reproduction: circuit-level block schematic for the PWL-TLN model", size=18, weight="700")

    blocks = [
        (80, 110, "x1 integrator", "dx1/dt"),
        (330, 110, "PWL f(x1)", "alpha,beta"),
        (600, 110, "v1 memory", "dv1/dt"),
        (80, 300, "x2 integrator", "dx2/dt"),
        (330, 300, "PWL f(x2)", "alpha,beta"),
        (600, 300, "v2 memory", "dv2/dt"),
    ]
    for x, y, title, subtitle in blocks:
        svg.rect(x, y, 170, 70, fill="#eef2ff", stroke="#1e40af", sw=1.5)
        svg.text(x + 85, y + 30, title, size=13, weight="700")
        svg.text(x + 85, y + 52, subtitle, size=10, fill="#374151")

    arrows = [
        (250, 145, 330, 145, "x1"),
        (500, 145, 600, 145, "-d f(x1)"),
        (770, 145, 250, 125, "v1 feedback"),
        (250, 335, 330, 335, "x2"),
        (500, 335, 600, 335, "-d f(x2)"),
        (770, 335, 250, 315, "v2 feedback"),
        (500, 150, 80, 335, "b21 f(x1)"),
        (500, 340, 80, 145, "b12 f(x2)"),
    ]
    for x1, y1, x2, y2, label in arrows:
        svg.line(x1, y1, x2, y2, stroke="#111827", sw=1.2)
        svg.circle(x2, y2, 3, fill="#111827")
        svg.text((x1 + x2) / 2, (y1 + y2) / 2 - 7, label, size=9, fill="#374151")

    svg.rect(850, 145, 180, 150, fill="#f8fafc", stroke="#475569", sw=1.2)
    svg.text(940, 180, "Analog realization", size=13, weight="700")
    svg.text(940, 208, "op-amps, capacitors,", size=10)
    svg.text(940, 228, "resistors, transistors,", size=10)
    svg.text(940, 248, "current sources, +/-15V", size=10)
    svg.text(560, 480, "Block schematic derived from the TLN equations; component-level values are not specified in the draft text.", size=11)

    path = outdir / "figure_7_circuit_block_schematic.svg"
    svg.save(path, "Figure 7 reproduction")
    return path
def generate_section_2_2_figures(outdir="figures_2_2", quality: str = "quick") -> list[Path]:
    """Generate SVG reproductions of the figures mentioned up to Section 2.2."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = [
        make_figure_2(outdir, quality),
        make_figure_3(outdir, quality),
        make_figure_4(outdir, quality),
        make_figure_5(outdir, quality),
        make_figure_6(outdir, quality),
        make_figure_7(outdir),
        make_figure_8(outdir, quality),
    ]
    return paths


# ---------------------------------------------------------------------------
# Verification runners
# ---------------------------------------------------------------------------


def verify_neqr_under_10_qubits() -> dict:
    test_image = np.array([[1, 5], [3, 6]], dtype=np.uint8)
    qc_neqr, _ = build_neqr_circuit(test_image, q=3, measure=False)
    recovered_image, address_probabilities = decode_neqr_statevector(qc_neqr, q=3, image_shape=test_image.shape)

    assert qc_neqr.num_qubits < 10
    assert np.array_equal(recovered_image, test_image)
    return {
        "qubits": qc_neqr.num_qubits,
        "input": test_image,
        "recovered": recovered_image,
        "probabilities": address_probabilities,
    }


def verify_igqat_under_10_qubits() -> dict:
    igqat_test_image = np.array(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [1, 3, 5, 7],
            [6, 4, 2, 0],
        ],
        dtype=np.uint8,
    )

    qc_neqr_for_igqat, neqr_info_for_igqat = build_neqr_circuit(igqat_test_image, q=3, measure=False)
    qc_neqr_igqat = append_igqat_to_neqr(qc_neqr_for_igqat, neqr_info_for_igqat, a=1, b=1, iterations=1)
    expected_scrambled = classical_igqat_image(igqat_test_image, a=1, b=1, iterations=1)
    recovered_scrambled, scrambled_probabilities = decode_neqr_statevector(
        qc_neqr_igqat,
        q=3,
        image_shape=igqat_test_image.shape,
    )

    assert qc_neqr_igqat.num_qubits < 10
    assert np.array_equal(recovered_scrambled, expected_scrambled)

    for yy in range(igqat_test_image.shape[0]):
        for xx in range(igqat_test_image.shape[1]):
            assert classical_igqat_map(yy, xx, 2, a=1, b=1, iterations=1) == classical_igqat_direct_map(
                yy, xx, 2, a=1, b=1, iterations=1
            )

    return {
        "qubits": qc_neqr_igqat.num_qubits,
        "input": igqat_test_image,
        "expected": expected_scrambled,
        "recovered": recovered_scrambled,
        "probabilities": scrambled_probabilities,
        "madd_metrics": madd_metrics_s11433(n=2),
        "igqat_metrics": igqat_metrics_video_paper(n=2, iterations=1),
    }
def verify_encryption_section_2_4_2() -> dict:
    """Verify the Section 2.4.2 reversible encryption pipeline."""
    image = np.arange(64, dtype=np.uint8).reshape(8, 8)
    password = "section-2.4.2-test-key"
    salt = bytes.fromhex("00112233445566778899aabbccddeeff")
    nonce = bytes.fromhex("102132435465768798a9bacb")
    key_material = derive_section_2_4_1_key_material(password, salt=salt, nonce=nonce)

    scrambled = classical_igqat_image(image, a=11, b=19, iterations=3)
    assert np.array_equal(inverse_classical_igqat_image(scrambled, a=11, b=19, iterations=3), image)

    Mc = csgc_matrix(8)
    assert np.array_equal(gf2_inverse_matrix(Mc), gf2_matrix_power(Mc, 7))
    csgc_image = csgc_pre_diffusion(scrambled)
    assert np.array_equal(inverse_csgc_pre_diffusion(csgc_image), scrambled)

    key_matrix = generate_tln_diffusion_key_matrix(image.shape, key_material, warmup=64, stride=3)
    assert key_matrix.shape == image.shape
    assert key_matrix.dtype == np.uint8
    diffused = apply_tln_xor_diffusion(csgc_image, key_matrix)
    assert np.array_equal(inverse_tln_xor_diffusion(diffused, key_matrix), csgc_image)

    substituted = block_sbox_substitution_section_2_4_2(diffused, key_material, block_power=2)
    desubstituted = block_sbox_substitution_section_2_4_2(substituted, key_material, block_power=2, inverse=True)
    assert np.array_equal(desubstituted, diffused)
    block_sboxes = [_section_2_4_2_block_sbox(index, key_material) for index in range(4)]
    assert all(len(np.unique(sbox)) == 256 for sbox in block_sboxes)
    assert all(np.array_equal(sbox, ESOP_INTENSITY_SBOX) for sbox in block_sboxes)
    assert len({tuple(sbox.tolist()) for sbox in block_sboxes}) == 1

    cipher, metadata = encrypt_section_2_4_2_image(
        image,
        password,
        salt=salt,
        nonce=nonce,
        arnold_iterations=3,
        arnold_r=11,
        arnold_z=19,
        block_power=2,
    )
    decrypted = decrypt_section_2_4_2_image(cipher, password, metadata)
    repeated_cipher, repeated_metadata = encrypt_section_2_4_2_image(
        image,
        password,
        salt=salt,
        nonce=nonce,
        arnold_iterations=3,
        arnold_r=11,
        arnold_z=19,
        block_power=2,
    )
    cipher_nonce_changed, _ = encrypt_section_2_4_2_image(
        image,
        password,
        salt=salt,
        nonce=bytes.fromhex("112132435465768798a9bacb"),
        arnold_iterations=3,
        arnold_r=11,
        arnold_z=19,
        block_power=2,
    )

    assert np.array_equal(decrypted, image)
    assert np.array_equal(cipher, repeated_cipher)
    assert metadata == repeated_metadata
    assert not np.array_equal(cipher, image)
    assert not np.array_equal(cipher, cipher_nonce_changed)

    return {
        "image_shape": image.shape,
        "cipher_checksum": int(np.sum(cipher, dtype=np.uint64)),
        "key_checksum": int(np.sum(key_matrix, dtype=np.uint64)),
        "metadata": metadata,
        "round_trip_ok": True,
        "unique_block_sboxes": len({tuple(sbox.tolist()) for sbox in block_sboxes}),
    }


def verify_quantum_encryption_section_2_4_2() -> dict:
    """Verify the small-statevector quantum circuit against Section 2.4.2."""
    image = np.array([[12, 34], [56, 78]], dtype=np.uint8)
    password = "section-2.4.2-quantum-test-key"
    salt = bytes.fromhex("00112233445566778899aabbccddeeff")
    nonce = bytes.fromhex("102132435465768798a9bacb")

    qc, metadata, _, key_matrix = build_section_2_4_2_quantum_circuit(
        image,
        password,
        salt=salt,
        nonce=nonce,
        arnold_iterations=1,
        arnold_r=11,
        arnold_z=19,
        block_power=0,
        gate_level=False,
    )
    recovered, probabilities = decode_neqr_statevector(qc, q=8, image_shape=image.shape)
    recovered = recovered.astype(np.uint8)

    classical_cipher, classical_metadata = encrypt_section_2_4_2_image(
        image,
        password,
        salt=salt,
        nonce=nonce,
        arnold_iterations=1,
        arnold_r=11,
        arnold_z=19,
        block_power=0,
    )
    decrypted = decrypt_section_2_4_2_image(recovered, password, metadata)

    assert qc.num_qubits == 10
    assert qc.metadata["gate_level"] is False
    assert metadata == classical_metadata
    assert key_matrix.shape == image.shape
    assert np.array_equal(recovered, classical_cipher)
    assert np.array_equal(decrypted, image)
    assert np.allclose(probabilities, 0.25)

    gate_qc, gate_metadata, _, gate_key_matrix = build_section_2_4_2_quantum_circuit(
        image,
        password,
        salt=salt,
        nonce=nonce,
        arnold_iterations=1,
        arnold_r=11,
        arnold_z=19,
        block_power=0,
        gate_level=True,
    )
    gate_ops = dict(gate_qc.count_ops())
    assert gate_metadata == classical_metadata
    assert np.array_equal(gate_key_matrix, key_matrix)
    assert gate_qc.num_qubits == 18
    assert gate_qc.metadata["gate_level"] is True
    assert "unitary" not in gate_ops

    forward_terms = esop_anf_terms_for_truth(ESOP_INTENSITY_SBOX, n_bits=8)
    inverse_terms = esop_anf_terms_for_truth(ESOP_INVERSE_INTENSITY_SBOX, n_bits=8)
    for value in range(256):
        assert evaluate_esop_terms(forward_terms, value, n_bits=8) == int(ESOP_INTENSITY_SBOX[value])
        assert evaluate_esop_terms(inverse_terms, value, n_bits=8) == int(ESOP_INVERSE_INTENSITY_SBOX[value])

    if exact_pseudokro_cover_from_truth is not None and minimize_esop_with_trace is not None:
        toy_sbox = np.array([(value * 7 + 3) % 16 for value in range(16)], dtype=np.uint8)
        toy_terms, toy_traces = esop_exorcism4_terms_for_truth(toy_sbox, n_bits=4, quality=2, max_steps=500)
        for value in range(16):
            assert evaluate_esop_cube_terms(toy_terms, value, n_bits=4) == int(toy_sbox[value])
        assert sum(len(terms) for terms in toy_terms) > 0
        assert len(toy_traces) == 4

    for matrix_lsb in (csgc_matrix(8), _gate_level_tln_xor_linear_matrix_q8()):
        matrix_register = _register_matrix_from_lsb_matrix(matrix_lsb)
        ops = _synthesize_linear_cnot_ops(matrix_register)
        assert np.array_equal(_matrix_from_linear_ops(8, ops), matrix_register)

    return {
        "qubits": qc.num_qubits,
        "depth": qc.depth(),
        "ops": dict(qc.count_ops()),
        "statevector_gate_level": qc.metadata["gate_level"],
        "gate_level_qubits": gate_qc.num_qubits,
        "gate_level_depth": gate_qc.depth(),
        "gate_level_ops": gate_ops,
        "gate_level": gate_qc.metadata["gate_level"],
        "sbox_esop_metrics": gate_qc.metadata["sbox_esop_metrics"],
        "gate_level_structural_ok": True,
        "input": image,
        "cipher": recovered,
        "cipher_checksum": int(np.sum(recovered, dtype=np.uint64)),
        "round_trip_ok": True,
    }
def run_all_verifications(skip_qiskit: bool = False) -> dict:
    results = {}
    if not skip_qiskit:
        results["neqr"] = verify_neqr_under_10_qubits()
        results["igqat"] = verify_igqat_under_10_qubits()
        results["quantum_2_4_2"] = verify_quantum_encryption_section_2_4_2()
    results["tln_2_1"] = verify_tln_section_2_1()
    results["tln_2_2"] = verify_tln_section_2_2()
    results["sbox_2_3"] = verify_sbox_section_2_3()
    results["keygen_2_4_1"] = verify_keygen_section_2_4_1()
    results["encryption_2_4_2"] = verify_encryption_section_2_4_2()
    results["cloud_2_4_3"] = verify_cloud_section_2_4_3()
    return results


def print_verification_summary(results: dict) -> None:
    if "neqr" in results:
        neqr = results["neqr"]
        print(f"NEQR verification passed: {neqr['qubits']} qubits")
        print("Recovered NEQR image:")
        print(neqr["recovered"])

    if "igqat" in results:
        igqat = results["igqat"]
        print(f"IGQAT verification passed: {igqat['qubits']} qubits")
        print("Recovered IGQAT image:")
        print(igqat["recovered"])
        print("MADD metrics:", igqat["madd_metrics"])
        print("IGQAT metrics:", igqat["igqat_metrics"])

    if "quantum_2_4_2" in results:
        quantum_2_4_2 = results["quantum_2_4_2"]
        print(
            "Quantum Section 2.4.2 verification passed: "
            f"{quantum_2_4_2['qubits']} qubits "
            f"depth={quantum_2_4_2['depth']} "
            f"cipher_checksum={quantum_2_4_2['cipher_checksum']} "
            f"statevector_gate_level={quantum_2_4_2['statevector_gate_level']}"
        )
        print(
            "Gate-level Section 2.4.2 circuit built: "
            f"{quantum_2_4_2['gate_level_qubits']} qubits "
            f"depth={quantum_2_4_2['gate_level_depth']} "
            f"ops={quantum_2_4_2['gate_level_ops']}"
        )
        print(
            "Gate-level S-box ESOP terms: "
            f"forward={quantum_2_4_2['sbox_esop_metrics']['forward_esop_terms']} "
            f"inverse={quantum_2_4_2['sbox_esop_metrics']['inverse_esop_terms']}"
        )

    tln_2_1 = results["tln_2_1"]
    print("TLN Section 2.1 verification passed")
    print("Final TLN state:", np.round(tln_2_1["final_state"], 6))

    tln_2_2 = results["tln_2_2"]
    print("Table 1 verified:", tln_2_2["table_1_ok"])
    for row in tln_2_2["table_1_report"]:
        print(
            f"  b11={row['b11']:.2f} max_error={row['max_error']:.2e} "
            f"stability={row['stability']} computed={rounded_complex_list(row['computed'])}"
        )
    print("Table 2 verified:", tln_2_2["table_2_ok"])
    for row in tln_2_2["table_2_report"]:
        print(
            f"  d={row['d']:.2f} max_error={row['max_error']:.2e} "
            f"stability={row['stability']} computed={rounded_complex_list(row['computed'])}"
        )
    print("Section 2.2 trajectory shape:", tln_2_2["trajectory_shape"])
    print("Section 2.2 local maxima count:", tln_2_2["maxima_count"])

    sbox_2_3 = results["sbox_2_3"]
    active_metrics = sbox_2_3["active_metrics"]
    ref_metrics = sbox_2_3["reference_metrics"]
    print("S-box Section 2.3 verification passed")
    print(
        "Active S-box matched: "
        f"{sbox_2_3['active_name']} "
        f"S[0]={sbox_2_3['active_spot_checks']['sbox_0']} "
        f"S[46]={sbox_2_3['active_spot_checks']['sbox_46']} "
        f"S[255]={sbox_2_3['active_spot_checks']['sbox_255']}"
    )
    print(
        "Active metrics: "
        f"NL={active_metrics['nonlinearity']} "
        f"DAP={active_metrics['dap']:.6f} "
        f"LAP={active_metrics['lap']:.6f} "
        f"SACoff={active_metrics['sac_offset']:.6f} "
        f"FP={active_metrics['fixed_points']} "
        f"RFP={active_metrics['reverse_fixed_points']}"
    )
    print(
        "Retained T-QCOBLAH reference matched: "
        f"poly_idx={sbox_2_3['reference_poly_index']} "
        f"poly=0x{sbox_2_3['reference_poly']:03X} "
        f"b={sbox_2_3['reference_b']}"
    )
    print(
        "Reference metrics: "
        f"NL={ref_metrics['nonlinearity']} "
        f"DAP={ref_metrics['dap']:.6f} "
        f"LAP={ref_metrics['lap']:.6f} "
        f"SACoff={ref_metrics['sac_offset']:.6f} "
        f"FP={ref_metrics['fixed_points']} "
        f"RFP={ref_metrics['reverse_fixed_points']}"
    )

    keygen_2_4_1 = results["keygen_2_4_1"]
    print("Key generation Section 2.4.1 verification passed")
    print(
        "Derived TLN initial conditions:",
        np.round(keygen_2_4_1["initial_conditions"], 6),
    )

    encryption_2_4_2 = results["encryption_2_4_2"]
    print("Encryption Section 2.4.2 verification passed")
    print(
        f"Round trip={encryption_2_4_2['round_trip_ok']} "
        f"cipher_checksum={encryption_2_4_2['cipher_checksum']} "
        f"unique_block_sboxes={encryption_2_4_2['unique_block_sboxes']}"
    )

    cloud_2_4_3 = results["cloud_2_4_3"]
    print("Cloud workflow Section 2.4.3 verification passed")
    print(
        f"records={cloud_2_4_3['records']} "
        f"record_id={cloud_2_4_3['record_id']} "
        f"tokens={cloud_2_4_3['token_count']} "
        f"kem_provider={cloud_2_4_3['kem_provider']} "
        f"round_trip={cloud_2_4_3['round_trip_ok']} "
        f"tamper_rejected={cloud_2_4_3['tamper_rejected']}"
    )
if __name__ == "__main__":
    raise SystemExit(main())
