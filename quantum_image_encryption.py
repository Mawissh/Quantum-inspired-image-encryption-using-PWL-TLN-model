"""Implement Sections 1.1 through 2.4.3 of the ELSA draft in plain Python.

This file replaces the exploratory notebook code. It contains:
- NEQR encoding/decoding from Section 1.1.
- Improved generalized quantum Arnold transform helpers.
- TLN model implementation from Section 2.1.
- Numerical simulation, table verification, and figure reproduction for Section 2.2.
- T-QCOBLAH S-box reference construction from Section 2.3.
- ESOP Table 1 S-box substitution, key generation, encryption, and small
  quantum-circuit checks through Section 2.4.2.
- Cloud-assisted encrypted storage and retrieval workflow from Section 2.4.3.

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


# ---------------------------------------------------------------------------
# Section 1.1: NEQR
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Section 1.2: Improved generalized quantum Arnold transform
# ---------------------------------------------------------------------------


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


def madd_metrics_s11433(n: int) -> dict[str, int]:
    """PG1/TR2 modular-adder metrics from the s11433 arithmetic paper."""
    if n < 2:
        raise ValueError("The published closed-form MADD metrics are for n >= 2.")
    return {
        "PG1": n - 1,
        "TR2": n - 2,
        "XOR": 5 * n - 10,
        "T_depth": 6 * n - 9,
        "total_depth": 16 * n - 19,
        "ancilla": 0,
    }


def igqat_metrics_video_paper(n: int, iterations: int = 1) -> dict[str, int]:
    """Improved iterative Arnold metrics from the video-encryption paper."""
    if n < 2:
        raise ValueError("The published closed-form IGQAT metrics are for n >= 2.")
    h = int(iterations)
    return {
        "PG1": 2 * h * (n - 1),
        "TR2": 2 * h * (n - 2),
        "XOR": 2 * h * (5 * n - 10),
        "T_depth": (12 * n - 18) * h,
        "total_depth": 2 * h * (16 * n - 19),
        "ancilla": 0,
        "width": 2 * n,
    }


def igqat_step_matrix(a: int = 1, b: int = 1) -> np.ndarray:
    """Generalized Arnold matrix for one iteration."""
    return np.array([[1, int(a)], [int(b), int(a) * int(b) + 1]], dtype=object)


def matrix_power_mod_2x2(matrix, exponent: int, modulus: int) -> np.ndarray:
    """Fast 2x2 matrix power modulo ``modulus``."""
    result = np.eye(2, dtype=object)
    base = np.array(matrix, dtype=object) % modulus
    exp = int(exponent)

    while exp > 0:
        if exp & 1:
            result = (result @ base) % modulus
        base = (base @ base) % modulus
        exp >>= 1

    return result.astype(int)


def igqat_direct_matrix(n: int, a: int = 1, b: int = 1, iterations: int = 1) -> np.ndarray:
    """Non-iterative matrix A^h mod 2^n, matching the video-paper idea."""
    return matrix_power_mod_2x2(igqat_step_matrix(a, b), iterations, 2**n)


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


def classical_igqat_image(image, a: int = 1, b: int = 1, iterations: int = 1) -> np.ndarray:
    """Apply the IGQAT coordinate permutation to a classical image array."""
    image = np.asarray(image)
    if image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError("IGQAT expects a square 2D image.")
    if not _is_power_of_two(image.shape[0]):
        raise ValueError("IGQAT expects side length 2^n.")

    side = image.shape[0]
    n = int(log2(side))
    output = np.zeros_like(image)

    for y_value in range(side):
        for x_value in range(side):
            y_new, x_new = classical_igqat_map(y_value, x_value, n, a, b, iterations)
            output[y_new, x_new] = image[y_value, x_value]

    return output


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


# ---------------------------------------------------------------------------
# Section 2.1: TLN model
# ---------------------------------------------------------------------------


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


def tln_pwl_activation(x, params) -> np.ndarray:
    """Piece-wise linear activation: f(x) = 0.5*a*(|x+b| - |x-b|)."""
    alpha = params["alpha"]
    beta = params["beta"]
    return 0.5 * alpha * (np.abs(x + beta) - np.abs(x - beta))


def tln_pwl_derivative(x, params) -> np.ndarray:
    """Derivative of the PWL activation away from the breakpoints +/- beta."""
    alpha = params["alpha"]
    beta = params["beta"]
    x = np.asarray(x, dtype=float)
    return np.where(np.abs(x) < beta, alpha, 0.0)


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


def tln_jacobian(state, params) -> np.ndarray:
    """Jacobian matrix of the TLN flow at the given state."""
    x1, x2, _, _ = np.asarray(state, dtype=float)
    fp1 = float(tln_pwl_derivative(x1, params))
    fp2 = float(tln_pwl_derivative(x2, params))

    return np.array(
        [
            [-params["a1"] + params["b11"] * fp1, params["b12"] * fp2, 1.0, 0.0],
            [params["b21"] * fp1, -params["a2"] + params["b22"] * fp2, 0.0, 1.0],
            [-params["d"] * fp1, 0.0, -params["c"], 0.0],
            [0.0, -params["d"] * fp2, 0.0, -params["c"]],
        ],
        dtype=float,
    )


def tln_divergence(state, params) -> float:
    """Volume contraction rate Psi = trace(J)."""
    return float(np.trace(tln_jacobian(state, params)))


def rk4_step(rhs, state, dt: float, params) -> np.ndarray:
    """One fixed-step fourth-order Runge-Kutta update."""
    state = np.asarray(state, dtype=float)
    k1 = rhs(state, params)
    k2 = rhs(state + 0.5 * dt * k1, params)
    k3 = rhs(state + 0.5 * dt * k2, params)
    k4 = rhs(state + dt * k3, params)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_tln(initial_state, params=None, steps: int = 2000, dt: float = 0.005, burn_in: int = 0):
    """Simulate the TLN model and return time points and trajectory."""
    if params is None:
        params = create_tln_parameters()
    if steps < 1:
        raise ValueError("steps must be at least 1.")
    if burn_in < 0 or burn_in >= steps:
        raise ValueError("burn_in must satisfy 0 <= burn_in < steps.")

    trajectory = np.zeros((steps + 1, 4), dtype=float)
    trajectory[0] = np.asarray(initial_state, dtype=float)

    for index in range(steps):
        trajectory[index + 1] = rk4_step(tln_rhs, trajectory[index], dt, params)

    time = np.arange(steps + 1, dtype=float) * dt
    if burn_in:
        return time[burn_in:], trajectory[burn_in:]
    return time, trajectory


def tln_unit_interval_sequence(trajectory) -> np.ndarray:
    """Map a TLN trajectory to [0,1) by taking the fractional part of abs(state)."""
    return np.mod(np.abs(np.asarray(trajectory, dtype=float)), 1.0)


# ---------------------------------------------------------------------------
# Section 2.2: numerical simulation and table verification
# ---------------------------------------------------------------------------


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


def classify_tln_equilibrium(eigenvalues, tol: float = 1e-9) -> str:
    """Classify local stability from eigenvalue real parts."""
    eigenvalues = np.asarray(eigenvalues, dtype=complex)
    real_parts = np.real(eigenvalues)
    positive = int(np.sum(real_parts > tol))
    negative = int(np.sum(real_parts < -tol))
    has_complex_pair = bool(np.any(np.abs(np.imag(eigenvalues)) > tol))

    if positive == 0 and negative == len(eigenvalues):
        return "Stable focus" if has_complex_pair else "Stable node"
    if positive > 0 and negative > 0:
        return f"Saddle-focus (index {positive})" if positive > 1 else "Saddle-focus"
    if positive > 0:
        return "Unstable focus" if has_complex_pair else "Unstable node"
    return "Non-hyperbolic"


def best_eigenvalue_match_error(calculated, expected) -> tuple[float, list[complex]]:
    """Best max absolute error over all eigenvalue orderings."""
    calculated = tuple(complex(z) for z in calculated)
    expected = tuple(complex(z) for z in expected)
    best = float("inf")
    best_order = None

    for order in permutations(range(len(calculated))):
        error = max(abs(calculated[order[i]] - expected[i]) for i in range(len(expected)))
        if error < best:
            best = float(error)
            best_order = order

    return best, [calculated[i] for i in best_order]


def rounded_complex_list(values, decimals: int = 4) -> list[complex | float]:
    """Readable rounded complex list for printing table checks."""
    out = []
    for value in values:
        value = complex(value)
        real = round(value.real, decimals)
        imag = round(value.imag, decimals)
        out.append(complex(real, imag) if abs(imag) > 10 ** (-decimals) else real)
    return out


def compute_table_1_eigenvalues() -> list[dict]:
    """Recompute Table 1 by sweeping b11 with d fixed at 0.4."""
    rows = []
    for b11 in [0.10, 0.30, 0.50, 0.70, 0.90, 1.10]:
        eigs = tln_origin_eigenvalues(section_2_2_table_params(b11=b11, d=0.4))
        rows.append({"b11": b11, "eigenvalues": eigs, "stability": classify_tln_equilibrium(eigs)})
    return rows


def compute_table_2_eigenvalues() -> list[dict]:
    """Recompute Table 2 by sweeping d with b11 fixed at 0.5."""
    rows = []
    for d_value in [0.10, 0.20, 0.40, 0.60, 0.80, 1.00]:
        eigs = tln_origin_eigenvalues(section_2_2_table_params(b11=0.5, d=d_value))
        rows.append({"d": d_value, "eigenvalues": eigs, "stability": classify_tln_equilibrium(eigs)})
    return rows


EXPECTED_TABLE_1 = {
    0.10: [-0.9717 + 1.1486j, -0.9717 - 1.1486j, -0.0383 + 0.6286j, -0.0383 - 0.6286j],
    0.30: [-0.7090 + 1.1114j, -0.7090 - 1.1114j, 0.0290 + 0.5111j, 0.0290 - 0.5111j],
    0.50: [-0.5050 + 1.0368j, -0.5050 - 1.0368j, 0.2734 + 0j, 0.0366 + 0j],
    0.70: [1.0847 + 0j, -0.3668 + 0j, -0.3790 + 0.9666j, -0.3790 - 0.9666j],
    0.90: [1.7662 + 0j, -0.5228 + 0j, -0.3117 + 0.9199j, -0.3117 - 0.9199j],
    1.10: [2.4469 + 0j, -0.6170 + 0j, -0.2749 + 0.8912j, -0.2749 - 0.8912j],
}

EXPECTED_TABLE_2 = {
    0.10: [1.1570 + 0j, -0.8470 + 0j, -0.5050 + 0.2915j, -0.5050 - 0.2915j],
    0.20: [0.9760 + 0j, -0.6660 + 0j, -0.5050 + 0.6442j, -0.5050 - 0.6442j],
    0.40: [-0.5050 + 1.0368j, -0.5050 - 1.0368j, 0.2734 + 0j, 0.0366 + 0j],
    0.60: [-0.5050 + 1.3172j, -0.5050 - 1.3172j, 0.1550 + 0.8037j, 0.1550 - 0.8037j],
    0.80: [0.1550 + 1.1428j, 0.1550 - 1.1428j, -0.5050 + 1.5476j, -0.5050 - 1.5476j],
    1.00: [0.1550 + 1.4021j, 0.1550 - 1.4021j, -0.5050 + 1.7478j, -0.5050 - 1.7478j],
}


def verify_eigenvalue_table(rows, expected_by_parameter, parameter_name: str, tolerance: float = 7e-4):
    """Compare recomputed eigenvalues against manuscript values rounded to 4 decimals."""
    report = []
    all_ok = True

    for row in rows:
        parameter = row[parameter_name]
        max_error, matched = best_eigenvalue_match_error(row["eigenvalues"], expected_by_parameter[parameter])
        ok = max_error <= tolerance
        all_ok = all_ok and ok
        report.append(
            {
                parameter_name: parameter,
                "ok": ok,
                "max_error": max_error,
                "computed": matched,
                "reported": expected_by_parameter[parameter],
                "stability": row["stability"],
            }
        )

    return all_ok, report


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


def local_maxima(series) -> np.ndarray:
    """Return local maxima values from a one-dimensional time series."""
    series = np.asarray(series, dtype=float)
    if len(series) < 3:
        return np.array([], dtype=float)
    mask = (series[1:-1] > series[:-2]) & (series[1:-1] >= series[2:])
    return series[1:-1][mask]


def parameter_plane_lyapunov_grid(
    x_name: str,
    x_values,
    y_name: str,
    y_values,
    base_params=None,
    initial_state=(-0.54, -0.1, 0.0, 0.0),
    steps: int = 1000,
    dt: float = 0.005,
    burn_in: int = 200,
    continuation: bool = False,
    reverse_x: bool = False,
    reverse_y: bool = False,
) -> np.ndarray:
    """Compute a maximum-Lyapunov grid for a two-parameter Section 2.2 chart."""
    if base_params is None:
        base_params = create_tln_parameters()

    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    grid = np.zeros((len(y_values), len(x_values)), dtype=float)

    x_order = list(range(len(x_values)))
    y_order = list(range(len(y_values)))
    if reverse_x:
        x_order = x_order[::-1]
    if reverse_y:
        y_order = y_order[::-1]

    row_initial = np.asarray(initial_state, dtype=float)
    for row_index in y_order:
        current_state = row_initial.copy()
        for col_index in x_order:
            params = dict(base_params)
            params[x_name] = float(x_values[col_index])
            params[y_name] = float(y_values[row_index])
            grid[row_index, col_index] = estimate_max_lyapunov_tln(
                current_state,
                params,
                steps=steps,
                dt=dt,
                burn_in=burn_in,
            )
            if continuation:
                _, short_trajectory = simulate_tln(current_state, params=params, steps=max(50, steps // 4), dt=dt)
                current_state = short_trajectory[-1]
        if continuation:
            row_initial = current_state

    return grid


def lyapunov_grid_b11_d(
    b11_values,
    d_values,
    base_params=None,
    initial_state=(-0.54, -0.1, 0.0, 0.0),
    steps: int = 1000,
    dt: float = 0.005,
    burn_in: int = 200,
    continuation: bool = False,
    reverse_x: bool = False,
    reverse_y: bool = False,
) -> np.ndarray:
    """Compute the Fig. 2-style b11-d maximum-Lyapunov plane."""
    return parameter_plane_lyapunov_grid(
        "b11",
        b11_values,
        "d",
        d_values,
        base_params=base_params,
        initial_state=initial_state,
        steps=steps,
        dt=dt,
        burn_in=burn_in,
        continuation=continuation,
        reverse_x=reverse_x,
        reverse_y=reverse_y,
    )


def lyapunov_grid_c_d(
    c_values,
    d_values,
    base_params=None,
    initial_state=(-0.54, -0.1, 0.0, 0.0),
    steps: int = 1000,
    dt: float = 0.005,
    burn_in: int = 200,
    continuation: bool = False,
    reverse_x: bool = False,
    reverse_y: bool = False,
) -> np.ndarray:
    """Compute the Fig. 2-style c-d maximum-Lyapunov plane."""
    return parameter_plane_lyapunov_grid(
        "c",
        c_values,
        "d",
        d_values,
        base_params=base_params,
        initial_state=initial_state,
        steps=steps,
        dt=dt,
        burn_in=burn_in,
        continuation=continuation,
        reverse_x=reverse_x,
        reverse_y=reverse_y,
    )


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


# ---------------------------------------------------------------------------
# Section 2.3: S-box construction and reference tables
# ---------------------------------------------------------------------------


IRREDUCIBLE_POLYS_8 = (
    0x11B, 0x11D, 0x12B, 0x12D, 0x139, 0x13F, 0x14D, 0x15F,
    0x163, 0x165, 0x169, 0x171, 0x177, 0x17B, 0x187, 0x18B,
    0x18D, 0x19F, 0x1A3, 0x1A9, 0x1B1, 0x1BD, 0x1C3, 0x1CF,
    0x1D7, 0x1DD, 0x1E7, 0x1F3, 0x1F5,
)


ESOP_INTENSITY_SBOX = np.array(
    [
        185, 175, 84, 203, 252, 187, 72, 10, 50, 91, 218, 127, 142, 16, 41, 117,
        31, 160, 78, 115, 206, 9, 46, 116, 229, 11, 20, 104, 216, 129, 86, 102,
        109, 28, 76, 222, 87, 224, 119, 213, 83, 144, 81, 198, 106, 227, 0, 165,
        254, 1, 107, 47, 192, 158, 99, 92, 90, 19, 15, 197, 5, 173, 27, 234,
        243, 172, 63, 249, 95, 51, 24, 89, 93, 94, 164, 96, 108, 169, 245, 3,
        202, 204, 232, 151, 133, 178, 135, 8, 59, 180, 139, 34, 65, 240, 237, 168,
        77, 255, 33, 75, 207, 64, 223, 209, 215, 177, 23, 239, 37, 219, 212, 61,
        53, 193, 45, 57, 155, 38, 188, 214, 149, 154, 205, 124, 163, 73, 167, 181,
        13, 226, 140, 17, 210, 145, 26, 166, 58, 136, 211, 194, 208, 85, 79, 146,
        131, 143, 162, 18, 80, 251, 101, 110, 62, 67, 120, 40, 25, 201, 137, 29,
        70, 118, 200, 82, 113, 153, 36, 174, 2, 97, 39, 32, 170, 190, 182, 22,
        161, 56, 55, 233, 231, 88, 246, 176, 54, 126, 138, 242, 21, 183, 6, 244,
        250, 221, 157, 122, 152, 30, 179, 103, 195, 132, 199, 150, 14, 217, 189, 68,
        171, 121, 220, 125, 71, 228, 35, 134, 49, 123, 248, 4, 128, 66, 12, 112,
        48, 184, 141, 52, 156, 114, 186, 69, 225, 191, 236, 44, 60, 74, 235, 105,
        253, 100, 147, 241, 7, 247, 43, 148, 42, 159, 196, 130, 98, 230, 238, 111,
    ],
    dtype=np.uint16,
)


TQCOBLAH_REFERENCE_SBOX = np.array(
    [
        97, 90, 21, 58, 187, 91, 78, 238, 157, 3, 116, 32, 148, 243, 163, 75,
        72, 134, 193, 9, 129, 229, 222, 164, 104, 204, 2, 182, 246, 184, 94, 141,
        196, 165, 185, 84, 86, 30, 123, 83, 136, 5, 120, 124, 12, 254, 199, 27,
        65, 93, 221, 250, 160, 152, 22, 26, 200, 133, 7, 169, 190, 175, 98, 92,
        156, 15, 166, 226, 102, 23, 4, 218, 255, 181, 96, 61, 241, 239, 53, 188,
        168, 10, 145, 126, 107, 77, 192, 143, 177, 146, 137, 139, 6, 128, 170, 154,
        228, 47, 36, 244, 150, 151, 34, 178, 108, 211, 130, 171, 33, 85, 203, 17,
        118, 227, 35, 213, 106, 197, 76, 206, 87, 74, 28, 131, 251, 198, 69, 161,
        41, 119, 43, 209, 60, 138, 73, 82, 80, 216, 64, 247, 240, 8, 167, 29,
        232, 113, 140, 105, 0, 88, 127, 1, 249, 202, 194, 212, 62, 54, 172, 149,
        45, 110, 225, 100, 162, 224, 59, 48, 219, 125, 14, 231, 55, 81, 153, 70,
        39, 205, 56, 217, 201, 20, 50, 230, 11, 99, 233, 109, 214, 89, 121, 66,
        25, 176, 174, 173, 117, 24, 51, 42, 147, 132, 242, 223, 37, 210, 189, 248,
        234, 208, 135, 220, 18, 245, 183, 114, 191, 49, 101, 57, 236, 31, 16, 235,
        122, 44, 40, 71, 68, 46, 215, 207, 186, 159, 253, 180, 111, 144, 38, 52,
        158, 142, 63, 95, 155, 112, 115, 195, 67, 19, 103, 237, 79, 252, 13, 179,
    ],
    dtype=np.uint16,
)


@dataclass
class TQCOBLAHParameters:
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


_INVERSE_TABLE_CACHE: dict[tuple[int, int], np.ndarray] = {}


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


def gf2_repair_matrix(matrix) -> np.ndarray:
    """Sequentially repair a binary matrix into an invertible GF(2) matrix."""
    matrix = np.asarray(matrix, dtype=np.uint8).copy() & 1
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("GF(2) repair expects a square matrix.")

    n = matrix.shape[0]
    basis_rows: list[np.ndarray] = []
    for row in matrix:
        candidate = basis_rows + [row.copy()]
        if gf2_rank(np.vstack(candidate)) > len(basis_rows):
            basis_rows.append(row.copy())
        if len(basis_rows) == n:
            break

    for index in range(n):
        unit = np.zeros(n, dtype=np.uint8)
        unit[index] = 1
        candidate = basis_rows + [unit]
        if gf2_rank(np.vstack(candidate)) > len(basis_rows):
            basis_rows.append(unit)
        if len(basis_rows) == n:
            break

    repaired = np.vstack(basis_rows[:n]).astype(np.uint8)
    if not gf2_is_invertible(repaired):
        raise AssertionError("GF(2) matrix repair failed.")
    return repaired


def stochastic_binarize(values, rng) -> np.ndarray:
    """Apply Eq. (12)/(18): bit=1 if random d < X[j]."""
    values = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    return (rng.random(values.shape) < values).astype(np.uint8)


def deterministic_binarize(values) -> np.ndarray:
    """Threshold binarization used only for deterministic fixtures/tests."""
    return (np.asarray(values, dtype=float) >= 0.5).astype(np.uint8)


def decode_sbox_chromosome_bits(bits, n_bits: int = 8) -> tuple[np.ndarray, np.ndarray]:
    bits = np.asarray(bits, dtype=np.uint8).ravel() & 1
    expected = n_bits * n_bits + n_bits
    if len(bits) != expected:
        raise ValueError(f"Expected {expected} chromosome bits, got {len(bits)}.")
    A = bits[: n_bits * n_bits].reshape(n_bits, n_bits)
    b = bits[n_bits * n_bits :].copy()
    return A.astype(np.uint8), b.astype(np.uint8)


def encode_sbox_chromosome_bits(A, b) -> np.ndarray:
    return np.concatenate([np.asarray(A, dtype=np.uint8).reshape(-1), np.asarray(b, dtype=np.uint8).ravel()])


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


def _bits_lsb(value: int, width: int) -> np.ndarray:
    return np.array([(int(value) >> shift) & 1 for shift in range(width)], dtype=np.uint8)


def _int_from_lsb_bits(bits) -> int:
    value = 0
    for index, bit in enumerate(bits):
        value |= int(bit) << index
    return value


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


def sbox_from_chromosome(chromosome: SBoxChromosome, n_bits: int = 8) -> np.ndarray:
    return sbox_from_affine_inverse(chromosome.A, chromosome.b, chromosome.p_idx, n_bits=n_bits)


def validate_sbox(sbox, n_bits: int = 8) -> None:
    sbox = np.asarray(sbox, dtype=int)
    expected = list(range(1 << n_bits))
    if len(sbox) != (1 << n_bits):
        raise ValueError("S-box has the wrong length.")
    if sorted(sbox.tolist()) != expected:
        raise ValueError("S-box is not bijective.")


def inverse_sbox(sbox) -> np.ndarray:
    sbox = np.asarray(sbox, dtype=np.uint16)
    inverse = np.zeros_like(sbox)
    for index, value in enumerate(sbox):
        inverse[int(value)] = index
    return inverse


ESOP_INVERSE_INTENSITY_SBOX = inverse_sbox(ESOP_INTENSITY_SBOX)


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


def recover_reference_tqcoblah_chromosome() -> SBoxChromosome:
    """Recover the affine chromosome for the local notebook's T-QCOBLAH S-box."""
    p_idx = 3
    recovered = recover_affine_from_inverse_sbox(TQCOBLAH_REFERENCE_SBOX, p_idx=p_idx, n_bits=8)
    if recovered is None:
        raise AssertionError("Reference S-box is not inverse-affine for polynomial index 3.")
    A, b = recovered
    X = encode_sbox_chromosome_bits(A, b).astype(float)
    return SBoxChromosome(X=X, A=A, b=b, p_idx=p_idx)


def qobl_vector(X, rng, lb=0.0, ub=1.0) -> np.ndarray:
    """QOBL operator from Eq. (14)-(16)."""
    X = np.asarray(X, dtype=float)
    opposite = lb + ub - X
    midpoint = (lb + ub) / 2.0
    r = rng.random(X.shape)
    q = np.where(X < midpoint, midpoint + (opposite - midpoint) * r, opposite + (midpoint - opposite) * r)
    return np.clip(q, lb, ub)


def coobl_vector(X, best_X) -> np.ndarray:
    """COOBL operator from Eq. (17), with clipping to [0,1]."""
    return np.clip(2.0 * np.asarray(best_X, dtype=float) - np.asarray(X, dtype=float), 0.0, 1.0)


def tln_chaotic_values(count: int, seed: int = 7, warmup: int = 300, dt: float = 0.01) -> np.ndarray:
    """Generate TabuNext values from the Section 2.1 TLN model."""
    rng = np.random.default_rng(seed)
    state = np.array([-0.54, -0.1, 0.0, 0.0], dtype=float) + 0.01 * rng.standard_normal(4)
    params = create_tln_parameters()
    values = []

    for _ in range(warmup):
        state = rk4_step(tln_rhs, state, dt, params)
    while len(values) < count:
        state = rk4_step(tln_rhs, state, dt, params)
        values.append(float(np.mod(abs(state[0]), 1.0)))

    return np.asarray(values, dtype=float)


def _parity(value: int) -> int:
    return int(value).bit_count() & 1


def _fwht(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=int).copy()
    h = 1
    while h < len(values):
        for start in range(0, len(values), 2 * h):
            for index in range(start, start + h):
                x = values[index]
                y = values[index + h]
                values[index] = x + y
                values[index + h] = x - y
        h *= 2
    return values


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


def differential_approximation_probability(sbox, n_bits: int = 8) -> tuple[int, float]:
    """Maximum differential count and probability over nonzero input differences."""
    sbox = np.asarray(sbox, dtype=np.uint16)
    size = 1 << n_bits
    max_count = 0

    for input_diff in range(1, size):
        counts = np.zeros(size, dtype=np.uint16)
        for x in range(size):
            counts[int(sbox[x]) ^ int(sbox[x ^ input_diff])] += 1
        max_count = max(max_count, int(np.max(counts)))

    return max_count, float(max_count / size)


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


def update_moora_weights(
    chromosomes: list[SBoxChromosome],
    weights: dict[str, float],
    rate: float = 0.05,
    floor: float = 0.02,
) -> dict[str, float]:
    """Adaptive weight update from Eq. (21)-(22)."""
    targets = {
        "nonlinearity": 112.0,
        "sac_offset": 0.0,
        "bic_nonlinearity": 112.0,
        "bic_sac_deviation": 0.0,
        "dap": 4.0 / 256.0,
        "lap": 0.0625,
    }
    beneficial = {"nonlinearity", "bic_nonlinearity"}
    updated = {}

    for name, target in targets.items():
        vals = np.array([float(ch.metrics[name]) for ch in chromosomes], dtype=float)
        best = float(np.max(vals) if name in beneficial else np.min(vals))
        if target != 0:
            gap = abs(best - target) / abs(target)
        else:
            ref = max(float(np.max(np.abs(vals))), 1e-12)
            gap = abs(best) / ref
        updated[name] = max(weights[name] * (1.0 + rate * gap), floor)

    total = sum(updated.values())
    return {name: value / total for name, value in updated.items()}


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


def tournament_select(population: list[SBoxChromosome], rng, k: int) -> SBoxChromosome:
    indices = rng.choice(len(population), size=min(k, len(population)), replace=False)
    return max((population[int(index)] for index in indices), key=lambda ch: ch.fitness)


def crossover_chromosomes(
    parent_a: SBoxChromosome,
    parent_b: SBoxChromosome,
    rng,
    params: TQCOBLAHParameters,
) -> SBoxChromosome:
    """Row-wise affine crossover, uniform b crossover, random polynomial inheritance."""
    n = params.n_bits
    if rng.random() >= params.crossover_probability:
        source = parent_a if rng.random() < 0.5 else parent_b
        return make_sbox_chromosome(source.X.copy(), source.p_idx, rng=rng, n_bits=n, stochastic=False)

    Xa = parent_a.X.copy()
    Xb = parent_b.X.copy()
    child = Xa.copy()
    rows_a = Xa[: n * n].reshape(n, n)
    rows_b = Xb[: n * n].reshape(n, n)
    child_rows = child[: n * n].reshape(n, n)
    for row in range(n):
        child_rows[row] = rows_a[row] if rng.random() < 0.5 else rows_b[row]

    b_a = Xa[n * n :]
    b_b = Xb[n * n :]
    child[n * n :] = np.where(rng.random(n) < 0.5, b_a, b_b)
    p_idx = parent_a.p_idx if rng.random() < 0.5 else parent_b.p_idx
    return make_sbox_chromosome(child, p_idx, rng=rng, n_bits=n, stochastic=True)


def mutate_chromosome(chromosome: SBoxChromosome, rng, params: TQCOBLAHParameters) -> SBoxChromosome:
    X = chromosome.X.copy()
    mask = rng.random(X.shape) < params.mutation_probability
    X[mask] = 1.0 - X[mask]
    p_idx = chromosome.p_idx
    if rng.random() < params.mutation_probability:
        max_delta = max(1, params.n_polynomials // 5)
        delta = int(rng.integers(1, max_delta + 1))
        p_idx = (p_idx + delta) % params.n_polynomials
    return make_sbox_chromosome(X, p_idx, rng=rng, n_bits=params.n_bits, stochastic=True)


def apply_generation_jumping(
    population: list[SBoxChromosome],
    best: SBoxChromosome,
    mode: str,
    rng,
    params: TQCOBLAHParameters,
    weights: dict[str, float],
) -> list[SBoxChromosome]:
    """Adaptive QOBL/COOBL generation jumping from Section 2.3.4."""
    candidates = list(population)
    if mode == "QOBL":
        for chromosome in population:
            X = qobl_vector(chromosome.X, rng, lb=float(np.min(chromosome.X)), ub=float(np.max(chromosome.X)))
            p_choices = [idx for idx in range(params.n_polynomials) if idx != chromosome.p_idx]
            candidate = make_sbox_chromosome(X, int(rng.choice(p_choices)), rng=rng, n_bits=params.n_bits, stochastic=True)
            evaluate_chromosome(candidate, n_bits=params.n_bits, full_metrics=True)
            candidates.append(candidate)
    else:
        for chromosome in population:
            X = coobl_vector(chromosome.X, best.X)
            candidate = make_sbox_chromosome(X, chromosome.p_idx, rng=rng, n_bits=params.n_bits, stochastic=True)
            evaluate_chromosome(candidate, n_bits=params.n_bits, full_metrics=True)
            if candidate.fitness >= chromosome.fitness:
                candidates.append(candidate)

    scores = moora_scores(candidates, weights)
    for chromosome, score in zip(candidates, scores):
        chromosome.moora_score = float(score)
    return select_top_chromosomes(candidates, params.population_size)


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


# ---------------------------------------------------------------------------
# Section 2.4.1: Key generation procedure
# ---------------------------------------------------------------------------


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


def hkdf_sha256(ikm: bytes, info: bytes, length: int = 32, salt: bytes | None = None) -> bytes:
    """RFC 5869 HKDF-SHA256 used for Section 2.4.1 key separation."""
    ikm = _ensure_bytes(ikm, "ikm")
    info = _ensure_bytes(info, "info")
    if salt is None:
        salt = bytes(hashlib.sha256().digest_size)
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    output = b""
    previous = b""
    counter = 1
    while len(output) < length:
        previous = hmac.new(prk, previous + info + bytes([counter]), hashlib.sha256).digest()
        output += previous
        counter += 1
        if counter > 255:
            raise ValueError("HKDF length too large for SHA-256.")
    return output[:length]


def derive_section_2_4_1_subkeys(master_secret: bytes) -> tuple[bytes, bytes]:
    """Eq. (29): split Mmaster into KTLN and KAuth."""
    k_tln = hkdf_sha256(master_secret, b"TLNkey", length=32)
    k_auth = hkdf_sha256(master_secret, b"Authkey", length=32)
    return k_tln, k_auth


def sha256_counter_stream(k_tln: bytes, nonce: bytes, length_bytes: int) -> bytes:
    """Eq. (30)-(31): SHA256(KTLN || N || enc32(j)) counter stream."""
    k_tln = _check_fixed_length(k_tln, 32, "k_tln")
    nonce = _check_fixed_length(nonce, 12, "nonce")
    if length_bytes < 0:
        raise ValueError("length_bytes must be non-negative.")

    blocks = []
    block_count = (length_bytes + 31) // 32
    for counter in range(block_count):
        if counter > 0xFFFFFFFF:
            raise ValueError("counter exceeds 32-bit enc32(j) range.")
        counter_bytes = counter.to_bytes(4, "big")
        blocks.append(hashlib.sha256(k_tln + nonce + counter_bytes).digest())
    return b"".join(blocks)[:length_bytes]


def uint32_be_segments(stream: bytes, count: int) -> np.ndarray:
    """Split stream into unsigned big-endian 32-bit segments."""
    stream = bytes(stream)
    needed = 4 * int(count)
    if len(stream) < needed:
        raise ValueError(f"Need at least {needed} bytes for {count} uint32 segments.")
    return np.array([int.from_bytes(stream[4 * i : 4 * i + 4], "big") for i in range(count)], dtype=np.uint64)


def normalized_uint32_values(stream: bytes, count: int = 4) -> np.ndarray:
    """Eq. (32): vk=BytesToInt(R[k-1])/2^32."""
    return uint32_be_segments(stream, count).astype(float) / float(2**32)


def validate_chaotic_basin_bounds(bounds) -> np.ndarray:
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (4, 2):
        raise ValueError("chaotic basin bounds must have shape (4,2).")
    if not np.all(np.isfinite(bounds)):
        raise ValueError("chaotic basin bounds must be finite.")
    if not np.all(bounds[:, 0] < bounds[:, 1]):
        raise ValueError("each basin lower bound must be less than its upper bound.")
    return bounds


def project_to_chaotic_basin(normalized_values, bounds=DEFAULT_TLN_CHAOTIC_BASIN_BOUNDS) -> np.ndarray:
    """Eq. (33): xk(0)=Lk+(Uk-Lk)vk."""
    normalized_values = np.asarray(normalized_values, dtype=float)
    if normalized_values.shape != (4,):
        raise ValueError("normalized_values must contain exactly four values.")
    if not np.all((0.0 <= normalized_values) & (normalized_values < 1.0)):
        raise ValueError("normalized values must lie in [0,1).")
    bounds = validate_chaotic_basin_bounds(bounds)
    return bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * normalized_values


def derive_section_2_4_1_key_material(
    master_key,
    salt: bytes | None = None,
    nonce: bytes | None = None,
    basin_bounds=DEFAULT_TLN_CHAOTIC_BASIN_BOUNDS,
    iterations: int = 200_000,
    stream_length: int = 16,
) -> Section241KeyMaterial:
    """Run the complete Section 2.4.1 key-generation pipeline."""
    salt = secrets.token_bytes(16) if salt is None else _check_fixed_length(salt, 16, "salt")
    nonce = secrets.token_bytes(12) if nonce is None else _check_fixed_length(nonce, 12, "nonce")
    basin_bounds = validate_chaotic_basin_bounds(basin_bounds)
    stream_length = max(int(stream_length), 16)

    m_master = derive_master_key_pbkdf2_sha256(master_key, salt, iterations=iterations, length=32)
    k_tln, k_auth = derive_section_2_4_1_subkeys(m_master)
    stream = sha256_counter_stream(k_tln, nonce, stream_length)
    normalized = normalized_uint32_values(stream, count=4)
    initial_conditions = project_to_chaotic_basin(normalized, basin_bounds)

    return Section241KeyMaterial(
        salt=salt,
        nonce=nonce,
        master_key=m_master,
        k_tln=k_tln,
        k_auth=k_auth,
        stream=stream,
        normalized_values=normalized,
        initial_conditions=initial_conditions,
        basin_bounds=basin_bounds.copy(),
    )


def section_2_4_1_header(key_material: Section241KeyMaterial) -> dict[str, str]:
    """Public header fields needed by the receiver: salt P and nonce N."""
    return {"salt_hex": key_material.salt.hex(), "nonce_hex": key_material.nonce.hex()}


# ---------------------------------------------------------------------------
# Section 2.4.2: Encryption algorithm
# ---------------------------------------------------------------------------


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


def byte_linear_transform(values, matrix, q: int = 8) -> np.ndarray:
    """Apply a q-bit GF(2) matrix to each integer value, using LSB-first bits."""
    values = np.asarray(values, dtype=np.uint16)
    matrix = np.asarray(matrix, dtype=np.uint8) & 1
    if matrix.shape != (q, q):
        raise ValueError(f"matrix must have shape ({q},{q}).")
    bits = ((values[..., None] >> np.arange(q, dtype=np.uint16)) & 1).astype(np.uint8)
    out_bits = np.mod(bits @ matrix.T, 2).astype(np.uint16)
    out = np.zeros(values.shape, dtype=np.uint16)
    for bit in range(q):
        out |= out_bits[..., bit] << bit
    return out.astype(np.uint8)


def csgc_pre_diffusion(image, q: int = 8) -> np.ndarray:
    """Apply the CSGC pre-diffusion matrix Mc to the intensity bits."""
    return byte_linear_transform(image, csgc_matrix(q), q=q)


def inverse_csgc_pre_diffusion(image, q: int = 8) -> np.ndarray:
    """Reverse CSGC pre-diffusion using the GF(2) inverse of the printed Mc."""
    matrix = csgc_matrix(q)
    return byte_linear_transform(image, gf2_inverse_matrix(matrix), q=q)


def tln_xor_diffusion_matrix_q8() -> np.ndarray:
    """Binary matrix L for Eq. (42), excluding the XOR key bits."""
    matrix = np.zeros((8, 8), dtype=np.uint8)
    csgc = csgc_matrix(8)
    for basis_bit in range(8):
        f = np.zeros(8, dtype=np.uint8)
        f[basis_bit] = 1
        g = (csgc @ f) & 1
        v = np.array(
            [
                g[1],
                g[2],
                g[3],
                g[4] ^ f[0],
                g[5] ^ f[0],
                g[6],
                g[7] ^ f[0],
                g[0],
            ],
            dtype=np.uint8,
        )
        matrix[:, basis_bit] = v
    return matrix


def tln_diffusion_parameters() -> dict[str, float]:
    """PWL-TLN parameters stated in Section 2.4.2 for the XOR diffusion key."""
    return create_tln_parameters(
        a1=0.1,
        a2=0.1,
        b11=0.04,
        b12=0.5,
        b21=-1.0,
        b22=2.0,
        i1=1.1,
        i2=2.0,
        c=0.1,
        d=6.0,
        alpha=3.3,
        beta=0.3,
    )


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


def _section_2_4_2_block_sbox(
    block_index: int,
    key_material: Section241KeyMaterial | None = None,
    seed_base: int = 7,
) -> np.ndarray:
    """Fixed ESOP Table 1 intensity S-box used by the active block substitution."""
    _ = (block_index, key_material, seed_base)
    return ESOP_INTENSITY_SBOX.copy()


def block_index_for_coordinate(y: int, x: int, side: int, block_power: int) -> int:
    """H(yx)=y_p...y_{m-1} x_p...x_{m-1} from Eq. (48), as an integer."""
    m = int(log2(side))
    if not (0 <= block_power <= m):
        raise ValueError("block_power must satisfy 0 <= p <= m.")
    blocks_per_axis = 1 << (m - block_power)
    return (int(y) >> block_power) * blocks_per_axis + (int(x) >> block_power)


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

    out = np.zeros_like(image)
    blocks_per_axis = side // block_size
    for by in range(blocks_per_axis):
        for bx in range(blocks_per_axis):
            block_index = by * blocks_per_axis + bx
            if inverse:
                sbox = ESOP_INVERSE_INTENSITY_SBOX
            else:
                sbox = _section_2_4_2_block_sbox(block_index, key_material, seed_base=seed_base)
            y0, y1 = by * block_size, (by + 1) * block_size
            x0, x1 = bx * block_size, (bx + 1) * block_size
            out[y0:y1, x0:x1] = sbox[image[y0:y1, x0:x1]].astype(np.uint8)
    return out


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


def _reverse_bit_order_matrix(q: int) -> np.ndarray:
    matrix = np.zeros((q, q), dtype=np.uint8)
    for bit in range(q):
        matrix[bit, q - 1 - bit] = 1
    return matrix


def _register_matrix_from_lsb_matrix(matrix_lsb) -> np.ndarray:
    """Convert an LSB-first byte matrix to the MSB-first register convention."""
    matrix_lsb = np.asarray(matrix_lsb, dtype=np.uint8) & 1
    q = matrix_lsb.shape[0]
    reverse = _reverse_bit_order_matrix(q)
    return (reverse @ matrix_lsb @ reverse) & 1


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


def append_linear_transform_gate_level(qc, qubits, matrix_lsb) -> None:
    """Append a CNOT/SWAP network for a q-bit GF(2) linear transform."""
    matrix_register = _register_matrix_from_lsb_matrix(matrix_lsb)
    for op, first, second in _synthesize_linear_cnot_ops(matrix_register):
        if op == "swap":
            qc.swap(qubits[first], qubits[second])
        else:
            qc.cx(qubits[first], qubits[second])


def _gate_level_tln_xor_linear_matrix_q8() -> np.ndarray:
    matrix = np.zeros((8, 8), dtype=np.uint8)
    zero_key = np.array([[0]], dtype=np.uint8)
    for bit in range(8):
        pixel = np.array([[1 << bit]], dtype=np.uint8)
        out = int(apply_tln_xor_diffusion(pixel, zero_key)[0, 0])
        matrix[:, bit] = _bits_lsb(out, 8)
    return matrix


def append_controlled_x_for_address(qc, address_qubits, address_value: int, target) -> None:
    width = len(address_qubits)
    # Qiskit's ctrl_state string is interpreted in the little-endian order of
    # the control list, while the paper/NEQR address is written MSB first.
    ctrl_state = "".join(str(bit) for bit in reversed(_bits_msb(address_value, width)))
    qc.mcx(address_qubits, target, ctrl_state=ctrl_state)


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


def esop_anf_terms_for_values(values, n_inputs: int = 8, n_outputs: int = 8) -> list[list[int]]:
    """Canonical positive-polarity ESOP terms for each output bit.

    This is the algebraic normal form (Reed-Muller ESOP). Terms are masks over
    LSB-first input variables; mask 0 denotes the constant term.
    """
    values = np.asarray(values, dtype=np.uint16)
    size = 1 << n_inputs
    if len(values) != size:
        raise ValueError(f"Expected {size} truth-table values.")
    output_terms: list[list[int]] = []
    for output_bit in range(n_outputs):
        coeffs = np.array([(int(values[x]) >> output_bit) & 1 for x in range(size)], dtype=np.uint8)
        for bit in range(n_inputs):
            step = 1 << bit
            for start in range(size):
                if start & step:
                    coeffs[start] ^= coeffs[start ^ step]
        output_terms.append([mask for mask, coeff in enumerate(coeffs) if coeff])
    return output_terms


def esop_anf_terms_for_truth(values, n_bits: int = 8) -> list[list[int]]:
    return esop_anf_terms_for_values(values, n_inputs=n_bits, n_outputs=n_bits)


def evaluate_esop_terms(
    output_terms: list[list[int]],
    value: int,
    n_bits: int = 8,
    n_output_bits: int | None = None,
) -> int:
    """Evaluate ESOP terms generated by ``esop_anf_terms_for_truth``."""
    out = 0
    width = len(output_terms) if n_output_bits is None else int(n_output_bits)
    for output_bit, terms in enumerate(output_terms):
        bit_value = 0
        for mask in terms:
            if mask == 0 or (value & mask) == mask:
                bit_value ^= 1
        out |= bit_value << output_bit
    return out & ((1 << width) - 1)


def esop_exorcism4_terms_for_values(
    values,
    n_inputs: int = 8,
    n_outputs: int = 8,
    quality: int = 1,
    max_dist: int = 4,
    last_gasp_rounds: int = 1,
    max_steps: int = 1000,
) -> tuple[list[list[object]], list[list[object]]]:
    """Generate minimized ternary ESOP cubes with the local EXORCISM-4 implementation."""
    if exact_pseudokro_cover_from_truth is None or minimize_esop_with_trace is None:
        raise RuntimeError("exorcism4_minimizer.py is required for EXORCISM-4 ESOP synthesis.")
    values = np.asarray(values, dtype=np.uint16)
    size = 1 << n_inputs
    if len(values) != size:
        raise ValueError(f"Expected {size} truth-table values.")

    output_terms: list[list[object]] = []
    traces: list[list[object]] = []
    for output_bit in range(n_outputs):
        truth = tuple((int(values[x]) >> output_bit) & 1 for x in range(size))
        cover = exact_pseudokro_cover_from_truth(truth, n_inputs)
        minimized, trace = minimize_esop_with_trace(
            cover,
            quality=quality,
            max_dist=max_dist,
            verify=n_inputs <= 12,
            max_steps=max_steps,
            last_gasp_rounds=last_gasp_rounds,
        )
        output_terms.append(sorted(minimized.cubes))
        traces.append(trace)
    return output_terms, traces


def esop_exorcism4_terms_for_truth(
    values,
    n_bits: int = 8,
    quality: int = 1,
    max_dist: int = 4,
    last_gasp_rounds: int = 1,
    max_steps: int = 1000,
) -> tuple[list[list[object]], list[list[object]]]:
    return esop_exorcism4_terms_for_values(
        values,
        n_inputs=n_bits,
        n_outputs=n_bits,
        quality=quality,
        max_dist=max_dist,
        last_gasp_rounds=last_gasp_rounds,
        max_steps=max_steps,
    )


def evaluate_esop_cube_terms(
    output_terms: list[list[object]],
    value: int,
    n_bits: int = 8,
    n_output_bits: int | None = None,
) -> int:
    """Evaluate ternary ESOP cubes generated by ``esop_exorcism4_terms_for_truth``."""
    out = 0
    width = len(output_terms) if n_output_bits is None else int(n_output_bits)
    for output_bit, cubes in enumerate(output_terms):
        bit_value = 0
        for cube in cubes:
            if cube.matches(int(value)):
                bit_value ^= 1
        out |= bit_value << output_bit
    return out & ((1 << width) - 1)


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


def append_esop_cube_terms_to_target_register(qc, source_qubits, target_qubits, output_terms: list[list[object]]) -> int:
    """Append ternary ESOP-controlled X gates from ``source`` into ``target``."""
    q = len(source_qubits)
    if len(target_qubits) < len(output_terms):
        raise ValueError("target register is smaller than the ESOP output width.")
    gate_count = 0
    for output_bit, cubes in enumerate(output_terms):
        target = target_qubits[len(output_terms) - 1 - output_bit]
        for cube in cubes:
            _append_cube_controlled_x(qc, source_qubits, target, cube)
            gate_count += 1
    return gate_count


def _summarize_exorcism_traces(traces: list[list[object]]) -> list[dict[str, int]]:
    summaries = []
    for output_bit, trace in enumerate(traces):
        summaries.append(
            {
                "output_bit": output_bit,
                "attempts": sum(int(item.attempts) for item in trace),
                "accepted": sum(int(item.accepted) for item in trace),
                "cube_delta": sum(int(item.cube_delta) for item in trace),
                "literal_delta": sum(int(item.literal_delta) for item in trace),
            }
        )
    return summaries


def _matrix_from_linear_ops(q: int, ops: list[tuple[str, int, int]]) -> np.ndarray:
    matrix = np.eye(q, dtype=np.uint8)
    for op, first, second in ops:
        if op == "swap":
            matrix[[first, second]] = matrix[[second, first]]
        else:
            _apply_row_cx(matrix, first, second)
    return matrix


def append_esop_to_target_register(qc, source_qubits, target_qubits, output_terms: list[list[int]]) -> int:
    """Append ESOP-controlled X gates from ``source`` into ``target``."""
    q = len(source_qubits)
    if len(target_qubits) < len(output_terms):
        raise ValueError("target register is smaller than the ESOP output width.")
    gate_count = 0
    for output_bit, terms in enumerate(output_terms):
        target = target_qubits[len(output_terms) - 1 - output_bit]
        for mask in terms:
            if mask == 0:
                qc.x(target)
            else:
                controls = [source_qubits[q - 1 - bit] for bit in range(q) if (mask >> bit) & 1]
                if len(controls) == 1:
                    qc.cx(controls[0], target)
                else:
                    qc.mcx(controls, target)
            gate_count += 1
    return gate_count


def append_sbox_esop_gate_level(
    qc,
    intensity_qubits,
    work_qubits,
    sbox=ESOP_INTENSITY_SBOX,
    synthesis: str = "anf",
    exorcism_quality: int = 1,
    exorcism_max_dist: int = 4,
    exorcism_max_steps: int = 1000,
) -> dict[str, object]:
    """Paper-style reversible S-box: compute, clear by inverse, swap output back."""
    sbox = np.asarray(sbox, dtype=np.uint16)
    inverse = inverse_sbox(sbox)
    synthesis = str(synthesis).lower()

    if synthesis == "anf":
        forward_terms = esop_anf_terms_for_truth(sbox, n_bits=len(intensity_qubits))
        inverse_terms = esop_anf_terms_for_truth(inverse, n_bits=len(intensity_qubits))
        forward_gate_count = append_esop_to_target_register(qc, intensity_qubits, work_qubits, forward_terms)
        inverse_gate_count = append_esop_to_target_register(qc, work_qubits, intensity_qubits, inverse_terms)
        trace_summary = None
    elif synthesis == "exorcism4":
        forward_terms, forward_trace = esop_exorcism4_terms_for_truth(
            sbox,
            n_bits=len(intensity_qubits),
            quality=exorcism_quality,
            max_dist=exorcism_max_dist,
            max_steps=exorcism_max_steps,
        )
        inverse_terms, inverse_trace = esop_exorcism4_terms_for_truth(
            inverse,
            n_bits=len(intensity_qubits),
            quality=exorcism_quality,
            max_dist=exorcism_max_dist,
            max_steps=exorcism_max_steps,
        )
        forward_gate_count = append_esop_cube_terms_to_target_register(qc, intensity_qubits, work_qubits, forward_terms)
        inverse_gate_count = append_esop_cube_terms_to_target_register(qc, work_qubits, intensity_qubits, inverse_terms)
        trace_summary = {
            "forward": _summarize_exorcism_traces(forward_trace),
            "inverse": _summarize_exorcism_traces(inverse_trace),
        }
    else:
        raise ValueError("synthesis must be 'anf' or 'exorcism4'.")

    for left, right in zip(intensity_qubits, work_qubits):
        qc.swap(left, right)

    metrics = {
        "synthesis": synthesis,
        "forward_esop_terms": sum(len(terms) for terms in forward_terms),
        "inverse_esop_terms": sum(len(terms) for terms in inverse_terms),
        "forward_gates": forward_gate_count,
        "inverse_gates": inverse_gate_count,
        "swaps": len(intensity_qubits),
    }
    if trace_summary is not None:
        metrics["exorcism_trace_summary"] = trace_summary
    return metrics


def intensity_permutation_gate(mapping, q: int = 8, label: str = "IntensityPerm"):
    """Build a q-qubit permutation gate for an intensity-only byte map."""
    _require_qiskit()
    size = 1 << q
    matrix = np.zeros((size, size), dtype=complex)
    for input_index in range(size):
        input_bits = [(input_index >> position) & 1 for position in range(q)]
        input_value = _int_from_msb_bits(input_bits)
        output_value = int(mapping(input_value)) % size
        output_bits = _bits_msb(output_value, q)
        output_index = sum(int(bit) << position for position, bit in enumerate(output_bits))
        matrix[output_index, input_index] = 1.0
    return UnitaryGate(matrix, label=label)


def neqr_address_conditioned_intensity_gate(
    side: int,
    mapping,
    q: int = 8,
    label: str = "AddrIntensityPerm",
):
    """Build a full NEQR-basis gate mapping |f,y,x> -> |mapping(f,y,x),y,x>."""
    _require_qiskit()
    if not _is_power_of_two(side):
        raise ValueError("side must be a power of two.")
    n = int(log2(side))
    total_qubits = q + 2 * n
    size = 1 << total_qubits
    matrix = np.zeros((size, size), dtype=complex)

    for input_index in range(size):
        intensity, y_value, x_value = _neqr_basis_values(input_index, q=q, n=n)
        output_intensity = int(mapping(intensity, y_value, x_value)) % (1 << q)
        output_index = _neqr_basis_index(output_intensity, y_value, x_value, q=q, n=n)
        matrix[output_index, input_index] = 1.0

    return UnitaryGate(matrix, label=label)


def csgc_quantum_gate(q: int = 8):
    """Quantum unitary for Eq. (36)-(37) CSGC pre-diffusion."""
    return intensity_permutation_gate(
        lambda value: int(csgc_pre_diffusion(np.array([[value]], dtype=np.uint8), q=q)[0, 0]),
        q=q,
        label="CSGC",
    )


def tln_xor_diffusion_quantum_gate(key_matrix, q: int = 8):
    """Address-conditioned quantum unitary for Eq. (42)."""
    key_matrix = np.asarray(key_matrix, dtype=np.uint8)
    side = key_matrix.shape[0]
    if key_matrix.ndim != 2 or key_matrix.shape[0] != key_matrix.shape[1]:
        raise ValueError("key_matrix must be square.")

    def mapping(intensity, y_value, x_value):
        pixel = np.array([[intensity]], dtype=np.uint8)
        key = np.array([[int(key_matrix[y_value, x_value])]], dtype=np.uint8)
        return int(apply_tln_xor_diffusion(pixel, key)[0, 0])

    return neqr_address_conditioned_intensity_gate(side, mapping, q=q, label="TLN_XOR")


def block_sbox_quantum_gate(
    side: int,
    key_material: Section241KeyMaterial,
    block_power: int = 4,
    q: int = 8,
    seed_base: int = 7,
):
    """Address-conditioned quantum unitary for Eq. (49)-(59) block S-box substitution."""
    if q != 8:
        raise NotImplementedError("Block S-box quantum gate currently targets q=8.")
    sbox_cache = {}

    def mapping(intensity, y_value, x_value):
        block_index = block_index_for_coordinate(y_value, x_value, side, block_power)
        if block_index not in sbox_cache:
            sbox_cache[block_index] = _section_2_4_2_block_sbox(block_index, key_material, seed_base=seed_base)
        return int(sbox_cache[block_index][intensity])

    return neqr_address_conditioned_intensity_gate(side, mapping, q=q, label="Block_SBOX")


def build_section_2_4_2_quantum_circuit(
    image,
    master_key,
    salt: bytes | None = None,
    nonce: bytes | None = None,
    arnold_iterations: int = 1,
    arnold_r: int = 11,
    arnold_z: int = 19,
    block_power: int = 0,
    gate_level: bool = True,
) -> tuple[object, Section242Metadata, Section241KeyMaterial, np.ndarray]:
    """Build a small NEQR quantum-circuit simulation of Section 2.4.2."""
    image = validate_uint8_square_image(image)
    side = image.shape[0]
    key_material = derive_section_2_4_1_key_material(master_key, salt=salt, nonce=nonce)
    qc, neqr_info = build_neqr_circuit(image, q=8, measure=False)

    qc = append_igqat_to_neqr(
        qc,
        neqr_info,
        a=arnold_r,
        b=arnold_z,
        iterations=arnold_iterations,
        gate_level=gate_level,
    )
    key_matrix = generate_tln_diffusion_key_matrix(image.shape, key_material, q=8, warmup=1000, stride=11)
    all_neqr_qubits = list(neqr_info["intensity"]) + list(neqr_info["y"]) + list(neqr_info["x"])

    if gate_level:
        append_linear_transform_gate_level(qc, list(neqr_info["intensity"]), csgc_matrix(8))
        append_linear_transform_gate_level(qc, list(neqr_info["intensity"]), _gate_level_tln_xor_linear_matrix_q8())
        append_address_conditioned_xor(
            qc,
            list(neqr_info["intensity"]),
            list(neqr_info["y"]),
            list(neqr_info["x"]),
            key_matrix,
            q=8,
        )
        work = QuantumRegister(8, "sbox_work")
        qc.add_register(work)
        esop_metrics = append_sbox_esop_gate_level(qc, list(neqr_info["intensity"]), list(work), ESOP_INTENSITY_SBOX)
        qc.metadata = {
            "gate_level": True,
            "sbox_synthesis": esop_metrics["synthesis"],
            "sbox_esop_metrics": esop_metrics,
            "sbox_work": work,
        }
    else:
        qc.append(csgc_quantum_gate(q=8), list(neqr_info["intensity"]))
        qc.append(tln_xor_diffusion_quantum_gate(key_matrix, q=8), all_neqr_qubits)
        qc.append(block_sbox_quantum_gate(side, key_material, block_power=block_power, q=8), all_neqr_qubits)
        qc.metadata = {"gate_level": False}

    metadata = Section242Metadata(
        salt=key_material.salt,
        nonce=key_material.nonce,
        arnold_iterations=int(arnold_iterations),
        arnold_r=int(arnold_r),
        arnold_z=int(arnold_z),
        block_power=int(block_power),
    )
    return qc, metadata, key_material, key_matrix


# ---------------------------------------------------------------------------
# Paper-framework 4x4 Qiskit demonstration and artifact export
# ---------------------------------------------------------------------------


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


def tln_zero_key_byte_map_values() -> np.ndarray:
    zero_key = np.array([[0]], dtype=np.uint8)
    return _byte_map_values(
        lambda value: int(apply_tln_xor_diffusion(np.array([[value]], dtype=np.uint8), zero_key)[0, 0]),
        q=8,
    )


def _anf_literal_count(output_terms: list[list[int]]) -> int:
    return sum(int(mask).bit_count() for terms in output_terms for mask in terms)


def _cube_literal_count(output_terms: list[list[object]]) -> int:
    return sum(int(cube.n_lits()) for terms in output_terms for cube in terms)


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


def _circuit_basic_metrics(qc) -> dict:
    return {
        "qubits": int(qc.num_qubits),
        "depth": int(qc.depth() or 0),
        "ops": {str(key): int(value) for key, value in dict(qc.count_ops()).items()},
    }


def build_esop_byte_map_circuit(
    values,
    n_inputs: int,
    n_outputs: int,
    synthesis: str = "anf",
    quality: int = 2,
    max_dist: int = 4,
    max_steps: int = 1000,
    last_gasp_rounds: int = 1,
    name: str = "ESOP_map",
) -> tuple[object, dict]:
    """Build an out-of-place Boolean-map circuit and return resource metrics."""
    _require_qiskit()
    values = np.asarray(values, dtype=np.uint16)
    source = QuantumRegister(n_inputs, "src")
    target = QuantumRegister(n_outputs, "dst")
    qc = QuantumCircuit(source, target, name=name)
    synthesis = str(synthesis).lower()

    if synthesis == "anf":
        terms = esop_anf_terms_for_values(values, n_inputs=n_inputs, n_outputs=n_outputs)
        gate_count = append_esop_to_target_register(qc, list(source), list(target), terms)
        term_count = sum(len(items) for items in terms)
        literal_count = _anf_literal_count(terms)
        for value in range(1 << n_inputs):
            if evaluate_esop_terms(terms, value, n_bits=n_inputs, n_output_bits=n_outputs) != int(values[value]):
                raise AssertionError(f"ANF ESOP map mismatch at input {value}.")
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
        gate_count = append_esop_cube_terms_to_target_register(qc, list(source), list(target), terms)
        term_count = sum(len(items) for items in terms)
        literal_count = _cube_literal_count(terms)
        for value in range(1 << n_inputs):
            if evaluate_esop_cube_terms(terms, value, n_bits=n_inputs, n_output_bits=n_outputs) != int(values[value]):
                raise AssertionError(f"EXORCISM-4 ESOP map mismatch at input {value}.")
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
        **_circuit_basic_metrics(qc),
    }
    if trace_summary is not None:
        metrics["trace_summary"] = trace_summary
    return qc, metrics


def build_neqr_esop_circuit(
    image,
    q: int = 8,
    synthesis: str = "exorcism4",
    quality: int = 2,
    max_dist: int = 4,
    max_steps: int = 1000,
    name: str = "NEQR_ESOP",
):
    """Build a full NEQR state-preparation circuit with an ESOP-minimized loader."""
    _require_qiskit()
    image, n = _validate_neqr_image(image, q)
    side = image.shape[0]

    intensity = QuantumRegister(q, "f")
    y = QuantumRegister(n, "y")
    x = QuantumRegister(n, "x")
    qc = QuantumCircuit(intensity, y, x, name=name)
    qc.h(y)
    qc.h(x)

    loader_metrics = append_boolean_values_to_target_register(
        qc,
        list(y) + list(x),
        list(intensity),
        neqr_data_loading_map_values(image.astype(np.uint8), q=q),
        n_inputs=2 * n,
        n_outputs=q,
        synthesis=synthesis,
        quality=quality,
        max_dist=max_dist,
        max_steps=max_steps,
        last_gasp_rounds=1 if max_dist >= 4 else 0,
    )

    metadata = {
        "q": q,
        "n": n,
        "side": side,
        "intensity": intensity,
        "y": y,
        "x": x,
        "classical": None,
        "loader_metrics": loader_metrics,
    }
    return qc, metadata


def tln_key_map_values(key_matrix, q: int = 8) -> np.ndarray:
    """Truth table for the address-conditioned TLN key XOR matrix."""
    key_matrix = np.asarray(key_matrix, dtype=np.uint8)
    if key_matrix.ndim != 2 or key_matrix.shape[0] != key_matrix.shape[1]:
        raise ValueError("key_matrix must be square.")
    side = int(key_matrix.shape[0])
    if not _is_power_of_two(side):
        raise ValueError("key_matrix side length must be a power of two.")
    n = int(log2(side))
    values = np.zeros(1 << (2 * n), dtype=np.uint16)
    for y_value in range(side):
        for x_value in range(side):
            address = (y_value << n) | x_value
            values[address] = int(key_matrix[y_value, x_value]) & ((1 << q) - 1)
    return values


def build_tln_diffusion_stage_circuit(
    key_matrix,
    q: int = 8,
    use_esop_key_xor: bool = False,
    synthesis: str = "exorcism4",
    quality: int = 2,
    max_dist: int = 4,
    max_steps: int = 1000,
    name: str = "TLN_diffusion_stage",
):
    """Build the full Eq. (42) TLN stage, including address-conditioned key XOR."""
    _require_qiskit()
    key_matrix = np.asarray(key_matrix, dtype=np.uint8)
    side = int(key_matrix.shape[0])
    n = int(log2(side))
    tln_f = QuantumRegister(q, "f")
    tln_y = QuantumRegister(n, "y")
    tln_x = QuantumRegister(n, "x")
    qc = QuantumCircuit(tln_f, tln_y, tln_x, name=name)
    append_linear_transform_gate_level(qc, list(tln_f), _gate_level_tln_xor_linear_matrix_q8())
    key_metrics = None
    if use_esop_key_xor:
        key_metrics = append_boolean_values_to_target_register(
            qc,
            list(tln_y) + list(tln_x),
            list(tln_f),
            tln_key_map_values(key_matrix, q=q),
            n_inputs=2 * n,
            n_outputs=q,
            synthesis=synthesis,
            quality=quality,
            max_dist=max_dist,
            max_steps=max_steps,
            last_gasp_rounds=1 if max_dist >= 4 else 0,
        )
    else:
        append_address_conditioned_xor(qc, list(tln_f), list(tln_y), list(tln_x), key_matrix, q=q)
    return qc, key_metrics


def build_sbox_stage_circuit(
    sbox=ESOP_INTENSITY_SBOX,
    synthesis: str = "anf",
    quality: int = 0,
    max_dist: int = 1,
    max_steps: int = 1000,
    name: str = "SBOX_q8_stage",
):
    """Build the full reversible q=8 S-box stage used in the pipeline."""
    _require_qiskit()
    sbox_f = QuantumRegister(8, "f")
    sbox_work = QuantumRegister(8, "sbox_work")
    qc = QuantumCircuit(sbox_f, sbox_work, name=name)
    sbox_metrics = append_sbox_esop_gate_level(
        qc,
        list(sbox_f),
        list(sbox_work),
        sbox,
        synthesis=synthesis,
        exorcism_quality=quality,
        exorcism_max_dist=max_dist,
        exorcism_max_steps=max_steps,
    )
    return qc, sbox_metrics


def _format_matrix_markdown(matrix) -> str:
    matrix = np.asarray(matrix)
    header = "| row | " + " | ".join(f"c{index}" for index in range(matrix.shape[1])) + " |"
    sep = "|---|" + "|".join("---" for _ in range(matrix.shape[1])) + "|"
    rows = [header, sep]
    for index, row in enumerate(matrix.tolist()):
        rows.append("| " + str(index) + " | " + " | ".join(str(int(value)) for value in row) + " |")
    return "\n".join(rows)


def _ops_summary(ops: dict) -> str:
    if not ops:
        return "-"
    return ", ".join(f"{key}:{value}" for key, value in sorted(ops.items()))


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_circuit_artifacts(qc, name: str, circuits_dir: Path, qasm_dir: Path) -> dict:
    circuits_dir.mkdir(parents=True, exist_ok=True)
    qasm_dir.mkdir(parents=True, exist_ok=True)
    text_path = circuits_dir / f"{name}.txt"
    text_path.write_text(str(qc.draw(output="text", fold=-1)) + "\n")

    qasm_path = qasm_dir / f"{name}.qasm"
    qasm_written = False
    if qiskit_qasm2 is not None:
        try:
            qasm_path.write_text(qiskit_qasm2.dumps(qc))
            qasm_written = True
        except Exception as exc:
            qasm_path.write_text(f"// QASM export failed for {name}: {exc}\n")
    else:
        qasm_path.write_text("// QASM export unavailable: qiskit.qasm2 is not installed.\n")

    image_path = circuits_dir / f"{name}.png"
    image_written = False
    try:
        qc.draw(output="mpl", filename=str(image_path), fold=25)
        image_written = image_path.exists()
    except Exception:
        image_written = False

    return {
        "text": str(text_path),
        "qasm": str(qasm_path),
        "qasm_written": qasm_written,
        "image": str(image_path) if image_written else None,
    }


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


def _build_esop_comparison_rows(
    outdir: Path,
    use_exorcism4: bool,
    circuits: dict,
    image=None,
    arnold_iterations: int = 2,
    arnold_r: int = 11,
    arnold_z: int = 19,
) -> tuple[list[dict], dict]:
    if image is None:
        image = PAPER_FRAMEWORK_INPUT_4X4
    image = validate_uint8_square_image(image)
    side = int(image.shape[0])
    n = int(log2(side))
    comparisons = [
        ("NEQR encoding", neqr_data_loading_map_values(image, q=8), 2 * n, 8, 4, "fixed 4x4 coordinate-to-intensity loader"),
        (
            "QAT",
            qat_coordinate_map_values(n, a=arnold_r, b=arnold_z, iterations=arnold_iterations),
            2 * n,
            2 * n,
            4,
            f"4x4 h={int(arnold_iterations)} coordinate Boolean map",
        ),
        ("CSGC pre-diffusion", csgc_byte_map_values(8), 8, 8, 4, "CSGC intensity Boolean map"),
        ("TLN diffusion core", tln_zero_key_byte_map_values(), 8, 8, 4, "zero-key Eq. (42) Boolean core"),
        ("q8 S-box", ESOP_INTENSITY_SBOX.astype(np.uint16), 8, 8, 1, "bounded q=8 S-box minimization"),
        ("appendix toy S-box", APPENDIX_TOY_SBOX_4X4.reshape(-1).astype(np.uint16), 4, 8, 4, "4-bit index to byte output"),
    ]
    rows: list[dict] = []
    metrics_by_name = {}
    circuits_dir = outdir / "circuits"
    qasm_dir = outdir / "qasm"

    for label, values, n_inputs, n_outputs, exorcism_max_dist, note in comparisons:
        anf_qc, anf_metrics = build_esop_byte_map_circuit(
            values,
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            synthesis="anf",
            name=f"{label.replace(' ', '_')}_ANF",
        )
        artifact_name = label.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
        circuits[f"esop_{artifact_name}_anf"] = _write_circuit_artifacts(
            anf_qc,
            f"esop_{artifact_name}_anf",
            circuits_dir,
            qasm_dir,
        )

        if use_exorcism4:
            ex_qc, ex_metrics = build_esop_byte_map_circuit(
                values,
                n_inputs=n_inputs,
                n_outputs=n_outputs,
                synthesis="exorcism4",
                quality=2 if n_inputs <= 4 else 0,
                max_dist=exorcism_max_dist,
                max_steps=1000,
                last_gasp_rounds=1 if exorcism_max_dist >= 4 else 0,
                name=f"{label.replace(' ', '_')}_EXORCISM4",
            )
            circuits[f"esop_{artifact_name}_exorcism4"] = _write_circuit_artifacts(
                ex_qc,
                f"esop_{artifact_name}_exorcism4",
                circuits_dir,
                qasm_dir,
            )
            with_text = (
                f"EXORCISM-4 d<={ex_metrics['max_dist']} terms={ex_metrics['term_count']}, "
                f"lits={ex_metrics['literal_count']}, depth={ex_metrics['depth']}"
            )
            metrics_by_name[label] = {"without": anf_metrics, "with": ex_metrics, "note": note}
        else:
            with_text = "not generated; rerun with --demo-use-exorcism4"
            metrics_by_name[label] = {"without": anf_metrics, "with": None, "note": note}

        rows.append(
            {
                "operation": label,
                "without_esop": (
                    f"ANF terms={anf_metrics['term_count']}, "
                    f"lits={anf_metrics['literal_count']}, depth={anf_metrics['depth']}"
                ),
                "with_esop": with_text,
            }
        )

    return rows, metrics_by_name


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


def _build_complete_depth_comparison_rows(
    image,
    key_matrix,
    use_exorcism4: bool,
    arnold_iterations: int = 2,
    arnold_r: int = 11,
    arnold_z: int = 19,
) -> list[dict]:
    """Measure full component circuit depths for the manuscript table."""
    image = validate_uint8_square_image(image)
    side = int(image.shape[0])
    n = int(log2(side))

    neqr_without_qc, _ = build_neqr_circuit(image, q=8, measure=False, name="NEQR_4x4_without_ESOP")
    neqr_with_qc, neqr_with_info = build_neqr_esop_circuit(
        image,
        q=8,
        synthesis="exorcism4" if use_exorcism4 else "anf",
        quality=2,
        max_dist=4,
        max_steps=10000,
        name="NEQR_4x4_with_ESOP",
    )
    neqr_with_matrix, _ = decode_neqr_statevector(neqr_with_qc, q=8, image_shape=image.shape)
    if not np.array_equal(neqr_with_matrix.astype(np.uint8), image):
        raise AssertionError("ESOP NEQR stage did not recover the input image.")

    qat_without_qc = build_igqat_component_circuit(
        n,
        a=arnold_r,
        b=arnold_z,
        iterations=arnold_iterations,
        name="QAT_MADD_without_ESOP",
    )
    qat_with_qc = qat_without_qc.copy(name="QAT_MADD_with_ESOP")

    csgc_without_qc = QuantumCircuit(QuantumRegister(8, "f"), name="CSGC_without_ESOP")
    append_linear_transform_gate_level(csgc_without_qc, list(csgc_without_qc.qubits), csgc_matrix(8))
    csgc_with_qc = csgc_without_qc.copy(name="CSGC_with_ESOP")

    tln_without_qc, _ = build_tln_diffusion_stage_circuit(
        key_matrix,
        q=8,
        use_esop_key_xor=False,
        name="TLN_diffusion_without_ESOP",
    )
    tln_with_qc, tln_key_metrics = build_tln_diffusion_stage_circuit(
        key_matrix,
        q=8,
        use_esop_key_xor=use_exorcism4,
        synthesis="exorcism4" if use_exorcism4 else "anf",
        quality=2,
        max_dist=4,
        max_steps=10000,
        name="TLN_diffusion_with_ESOP",
    )

    sbox_without_qc, sbox_without_details = build_sbox_stage_circuit(
        ESOP_INTENSITY_SBOX,
        synthesis="anf",
        name="SBOX_q8_without_ESOP",
    )
    sbox_with_qc, sbox_with_details = build_sbox_stage_circuit(
        ESOP_INTENSITY_SBOX,
        synthesis="exorcism4" if use_exorcism4 else "anf",
        quality=0,
        max_dist=3,
        max_steps=10000,
        name="SBOX_q8_with_ESOP",
    )

    return [
        _complete_depth_row(
            "NEQR encoding",
            neqr_without_qc,
            neqr_with_qc,
            with_details=neqr_with_info.get("loader_metrics", {}),
            note="full NEQR preparation: Hadamards plus fixed image data loader",
        ),
        _complete_depth_row(
            "QAT",
            qat_without_qc,
            qat_with_qc,
            note="complete MADD component; no separate ESOP rewrite is applied to the opaque MADD block",
        ),
        _complete_depth_row(
            "CSGC",
            csgc_without_qc,
            csgc_with_qc,
            note="complete in-place CNOT/SWAP network; already linear",
        ),
        _complete_depth_row(
            "Chaotic XOR diffusion",
            tln_without_qc,
            tln_with_qc,
            with_details=tln_key_metrics or {},
            note="complete Eq. (42) stage; ESOP minimizes the address-conditioned key-XOR part",
        ),
        _complete_depth_row(
            "Substitution box",
            sbox_without_qc,
            sbox_with_qc,
            without_details=sbox_without_details,
            with_details=sbox_with_details,
            note="full reversible q=8 S-box: forward map, inverse cleanup, and swaps",
        ),
    ]


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


def build_paper_framework_demo(
    outdir="paper_framework_demo",
    arnold_n: int = 6,
    arnold_iterations: int = 2,
    arnold_r: int = 11,
    arnold_z: int = 19,
    use_exorcism4: bool = False,
) -> dict:
    """Generate the requested 4x4 paper-framework Qiskit demonstration bundle."""
    _require_qiskit()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    circuits_dir = outdir / "circuits"
    qasm_dir = outdir / "qasm"
    circuits: dict[str, dict] = {}
    resource_rows: list[dict] = []

    image = PAPER_FRAMEWORK_INPUT_4X4.copy()
    password = "paper-framework-demo-key"
    salt = bytes.fromhex("00112233445566778899aabbccddeeff")
    nonce = bytes.fromhex("102132435465768798a9bacb")
    key_material = derive_section_2_4_1_key_material(password, salt=salt, nonce=nonce)

    neqr_qc, neqr_info = build_neqr_circuit(image, q=8, measure=False, name="NEQR_4x4")
    neqr_matrix, neqr_probabilities = decode_neqr_statevector(neqr_qc, q=8, image_shape=image.shape)
    if not np.array_equal(neqr_matrix.astype(np.uint8), image):
        raise AssertionError("NEQR stage did not recover the input matrix.")
    circuits["neqr_4x4"] = _write_circuit_artifacts(neqr_qc, "neqr_4x4", circuits_dir, qasm_dir)
    _append_resource_row(resource_rows, "NEQR 4x4 encoding", neqr_qc, "q=8, n=2")

    arnold_component_n2 = build_igqat_component_circuit(2, a=arnold_r, b=arnold_z, iterations=arnold_iterations)
    circuits["arnold_madd_n2_component"] = _write_circuit_artifacts(
        arnold_component_n2,
        "arnold_madd_n2_component",
        circuits_dir,
        qasm_dir,
    )
    _append_resource_row(resource_rows, "Arnold MADD component n=2", arnold_component_n2, "paper-level MADD blocks")

    arnold_component_n = build_igqat_component_circuit(
        arnold_n,
        a=arnold_r,
        b=arnold_z,
        iterations=arnold_iterations,
        name=f"IGQAT_MADD_n{arnold_n}",
    )
    circuits[f"arnold_madd_n{arnold_n}_component"] = _write_circuit_artifacts(
        arnold_component_n,
        f"arnold_madd_n{arnold_n}_component",
        circuits_dir,
        qasm_dir,
    )
    _append_resource_row(
        resource_rows,
        f"Arnold MADD component n={arnold_n}",
        arnold_component_n,
        "manuscript-scale component diagram",
    )

    arnold_qc = append_igqat_to_neqr(
        neqr_qc,
        neqr_info,
        a=arnold_r,
        b=arnold_z,
        iterations=arnold_iterations,
        gate_level=False,
    )
    arnold_matrix, _ = decode_neqr_statevector(arnold_qc, q=8, image_shape=image.shape)
    expected_arnold = classical_igqat_image(image, a=arnold_r, b=arnold_z, iterations=arnold_iterations)
    if not np.array_equal(arnold_matrix.astype(np.uint8), expected_arnold):
        raise AssertionError("Arnold Qiskit stage does not match the classical IGQAT map.")
    for iteration in range(3):
        expected = classical_igqat_image(image, a=arnold_r, b=arnold_z, iterations=iteration)
        trial_qc = append_igqat_to_neqr(neqr_qc, neqr_info, a=arnold_r, b=arnold_z, iterations=iteration)
        recovered, _ = decode_neqr_statevector(trial_qc, q=8, image_shape=image.shape)
        if not np.array_equal(recovered.astype(np.uint8), expected):
            raise AssertionError(f"Arnold verification failed for iteration {iteration}.")
    circuits["neqr_plus_arnold_4x4"] = _write_circuit_artifacts(
        arnold_qc,
        "neqr_plus_arnold_4x4",
        circuits_dir,
        qasm_dir,
    )
    _append_resource_row(resource_rows, "NEQR + Arnold 4x4", arnold_qc, f"{arnold_iterations} MADD iterations")

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


# ---------------------------------------------------------------------------
# Section 2.4.3: cloud-assisted secure medical image storage and retrieval
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Section243KEMKeyPair:
    """Recipient KEM key material for the Section 2.4.3 workflow.

    The PDF specifies ML-KEM-768, implemented here with ``pqcrypto``.
    """

    public_key: bytes
    private_key: bytes
    provider: str = "ML-KEM-768"


@dataclass(frozen=True)
class Section243CloudRecord:
    """Cloud record R={Ienc,T,Ckem,Ckey,eta_aes,tau} from Algorithm 4."""

    record_id: str
    encrypted_image: np.ndarray
    token_set: tuple[bytes, ...]
    ckem: bytes
    encrypted_key_bundle: bytes
    aes_nonce: bytes
    image_tag: bytes
    kem_provider: str


def _require_cloud_crypto() -> None:
    if AESGCM is None:
        raise RuntimeError("cryptography is required for Section 2.4.3 AES-GCM workflows.")
    if ml_kem_768 is None:
        raise RuntimeError("pqcrypto is required for Section 2.4.3 ML-KEM-768 workflows.")


def normalize_search_keyword(keyword) -> bytes:
    """Norm(w): case-fold, trim, and collapse internal whitespace."""
    if isinstance(keyword, bytes):
        keyword = keyword.decode("utf-8")
    if not isinstance(keyword, str):
        raise TypeError("keyword must be str or bytes.")
    return " ".join(keyword.casefold().strip().split()).encode("utf-8")


def section_2_4_3_search_token(search_key, keyword) -> bytes:
    """Eq. (60): Ti=HMACSHA256(Ks, Norm(wi))."""
    search_key = _check_fixed_length(_ensure_bytes(search_key, "search_key"), 32, "search_key")
    return hmac.new(search_key, normalize_search_keyword(keyword), hashlib.sha256).digest()


def section_2_4_3_token_set(search_key, keywords) -> tuple[bytes, ...]:
    tokens = {section_2_4_3_search_token(search_key, keyword) for keyword in keywords}
    return tuple(sorted(tokens))


def section_2_4_3_generate_kem_keypair() -> Section243KEMKeyPair:
    """Generate the physician recipient key pair used by Algorithms 4 and 6."""
    _require_cloud_crypto()
    public_key, private_key = ml_kem_768.generate_keypair()
    return Section243KEMKeyPair(public_key=public_key, private_key=private_key)


def section_2_4_3_kem_encapsulate(
    public_key: bytes,
    ephemeral_private_key: bytes | None = None,
) -> tuple[bytes, bytes]:
    """Eq. (61): return ``(Ckem, Kshared)`` using ML-KEM-768."""
    _require_cloud_crypto()
    _ = ephemeral_private_key
    public_key = _check_fixed_length(public_key, ml_kem_768.PUBLIC_KEY_SIZE, "public_key")
    ckem, k_shared = ml_kem_768.encrypt(public_key)
    return ckem, k_shared


def section_2_4_3_kem_decapsulate(private_key: bytes, ckem: bytes) -> bytes:
    _require_cloud_crypto()
    private_key = _check_fixed_length(private_key, ml_kem_768.SECRET_KEY_SIZE, "private_key")
    ckem = _check_fixed_length(ckem, ml_kem_768.CIPHERTEXT_SIZE, "ckem")
    return ml_kem_768.decrypt(private_key, ckem)


def section_2_4_3_bundle_keys(k_shared: bytes) -> tuple[bytes, bytes]:
    """Eq. (62): derive Kaes and Kmac from Kshared."""
    k_shared = _ensure_bytes(k_shared, "k_shared")
    k_aes = hkdf_sha256(k_shared, b"QIE-cloud-key-bundle-v1", length=32)
    k_mac = hkdf_sha256(k_shared, b"QIE-cloud-image-mac-v1", length=32)
    return k_aes, k_mac


def section_2_4_2_metadata_to_dict(metadata: Section242Metadata) -> dict:
    return {
        "salt_hex": metadata.salt.hex(),
        "nonce_hex": metadata.nonce.hex(),
        "arnold_iterations": metadata.arnold_iterations,
        "arnold_r": metadata.arnold_r,
        "arnold_z": metadata.arnold_z,
        "block_power": metadata.block_power,
        "q": metadata.q,
        "diffusion_warmup": metadata.diffusion_warmup,
        "diffusion_stride": metadata.diffusion_stride,
    }


def section_2_4_2_metadata_from_dict(data: dict) -> Section242Metadata:
    return Section242Metadata(
        salt=bytes.fromhex(data["salt_hex"]),
        nonce=bytes.fromhex(data["nonce_hex"]),
        arnold_iterations=int(data["arnold_iterations"]),
        arnold_r=int(data["arnold_r"]),
        arnold_z=int(data["arnold_z"]),
        block_power=int(data["block_power"]),
        q=int(data.get("q", 8)),
        diffusion_warmup=int(data.get("diffusion_warmup", 1000)),
        diffusion_stride=int(data.get("diffusion_stride", 11)),
    )


def serialize_section_2_4_3_key_bundle(image_key: bytes, metadata: Section242Metadata) -> bytes:
    """B=Serialize(Kimg,Phi) from Algorithm 4."""
    bundle = {
        "image_key_hex": _ensure_bytes(image_key, "image_key").hex(),
        "metadata": section_2_4_2_metadata_to_dict(metadata),
    }
    return json.dumps(bundle, sort_keys=True, separators=(",", ":")).encode("utf-8")


def deserialize_section_2_4_3_key_bundle(bundle: bytes) -> tuple[bytes, Section242Metadata]:
    data = json.loads(_ensure_bytes(bundle, "bundle").decode("utf-8"))
    return bytes.fromhex(data["image_key_hex"]), section_2_4_2_metadata_from_dict(data["metadata"])


def _image_authentication_bytes(image) -> bytes:
    image = np.asarray(image, dtype=np.uint8)
    header = json.dumps(
        {"shape": list(image.shape), "dtype": str(image.dtype)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return header + b"\0" + image.tobytes(order="C")


def section_2_4_3_encrypt_and_upload(
    image,
    keywords,
    physician_public_key: bytes,
    search_key,
    cloud_database: list[Section243CloudRecord] | None = None,
    image_key: bytes | None = None,
    record_id: str | None = None,
    aes_nonce: bytes | None = None,
    salt: bytes | None = None,
    nonce: bytes | None = None,
    arnold_iterations: int = 59,
    arnold_r: int = 11,
    arnold_z: int = 19,
    block_power: int = 4,
    ephemeral_private_key: bytes | None = None,
) -> Section243CloudRecord:
    """Algorithm 4: image encryption and cloud upload."""
    _require_cloud_crypto()
    image = validate_uint8_square_image(image)
    image_key = secrets.token_bytes(32) if image_key is None else _ensure_bytes(image_key, "image_key")
    record_id = secrets.token_hex(16) if record_id is None else str(record_id)
    aes_nonce = secrets.token_bytes(12) if aes_nonce is None else _check_fixed_length(aes_nonce, 12, "aes_nonce")

    encrypted_image, metadata = encrypt_section_2_4_2_image(
        image,
        image_key,
        salt=salt,
        nonce=nonce,
        arnold_iterations=arnold_iterations,
        arnold_r=arnold_r,
        arnold_z=arnold_z,
        block_power=block_power,
    )
    bundle = serialize_section_2_4_3_key_bundle(image_key, metadata)
    tokens = section_2_4_3_token_set(search_key, keywords)
    ckem, k_shared = section_2_4_3_kem_encapsulate(physician_public_key, ephemeral_private_key=ephemeral_private_key)
    k_aes, k_mac = section_2_4_3_bundle_keys(k_shared)
    encrypted_key_bundle = AESGCM(k_aes).encrypt(aes_nonce, bundle, record_id.encode("utf-8"))
    image_tag = hmac.new(k_mac, _image_authentication_bytes(encrypted_image), hashlib.sha256).digest()

    record = Section243CloudRecord(
        record_id=record_id,
        encrypted_image=encrypted_image.astype(np.uint8),
        token_set=tokens,
        ckem=ckem,
        encrypted_key_bundle=encrypted_key_bundle,
        aes_nonce=aes_nonce,
        image_tag=image_tag,
        kem_provider="ML-KEM-768",
    )
    if cloud_database is not None:
        cloud_database.append(record)
    return record


def section_2_4_3_cloud_match(
    query_tokens,
    cloud_database,
    mode: str = "AND",
) -> list[Section243CloudRecord]:
    """Algorithm 5: cloud-side encrypted token matching."""
    query_set = set(query_tokens)
    mode = str(mode).upper()
    if mode not in {"AND", "OR"}:
        raise ValueError("mode must be 'AND' or 'OR'.")

    matches = []
    for record in cloud_database:
        stored = set(record.token_set)
        matched = query_set.issubset(stored) if mode == "AND" else bool(query_set & stored)
        if matched:
            matches.append(record)
    return matches


def section_2_4_3_physician_retrieve_and_decrypt(
    query_keywords,
    physician_private_key: bytes,
    search_key,
    cloud_database,
    mode: str = "AND",
    selection_index: int = 0,
) -> tuple[np.ndarray | None, Section243CloudRecord | None]:
    """Algorithm 6: physician-side retrieval and decryption."""
    _require_cloud_crypto()
    query_tokens = section_2_4_3_token_set(search_key, query_keywords)
    matches = section_2_4_3_cloud_match(query_tokens, cloud_database, mode=mode)
    if not matches:
        return None, None
    if not (0 <= selection_index < len(matches)):
        raise IndexError("selection_index is outside the matching record set.")

    record = matches[selection_index]
    k_shared = section_2_4_3_kem_decapsulate(physician_private_key, record.ckem)
    k_aes, k_mac = section_2_4_3_bundle_keys(k_shared)
    expected_tag = hmac.new(k_mac, _image_authentication_bytes(record.encrypted_image), hashlib.sha256).digest()
    if not hmac.compare_digest(expected_tag, record.image_tag):
        raise ValueError("Integrity verification failed.")

    bundle = AESGCM(k_aes).decrypt(record.aes_nonce, record.encrypted_key_bundle, record.record_id.encode("utf-8"))
    image_key, metadata = deserialize_section_2_4_3_key_bundle(bundle)
    recovered = decrypt_section_2_4_2_image(record.encrypted_image, image_key, metadata)
    return recovered, record


# ---------------------------------------------------------------------------
# Minimal SVG plotting helpers
# ---------------------------------------------------------------------------


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


def _finite_minmax(values, default=(-1.0, 1.0), pad=0.05):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return default
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if abs(vmax - vmin) < 1e-12:
        return vmin - 1.0, vmax + 1.0
    span = vmax - vmin
    return vmin - pad * span, vmax + pad * span


def _scale(value, src_min, src_max, dst_min, dst_max):
    if abs(src_max - src_min) < 1e-12:
        return 0.5 * (dst_min + dst_max)
    return dst_min + (float(value) - src_min) * (dst_max - dst_min) / (src_max - src_min)


def _color_map(value, vmin, vmax):
    if not np.isfinite(value):
        return "#9ca3af"
    t = max(0.0, min(1.0, _scale(value, vmin, vmax, 0.0, 1.0)))
    if t < 0.5:
        u = 2 * t
        r = int(37 + u * (245 - 37))
        g = int(99 + u * (245 - 99))
        b = int(235 + u * (245 - 235))
    else:
        u = 2 * (t - 0.5)
        r = int(245 + u * (220 - 245))
        g = int(245 + u * (38 - 245))
        b = int(245 + u * (38 - 245))
    return f"#{r:02x}{g:02x}{b:02x}"


def _draw_axes(svg: SvgCanvas, x, y, w, h, title, xlabel, ylabel):
    svg.rect(x, y, w, h, fill="#ffffff", stroke="#374151", sw=1)
    svg.text(x + w / 2, y - 8, title, size=13, weight="700")
    svg.text(x + w / 2, y + h + 34, xlabel, size=11)
    svg.text(x - 36, y + h / 2, ylabel, size=11, rotate=-90)


def _draw_heatmap(svg: SvgCanvas, x, y, w, h, grid, title, xlabel, ylabel, vmin=None, vmax=None):
    grid = np.asarray(grid, dtype=float)
    rows, cols = grid.shape
    if vmin is None or vmax is None:
        vmin, vmax = _finite_minmax(grid, pad=0.0)
    cell_w = w / cols
    cell_h = h / rows

    _draw_axes(svg, x, y, w, h, title, xlabel, ylabel)
    for row in range(rows):
        for col in range(cols):
            fill = _color_map(grid[row, col], vmin, vmax)
            svg.rect(x + col * cell_w, y + (rows - row - 1) * cell_h, cell_w + 0.2, cell_h + 0.2, fill=fill, stroke=fill)
    svg.rect(x, y, w, h, fill="none", stroke="#111827", sw=1)
    svg.text(x, y + h + 15, f"{vmin:.2f}", size=9, anchor="start", fill="#4b5563")
    svg.text(x + w, y + h + 15, f"{vmax:.2f}", size=9, anchor="end", fill="#4b5563")


def _draw_line_panel(svg, x, y, w, h, xs, series, labels, title, xlabel, ylabel, colors=None):
    xs = np.asarray(xs, dtype=float)
    if colors is None:
        colors = ["#1d4ed8", "#dc2626", "#059669", "#7c3aed"]
    all_y = np.concatenate([np.asarray(yv, dtype=float) for yv in series if len(yv)])
    xmin, xmax = _finite_minmax(xs, pad=0.0)
    ymin, ymax = _finite_minmax(all_y)
    _draw_axes(svg, x, y, w, h, title, xlabel, ylabel)
    svg.line(x, _scale(0, ymin, ymax, y + h, y), x + w, _scale(0, ymin, ymax, y + h, y), stroke="#d1d5db", sw=1)

    for i, values in enumerate(series):
        values = np.asarray(values, dtype=float)
        step = max(1, len(values) // 900)
        points = [
            (_scale(xs[j], xmin, xmax, x, x + w), _scale(values[j], ymin, ymax, y + h, y))
            for j in range(0, len(values), step)
            if np.isfinite(values[j])
        ]
        svg.polyline(points, stroke=colors[i % len(colors)], sw=1.2)
        if labels:
            svg.text(x + 10 + i * 55, y + 15, labels[i], size=9, anchor="start", fill=colors[i % len(colors)])


def _draw_scatter_panel(svg, x, y, w, h, xs, ys, title, xlabel, ylabel, color="#111827", radius=1.2):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    finite = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[finite]
    ys = ys[finite]
    xmin, xmax = _finite_minmax(xs, pad=0.0)
    ymin, ymax = _finite_minmax(ys)
    _draw_axes(svg, x, y, w, h, title, xlabel, ylabel)
    for x_value, y_value in zip(xs, ys):
        svg.circle(_scale(x_value, xmin, xmax, x, x + w), _scale(y_value, ymin, ymax, y + h, y), radius, fill=color, opacity=0.55)


def _draw_phase_panel(svg, x, y, w, h, x_values, y_values, title, xlabel, ylabel, color="#1d4ed8"):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    xmin, xmax = _finite_minmax(x_values)
    ymin, ymax = _finite_minmax(y_values)
    _draw_axes(svg, x, y, w, h, title, xlabel, ylabel)
    step = max(1, len(x_values) // 1200)
    points = [
        (_scale(x_values[i], xmin, xmax, x, x + w), _scale(y_values[i], ymin, ymax, y + h, y))
        for i in range(0, len(x_values), step)
        if np.isfinite(x_values[i]) and np.isfinite(y_values[i])
    ]
    svg.polyline(points, stroke=color, sw=0.8, opacity=0.85)


def _flatten_bifurcation_records(records):
    x_points = []
    y_points = []
    lyap_x = []
    lyap_y = []
    for record in records:
        parameter = record["parameter"]
        maxima = np.asarray(record["maxima_x1"], dtype=float)
        x_points.extend([parameter] * len(maxima))
        y_points.extend(maxima.tolist())
        lyap_x.append(parameter)
        lyap_y.append(record["max_lyapunov"])
    return np.asarray(x_points), np.asarray(y_points), np.asarray(lyap_x), np.asarray(lyap_y)


def _classify_basin_state(initial_state, params, steps=650, dt=0.005, burn_in=260):
    _, trajectory = simulate_tln(initial_state, params=params, steps=steps, dt=dt, burn_in=burn_in)
    if not np.all(np.isfinite(trajectory)) or np.max(np.abs(trajectory)) > 1e4:
        return 0
    maxima = local_maxima(trajectory[:, 0])
    if len(maxima) < 2:
        return 1
    return 2 if np.std(maxima[-min(8, len(maxima)) :]) > 0.02 else 1


def _basin_grid_x1_x2(x1_values, x2_values, params):
    grid = np.zeros((len(x2_values), len(x1_values)), dtype=float)
    for row, x2 in enumerate(x2_values):
        for col, x1 in enumerate(x1_values):
            grid[row, col] = _classify_basin_state([x1, x2, 0.0, 0.0], params)
    return grid


def _basin_grid_x1_v1(x1_values, v1_values, params):
    grid = np.zeros((len(v1_values), len(x1_values)), dtype=float)
    for row, v1 in enumerate(v1_values):
        for col, x1 in enumerate(x1_values):
            grid[row, col] = _classify_basin_state([x1, -0.1, v1, 0.0], params)
    return grid


# ---------------------------------------------------------------------------
# Figure reproduction for Section 2.2
# ---------------------------------------------------------------------------


def make_figure_2(outdir: Path, quality: str) -> Path:
    n = 14 if quality == "quick" else 32
    steps = 350 if quality == "quick" else 1400
    burn_in = 120 if quality == "quick" else 400
    base = create_tln_parameters()
    b11_values = np.linspace(0.05, 0.75, n)
    c_values = np.linspace(0.03, 0.65, n)
    d_values = np.linspace(1.0, 6.0, n)

    b11_d_up = lyapunov_grid_b11_d(
        b11_values, d_values, base_params=base, steps=steps, burn_in=burn_in, continuation=True
    )
    b11_d_down = lyapunov_grid_b11_d(
        b11_values,
        d_values,
        base_params=base,
        steps=steps,
        burn_in=burn_in,
        continuation=True,
        reverse_x=True,
        reverse_y=True,
    )
    c_d_up = lyapunov_grid_c_d(
        c_values, d_values, base_params=base, steps=steps, burn_in=burn_in, continuation=True
    )
    c_d_down = lyapunov_grid_c_d(
        c_values,
        d_values,
        base_params=base,
        steps=steps,
        burn_in=burn_in,
        continuation=True,
        reverse_x=True,
        reverse_y=True,
    )

    all_values = np.concatenate([b11_d_up.ravel(), b11_d_down.ravel(), c_d_up.ravel(), c_d_down.ravel()])
    vmin, vmax = _finite_minmax(all_values, pad=0.0)
    svg = SvgCanvas(1120, 760)
    svg.text(560, 32, "Figure 2 reproduction: two-parameter maximum Lyapunov maps", size=18, weight="700")
    _draw_heatmap(svg, 75, 80, 430, 250, b11_d_up, "(a) b11-d increasing", "b11", "d", vmin, vmax)
    _draw_heatmap(svg, 610, 80, 430, 250, b11_d_down, "(b) b11-d decreasing", "b11", "d", vmin, vmax)
    _draw_heatmap(svg, 75, 430, 430, 250, c_d_up, "(c) c-d increasing", "c", "d", vmin, vmax)
    _draw_heatmap(svg, 610, 430, 430, 250, c_d_down, "(d) c-d decreasing", "c", "d", vmin, vmax)
    svg.text(560, 727, "Blue: lower lambda_max, red: higher lambda_max. Generated by RK4 + Benettin estimate.", size=11)
    path = outdir / "figure_2_two_parameter_lyapunov.svg"
    svg.save(path, "Figure 2 reproduction")
    return path


def make_figure_3(outdir: Path, quality: str) -> Path:
    count = 32 if quality == "quick" else 90
    steps = 900 if quality == "quick" else 3200
    burn_in = 260 if quality == "quick" else 1000
    lyap_steps = 500 if quality == "quick" else 1800

    base_a = create_tln_parameters(b11=0.1)
    d_values = np.linspace(1.0, 6.0, count)
    d_scan = one_parameter_bifurcation_scan(
        "d", d_values, base_params=base_a, steps=steps, burn_in=burn_in, lyapunov_steps=lyap_steps
    )

    base_b = create_tln_parameters(d=4.0)
    c_values = np.linspace(0.03, 0.70, count)
    c_scan = one_parameter_bifurcation_scan(
        "c", c_values, base_params=base_b, steps=steps, burn_in=burn_in, lyapunov_steps=lyap_steps
    )

    d_x, d_y, d_lx, d_ly = _flatten_bifurcation_records(d_scan)
    c_x, c_y, c_lx, c_ly = _flatten_bifurcation_records(c_scan)

    svg = SvgCanvas(1120, 760)
    svg.text(560, 32, "Figure 3 reproduction: bifurcation diagrams and maximum Lyapunov traces", size=18, weight="700")
    _draw_scatter_panel(svg, 75, 80, 430, 230, d_x, d_y, "(a-i) local maxima of x1 vs d", "d", "max x1")
    _draw_line_panel(svg, 610, 80, 430, 230, d_lx, [d_ly], ["lambda"], "(a-ii) lambda_max vs d", "d", "lambda_max")
    _draw_scatter_panel(svg, 75, 430, 430, 230, c_x, c_y, "(b-i) local maxima of x1 vs c", "c", "max x1")
    _draw_line_panel(svg, 610, 430, 430, 230, c_lx, [c_ly], ["lambda"], "(b-ii) lambda_max vs c", "c", "lambda_max")
    path = outdir / "figure_3_bifurcation_lyapunov.svg"
    svg.save(path, "Figure 3 reproduction")
    return path


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


def make_figure_5(outdir: Path, quality: str) -> Path:
    steps = 4200 if quality == "quick" else 12000
    burn_in = 900 if quality == "quick" else 2500
    params = create_tln_parameters(d=2.0)
    time, trajectory = simulate_tln([-0.54, -0.1, 0.0, 0.0], params=params, steps=steps, burn_in=burn_in)
    t = time - time[0]

    svg = SvgCanvas(1120, 620)
    svg.text(560, 32, "Figure 5 reproduction: chaotic TLN signal used for phase portraits", size=18, weight="700")
    _draw_line_panel(
        svg,
        80,
        95,
        960,
        390,
        t,
        [trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], trajectory[:, 3]],
        ["x1", "x2", "v1", "v2"],
        "state variables",
        "time",
        "amplitude",
    )
    path = outdir / "figure_5_time_series_d2.svg"
    svg.save(path, "Figure 5 reproduction")
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


def make_figure_8(outdir: Path, quality: str) -> Path:
    steps = 3200 if quality == "quick" else 9000
    burn_in = 600 if quality == "quick" else 1800
    params = create_tln_parameters(d=2.0)
    time, trajectory = simulate_tln([-0.54, -0.1, 0.0, 0.0], params=params, steps=steps, burn_in=burn_in)
    t = time - time[0]

    svg = SvgCanvas(1120, 820)
    svg.text(560, 32, "Figure 8 reproduction: extended phase portraits and corresponding time series", size=18, weight="700")
    _draw_phase_panel(svg, 70, 80, 300, 220, trajectory[:, 0], trajectory[:, 3], "(a) x1-v2", "x1", "v2", "#7c3aed")
    _draw_phase_panel(svg, 430, 80, 300, 220, trajectory[:, 1], trajectory[:, 2], "(b) x2-v1", "x2", "v1", "#059669")
    _draw_phase_panel(svg, 790, 80, 300, 220, trajectory[:, 2], trajectory[:, 3], "(c) v1-v2", "v1", "v2", "#dc2626")
    _draw_line_panel(
        svg,
        80,
        410,
        960,
        260,
        t,
        [trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], trajectory[:, 3]],
        ["x1", "x2", "v1", "v2"],
        "(d) corresponding time series",
        "time",
        "state",
    )
    path = outdir / "figure_8_extended_phase_portraits_timeseries.svg"
    svg.save(path, "Figure 8 reproduction")
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


def verify_tln_section_2_1() -> dict:
    tln_params = create_tln_parameters()
    initial_state = np.array([-0.54, -0.10, 0.0, 0.0], dtype=float)
    _, trajectory = simulate_tln(initial_state, params=tln_params, steps=1000, dt=0.005, burn_in=0)

    activation_probe = np.linspace(-1.0, 1.0, 101)
    activation_values = tln_pwl_activation(activation_probe, tln_params)
    activation_bound = tln_params["alpha"] * tln_params["beta"]

    assert trajectory.shape == (1001, 4)
    assert np.all(np.isfinite(trajectory))
    assert np.max(np.abs(activation_values)) <= activation_bound + 1e-12

    return {
        "params": tln_params,
        "initial_state": initial_state,
        "final_state": trajectory[-1],
        "activation_bound": activation_bound,
        "initial_divergence": tln_divergence(initial_state, tln_params),
        "final_divergence": tln_divergence(trajectory[-1], tln_params),
        "final_eigenvalues": np.linalg.eigvals(tln_jacobian(trajectory[-1], tln_params)),
    }


def verify_tln_section_2_2() -> dict:
    table_1_rows = compute_table_1_eigenvalues()
    table_2_rows = compute_table_2_eigenvalues()
    table_1_ok, table_1_report = verify_eigenvalue_table(table_1_rows, EXPECTED_TABLE_1, "b11")
    table_2_ok, table_2_report = verify_eigenvalue_table(table_2_rows, EXPECTED_TABLE_2, "d")

    assert table_1_ok
    assert table_2_ok

    _, trajectory = simulate_tln(
        [-0.54, -0.1, 0.0, 0.0],
        params=create_tln_parameters(),
        steps=1500,
        dt=5e-3,
        burn_in=100,
    )
    maxima_x1 = local_maxima(trajectory[:, 0])
    grid_check = lyapunov_grid_b11_d([0.1, 0.2], [2.0, 2.2], steps=100, burn_in=20)
    bifurcation_check = one_parameter_bifurcation_scan(
        "b11",
        [0.1, 0.2],
        steps=300,
        burn_in=50,
        lyapunov_steps=100,
    )

    assert np.all(np.isfinite(trajectory))
    assert len(maxima_x1) > 0
    assert np.all(np.isfinite(grid_check))
    assert all(np.isfinite(row["max_lyapunov"]) for row in bifurcation_check)

    return {
        "table_1_ok": table_1_ok,
        "table_1_report": table_1_report,
        "table_2_ok": table_2_ok,
        "table_2_report": table_2_report,
        "trajectory_shape": trajectory.shape,
        "maxima_count": len(maxima_x1),
        "grid_check": grid_check,
        "bifurcation_records": len(bifurcation_check),
    }


def verify_sbox_section_2_3() -> dict:
    """Verify the active ESOP S-box and retained T-QCOBLAH reference data."""
    validate_sbox(ESOP_INTENSITY_SBOX, n_bits=8)
    assert np.array_equal(ESOP_INVERSE_INTENSITY_SBOX, inverse_sbox(ESOP_INTENSITY_SBOX))
    for value in range(256):
        assert int(ESOP_INVERSE_INTENSITY_SBOX[int(ESOP_INTENSITY_SBOX[value])]) == value
    assert int(ESOP_INTENSITY_SBOX[0]) == 185
    assert int(ESOP_INTENSITY_SBOX[46]) == 0
    assert int(ESOP_INTENSITY_SBOX[255]) == 111
    assert int(ESOP_INVERSE_INTENSITY_SBOX[0]) == 46
    assert int(ESOP_INVERSE_INTENSITY_SBOX[1]) == 49
    assert int(ESOP_INVERSE_INTENSITY_SBOX[255]) == 97

    esop_metrics = evaluate_sbox_metrics(ESOP_INTENSITY_SBOX, n_bits=8, full=True)
    assert esop_metrics["bijective"]
    assert esop_metrics["fixed_points"] == 0
    assert esop_metrics["reverse_fixed_points"] == 0
    assert esop_metrics["nonlinearity"] == 92
    assert esop_metrics["max_differential_count"] == 10
    assert np.isclose(esop_metrics["dap"], 10 / 256)
    assert np.isclose(esop_metrics["lap"], 0.140625)
    assert np.isclose(esop_metrics["sac_offset"], 0.036865234375)

    reference_chromosome = recover_reference_tqcoblah_chromosome()
    reference_sbox = sbox_from_chromosome(reference_chromosome, n_bits=8)
    assert np.array_equal(reference_sbox, TQCOBLAH_REFERENCE_SBOX)
    assert gf2_is_invertible(reference_chromosome.A)
    assert reference_chromosome.p_idx == 3
    assert IRREDUCIBLE_POLYS_8[reference_chromosome.p_idx] == 0x12D
    assert _int_from_lsb_bits(reference_chromosome.b) == 97

    reference_metrics = evaluate_sbox_metrics(reference_sbox, n_bits=8, full=True)
    reference_fitness = tqcoblah_fitness(reference_metrics, n_bits=8)
    assert reference_metrics["bijective"]
    assert reference_metrics["fixed_points"] == 0
    assert reference_metrics["reverse_fixed_points"] == 0
    assert reference_metrics["min_cycle"] == 256
    assert reference_metrics["cycle_count"] == 1
    assert reference_metrics["nonlinearity"] == 112
    assert reference_metrics["max_differential_count"] == 4
    assert np.isclose(reference_metrics["dap"], 4 / 256)
    assert np.isclose(reference_metrics["lap"], 0.0625)
    assert np.isclose(reference_metrics["sac_offset"], 0.021484375)

    chaotic_values = tln_chaotic_values(80, seed=7)
    assert chaotic_values.shape == (80,)
    assert np.all((0.0 <= chaotic_values) & (chaotic_values < 1.0))

    quick_params = TQCOBLAHParameters(
        population_size=6,
        generations=1,
        stagnation_threshold=3,
        jumping_rate=0.0,
        mutation_probability=0.04,
        crossover_probability=0.90,
        tournament_size=3,
        elite_fraction=0.05,
    )
    generated_sbox, best_chromosome = run_tqcoblah_sbox(seed=7, params=quick_params)
    generated_metrics = evaluate_sbox_metrics(generated_sbox, n_bits=8, full=False)
    assert generated_metrics["bijective"]
    assert gf2_is_invertible(best_chromosome.A)
    assert 0 <= best_chromosome.p_idx < len(IRREDUCIBLE_POLYS_8)

    return {
        "active_name": "ESOP Table 1 intensity S-box",
        "active_metrics": esop_metrics,
        "active_spot_checks": {
            "sbox_0": int(ESOP_INTENSITY_SBOX[0]),
            "sbox_46": int(ESOP_INTENSITY_SBOX[46]),
            "sbox_255": int(ESOP_INTENSITY_SBOX[255]),
            "inverse_0": int(ESOP_INVERSE_INTENSITY_SBOX[0]),
            "inverse_1": int(ESOP_INVERSE_INTENSITY_SBOX[1]),
            "inverse_255": int(ESOP_INVERSE_INTENSITY_SBOX[255]),
        },
        "reference_poly_index": reference_chromosome.p_idx,
        "reference_poly": IRREDUCIBLE_POLYS_8[reference_chromosome.p_idx],
        "reference_b": _int_from_lsb_bits(reference_chromosome.b),
        "reference_metrics": reference_metrics,
        "reference_fitness": reference_fitness,
        "generated_poly_index": best_chromosome.p_idx,
        "generated_metrics": generated_metrics,
    }


def verify_keygen_section_2_4_1() -> dict:
    """Verify the Section 2.4.1 key derivation and TLN projection pipeline."""
    password = "section-2.4.1-test-key"
    salt = bytes.fromhex("00112233445566778899aabbccddeeff")
    nonce = bytes.fromhex("102132435465768798a9bacb")

    material = derive_section_2_4_1_key_material(password, salt=salt, nonce=nonce)
    repeated = derive_section_2_4_1_key_material(password, salt=salt, nonce=nonce)
    nonce_changed = derive_section_2_4_1_key_material(password, salt=salt, nonce=bytes.fromhex("112132435465768798a9bacb"))
    salt_changed = derive_section_2_4_1_key_material(password, salt=bytes.fromhex("10112233445566778899aabbccddeeff"), nonce=nonce)

    assert material.salt == salt
    assert material.nonce == nonce
    assert len(material.master_key) == 32
    assert len(material.k_tln) == 32
    assert len(material.k_auth) == 32
    assert material.k_tln != material.k_auth
    assert material.master_key == repeated.master_key
    assert material.k_tln == repeated.k_tln
    assert material.k_auth == repeated.k_auth
    assert material.stream == repeated.stream
    assert material.master_key == nonce_changed.master_key
    assert material.k_tln == nonce_changed.k_tln
    assert material.stream != nonce_changed.stream
    assert material.master_key != salt_changed.master_key

    expected_block_0 = hashlib.sha256(material.k_tln + nonce + (0).to_bytes(4, "big")).digest()
    assert material.stream[:32] == expected_block_0[: len(material.stream)]

    expected_segments = np.array(
        [int.from_bytes(material.stream[4 * i : 4 * i + 4], "big") for i in range(4)],
        dtype=np.uint64,
    )
    assert np.array_equal(uint32_be_segments(material.stream, 4), expected_segments)
    assert np.allclose(material.normalized_values, expected_segments.astype(float) / float(2**32))
    assert np.all((0.0 <= material.normalized_values) & (material.normalized_values < 1.0))

    lower = material.basin_bounds[:, 0]
    upper = material.basin_bounds[:, 1]
    assert np.all(material.initial_conditions >= lower)
    assert np.all(material.initial_conditions < upper)

    _, trajectory = simulate_tln(material.initial_conditions, params=create_tln_parameters(), steps=64, dt=0.005)
    assert np.all(np.isfinite(trajectory))

    header = section_2_4_1_header(material)
    assert header == {"salt_hex": salt.hex(), "nonce_hex": nonce.hex()}

    return {
        "salt_hex": material.salt.hex(),
        "nonce_hex": material.nonce.hex(),
        "master_key_prefix": material.master_key[:4].hex(),
        "k_tln_prefix": material.k_tln[:4].hex(),
        "k_auth_prefix": material.k_auth[:4].hex(),
        "normalized_values": material.normalized_values,
        "initial_conditions": material.initial_conditions,
        "trajectory_final": trajectory[-1],
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


def verify_cloud_section_2_4_3() -> dict:
    """Verify Algorithms 4-6 for cloud-assisted storage and retrieval."""
    image = np.arange(64, dtype=np.uint8).reshape(8, 8)
    physician_keys = section_2_4_3_generate_kem_keypair()
    search_key = hashlib.sha256(b"section-2.4.3-search-key").digest()
    image_key = hashlib.sha256(b"section-2.4.3-image-key").digest()
    salt = bytes.fromhex("00112233445566778899aabbccddeeff")
    nonce = bytes.fromhex("102132435465768798a9bacb")
    aes_nonce = bytes.fromhex("0102030405060708090a0b0c")
    cloud_database: list[Section243CloudRecord] = []

    record = section_2_4_3_encrypt_and_upload(
        image,
        ["MRI", " Patient   42 ", "Brain Scan"],
        physician_keys.public_key,
        search_key,
        cloud_database=cloud_database,
        image_key=image_key,
        record_id="study-0001",
        aes_nonce=aes_nonce,
        salt=salt,
        nonce=nonce,
        arnold_iterations=3,
        arnold_r=11,
        arnold_z=19,
        block_power=2,
    )

    assert len(cloud_database) == 1
    assert record.record_id == "study-0001"
    assert record.kem_provider == physician_keys.provider
    assert len(record.token_set) == 3
    assert all(len(token) == 32 for token in record.token_set)
    assert record.aes_nonce == aes_nonce
    assert len(physician_keys.public_key) == ml_kem_768.PUBLIC_KEY_SIZE
    assert len(physician_keys.private_key) == ml_kem_768.SECRET_KEY_SIZE
    assert len(record.ckem) == ml_kem_768.CIPHERTEXT_SIZE
    assert len(record.image_tag) == 32
    assert not np.array_equal(record.encrypted_image, image)

    query_tokens = section_2_4_3_token_set(search_key, ["mri", "patient 42"])
    assert section_2_4_3_cloud_match(query_tokens, cloud_database, mode="AND") == [record]
    assert section_2_4_3_cloud_match(
        section_2_4_3_token_set(search_key, ["unknown", "brain scan"]),
        cloud_database,
        mode="OR",
    ) == [record]
    assert section_2_4_3_cloud_match(
        section_2_4_3_token_set(search_key, ["unknown"]),
        cloud_database,
        mode="AND",
    ) == []

    recovered, selected = section_2_4_3_physician_retrieve_and_decrypt(
        ["MRI", "patient 42"],
        physician_keys.private_key,
        search_key,
        cloud_database,
        mode="AND",
    )
    assert selected == record
    assert np.array_equal(recovered, image)

    tampered_image = record.encrypted_image.copy()
    tampered_image[0, 0] ^= np.uint8(1)
    tampered_record = Section243CloudRecord(
        record_id=record.record_id,
        encrypted_image=tampered_image,
        token_set=record.token_set,
        ckem=record.ckem,
        encrypted_key_bundle=record.encrypted_key_bundle,
        aes_nonce=record.aes_nonce,
        image_tag=record.image_tag,
        kem_provider=record.kem_provider,
    )
    try:
        section_2_4_3_physician_retrieve_and_decrypt(
            ["MRI"],
            physician_keys.private_key,
            search_key,
            [tampered_record],
            mode="AND",
        )
        raise AssertionError("tampered record was accepted")
    except ValueError as exc:
        assert "Integrity verification failed" in str(exc)

    no_match_image, no_match_record = section_2_4_3_physician_retrieve_and_decrypt(
        ["ultrasound"],
        physician_keys.private_key,
        search_key,
        cloud_database,
        mode="AND",
    )
    assert no_match_image is None
    assert no_match_record is None

    return {
        "records": len(cloud_database),
        "record_id": record.record_id,
        "token_count": len(record.token_set),
        "kem_provider": record.kem_provider,
        "cipher_checksum": int(np.sum(record.encrypted_image, dtype=np.uint64)),
        "round_trip_ok": True,
        "tamper_rejected": True,
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures-dir", default="figures_2_2", help="Directory for generated SVG figures.")
    parser.add_argument("--quality", choices=["quick", "full"], default="quick", help="Figure computation quality.")
    parser.add_argument("--skip-figures", action="store_true", help="Run verification only.")
    parser.add_argument("--skip-qiskit", action="store_true", help="Skip quantum-circuit checks.")
    parser.add_argument("--paper-framework-demo", action="store_true", help="Generate the 4x4 paper-framework Qiskit demo bundle.")
    parser.add_argument("--demo-dir", default="paper_framework_demo", help="Output directory for --paper-framework-demo artifacts.")
    parser.add_argument("--demo-arnold-n", type=int, default=6, help="Manuscript-scale Arnold/MADD circuit size.")
    parser.add_argument("--demo-arnold-iterations", type=int, default=2, help="Arnold iterations for the 4x4 demo.")
    parser.add_argument("--demo-use-exorcism4", action="store_true", help="Generate EXORCISM-4 ESOP comparison circuits.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    results = run_all_verifications(skip_qiskit=args.skip_qiskit)
    print_verification_summary(results)

    if not args.skip_figures:
        paths = generate_section_2_2_figures(args.figures_dir, quality=args.quality)
        print("Generated Section 2.2 figure reproductions:")
        for path in paths:
            print(f"  {path}")

    if args.paper_framework_demo:
        if args.skip_qiskit:
            raise RuntimeError("--paper-framework-demo requires Qiskit; remove --skip-qiskit.")
        demo = build_paper_framework_demo(
            outdir=args.demo_dir,
            arnold_n=args.demo_arnold_n,
            arnold_iterations=args.demo_arnold_iterations,
            use_exorcism4=args.demo_use_exorcism4,
        )
        print("Generated paper-framework demo bundle:")
        print(f"  directory={args.demo_dir}")
        print(f"  matrices={Path(args.demo_dir) / 'matrices.md'}")
        print(f"  resources={Path(args.demo_dir) / 'resource_table.md'}")
        print(f"  esop={Path(args.demo_dir) / 'esop_comparison_table.md'}")
        print(f"  circuits={len(demo['circuits'])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
