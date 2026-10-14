#!/usr/bin/env python3
"""
StructSurrogate-TopOpt: Physics-Informed Euler-Bernoulli Inverse Stiffness Solver,
Monte Carlo / FORM Bridge Reliability Engine, and SIMP 2D Topology Optimizer
====================================================================================
Author: Kshitija Karmore (B.Tech Civil Engineering, IIT Delhi)
Course Grounding: CVL2502 (Engineering Mechanics of Determinate Structures)
Method reference (not an exact paper reproduction or neural network):
  Sigmund, O. (2001), "A 99 line topology optimization code written in Matlab",
     Structural and Multidisciplinary Optimization, 21(2), 120-127.

Zero-Dependency Pure-Python Implementation (Custom Banded Cholesky & Gaussian Solver,
Hermite Cubic Beam FEM, Physics-Informed Spectral Collocation, Tikhonov Inverse Solver,
100,000-Trial Monte Carlo & Hasofer-Lind FORM, and Q4 Plane-Stress SIMP Optimizer).
"""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


# ==============================================================================
# FAST PURE-PYTHON LINEAR ALGEBRA & BANDED SPD CHOLESKY SOLVER
# ==============================================================================

def solve_dense_linear(A: List[List[float]], b: List[float]) -> List[float]:
    """Solves A * x = b via Gaussian elimination with partial pivoting."""
    n = len(b)
    if not n or len(A) != n or any(len(row) != n for row in A):
        raise ValueError("Expected a nonempty square system")
    if any(not math.isfinite(value) for row in A for value in row) or any(not math.isfinite(value) for value in b):
        raise ValueError("Linear system must be finite")
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for k in range(n):
        max_row = k
        max_val = abs(M[k][k])
        for i in range(k + 1, n):
            if abs(M[i][k]) > max_val:
                max_val = abs(M[i][k])
                max_row = i
        if max_row != k:
            M[k], M[max_row] = M[max_row], M[k]
        piv = M[k][k]
        if abs(piv) < 1e-18:
            raise ValueError("Singular or numerically unresolved linear system")
        for i in range(k + 1, n):
            fac = M[i][k] / piv
            if fac != 0.0:
                row_k = M[k]
                row_i = M[i]
                for j in range(k, n + 1):
                    row_i[j] -= fac * row_k[j]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = M[i][n]
        row_i = M[i]
        for j in range(i + 1, n):
            s -= row_i[j] * x[j]
        x[i] = s / row_i[i]
    return x


def solve_banded_spd(A_band: List[List[float]], b: List[float], bw: int) -> List[float]:
    """
    Fast Cholesky LDL^T solver for a symmetric positive-definite banded matrix
    where A_band[i][d] stores A[i, i + d] for 0 <= d <= bw.
    Complexity: O(N * bw^2) instead of O(N^3).
    """
    n = len(b)
    if not n or not isinstance(bw, int) or bw < 0 or len(A_band) != n or any(len(row) != bw+1 for row in A_band):
        raise ValueError("Invalid banded system dimensions")
    if any(not math.isfinite(value) for row in A_band for value in row) or any(not math.isfinite(value) for value in b):
        raise ValueError("Linear system must be finite")
    # In-place LDL^T factorization in lower-triangular band L[i][d] = L_{i, i-d}
    L = [[0.0] * (bw + 1) for _ in range(n)]
    D = [0.0] * n

    for i in range(n):
        d_min = max(0, i - bw)
        # Compute L[i][i - j] for j in d_min..i-1
        for j in range(d_min, i):
            s = A_band[j][i - j]
            k_min = max(d_min, j - bw)
            for k in range(k_min, j):
                s -= L[i][i - k] * D[k] * L[j][j - k]
            L[i][i - j] = s / D[j]
        s_diag = A_band[i][0]
        for k in range(d_min, i):
            s_diag -= (L[i][i - k] ** 2) * D[k]
        if s_diag <= 0 or not math.isfinite(s_diag):
            raise ValueError("Matrix must be positive definite; no pivot clipping applied")
        D[i] = s_diag

    # Forward substitution: L * y = b
    y = [0.0] * n
    for i in range(n):
        s = b[i]
        d_min = max(0, i - bw)
        for k in range(d_min, i):
            s -= L[i][i - k] * y[k]
        y[i] = s

    # Diagonal scaling + Backward substitution: L^T * x = D^{-1} y
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = y[i] / D[i]
        d_max = min(n - 1, i + bw)
        for k in range(i + 1, d_max + 1):
            s -= L[k][k - i] * x[k]
        x[i] = s
    return x


# ==============================================================================
# PART 1: EULER-BERNOULLI BRIDGE GIRDER FORWARD & PHYSICS-INFORMED INVERSE SOLVER
# ==============================================================================

class EulerBernoulliBridgeSolver:
    """
    Solves the 4th-order Euler-Bernoulli beam PDE:
        d^2/dx^2 ( EI(x) * d^2w/dx^2 ) = q(x)
    using exact 2-node Hermite Cubic Finite Elements (4 DOFs per element: w_i, theta_i, w_j, theta_j)
    and Physics-Informed Spectral Collocation Surrogate + Tikhonov-Regularized Inverse EI(x)
    damage identification from sparse noisy sensor measurements.
    """

    def __init__(
        self,
        span_m: float = 24.0,
        num_elements: int = 48,
        ei_intact_nm2: float = 8.40e9,   # E = 35 GPa, I = 0.24 m^4 -> EI = 8.40e9 N*m^2
        q_dead_n_per_m: float = 45000.0, # 45 kN/m uniform dead + wearing course load
        point_load_n: float = 240000.0,  # 240 kN concentrated axle load at midspan
    ) -> None:
        if not isinstance(num_elements, int) or isinstance(num_elements, bool) or num_elements < 2 or num_elements % 2:
            raise ValueError("Use an even number of elements >= 2 so the midspan load lies on a node")
        if not all(math.isfinite(value) and value > 0 for value in (span_m, ei_intact_nm2)):
            raise ValueError("Span and stiffness must be finite and positive")
        if not all(math.isfinite(value) and value >= 0 for value in (q_dead_n_per_m, point_load_n)):
            raise ValueError("Loads must be finite and nonnegative")
        self.L = span_m
        self.ne = num_elements
        self.nn = num_elements + 1
        self.x_nodes = [i * (self.L / self.ne) for i in range(self.nn)]
        self.dx = self.L / self.ne
        self.ei0 = ei_intact_nm2
        self.q0 = q_dead_n_per_m
        self.P = point_load_n

    def exact_intact_midspan_deflection_m(self) -> float:
        w_udl = (5.0 * self.q0 * (self.L ** 4)) / (384.0 * self.ei0)
        w_pt = (self.P * (self.L ** 3)) / (48.0 * self.ei0)
        return w_udl + w_pt

    def exact_intact_deflection_at(self, x: float) -> float:
        L = self.L
        ei = self.ei0
        w_udl = (self.q0 * x * (L**3 - 2.0 * L * (x**2) + (x**3))) / (24.0 * ei)
        if x <= L / 2.0:
            w_pt = (self.P * x * (3.0 * (L**2) - 4.0 * (x**2))) / (48.0 * ei)
        else:
            dx_r = L - x
            w_pt = (self.P * dx_r * (3.0 * (L**2) - 4.0 * (dx_r**2))) / (48.0 * ei)
        return w_udl + w_pt

    def solve_fem(self, ei_element: Sequence[float]) -> Tuple[List[float], List[float], List[float]]:
        if len(ei_element) != self.ne or any(not math.isfinite(value) or value <= 0 for value in ei_element):
            raise ValueError("Supply one finite positive stiffness per element")
        ndof = 2 * self.nn
        bw = 3
        K_band = [[0.0] * (bw + 1) for _ in range(ndof)]
        F = [0.0] * ndof
        le = self.dx
        inv_le3 = 1.0 / (le ** 3)

        k_base = [
            [12.0 * inv_le3,         6.0 * le * inv_le3,    -12.0 * inv_le3,         6.0 * le * inv_le3],
            [6.0 * le * inv_le3,     4.0 * le**2 * inv_le3, -6.0 * le * inv_le3,     2.0 * le**2 * inv_le3],
            [-12.0 * inv_le3,       -6.0 * le * inv_le3,     12.0 * inv_le3,        -6.0 * le * inv_le3],
            [6.0 * le * inv_le3,     2.0 * le**2 * inv_le3, -6.0 * le * inv_le3,     4.0 * le**2 * inv_le3],
        ]
        f_udl_elem = [
            self.q0 * le / 2.0,
            self.q0 * (le ** 2) / 12.0,
            self.q0 * le / 2.0,
            -self.q0 * (le ** 2) / 12.0,
        ]

        for e in range(self.ne):
            base = 2 * e
            ei_e = ei_element[e]
            for i_loc in range(4):
                ig = base + i_loc
                F[ig] += f_udl_elem[i_loc]
                for j_loc in range(i_loc, 4):
                    jg = base + j_loc
                    K_band[ig][jg - ig] += ei_e * k_base[i_loc][j_loc]

        mid_node = self.ne // 2
        F[2 * mid_node] += self.P

        original_stiffness = [row[:] for row in K_band]
        original_force = F[:]
        supports = (0, 2*self.ne)
        for support in supports:
            for row in range(max(0, support-bw), support):
                K_band[row][support-row] = 0.0
            K_band[support] = [1.0] + [0.0]*bw
            F[support] = 0.0

        u = solve_banded_spd(K_band, F, bw)
        residual = [-value for value in original_force]
        for row in range(ndof):
            for offset in range(min(bw+1, ndof-row)):
                column = row+offset
                value = original_stiffness[row][offset]
                residual[row] += value*u[column]
                if offset:
                    residual[column] += value*u[row]
        self.last_equilibrium = {
            "support_reactions_n": [-residual[index] for index in supports],
            "total_applied_load_n": self.q0*self.L+self.P,
            "force_balance_residual_n": sum(residual[index] for index in supports)+self.q0*self.L+self.P,
            "free_dof_max_residual": max(abs(value) for index, value in enumerate(residual) if index not in supports),
        }
        w_nodes = [u[2 * i] for i in range(self.nn)]
        theta_nodes = [u[2 * i + 1] for i in range(self.nn)]

        # Statically determinate exact internal bending moment M(x) at element midpoints
        r_a = 0.5 * self.q0 * self.L + 0.5 * self.P
        moment_elem = []
        for e in range(self.ne):
            x_mid = (e + 0.5) * le
            m_val = r_a * x_mid - 0.5 * self.q0 * (x_mid ** 2)
            if x_mid > self.L / 2.0:
                m_val -= self.P * (x_mid - self.L / 2.0)
            moment_elem.append(m_val)

        return w_nodes, theta_nodes, moment_elem

    def train_physics_informed_spectral_surrogate(
        self, n_modes: int = 9, n_collocation: int = 96
    ) -> Dict[str, float]:
        setup_started = time.perf_counter()
        if not isinstance(n_modes, int) or n_modes < 1 or not isinstance(n_collocation, int) or n_collocation < max(16, n_modes+1):
            raise ValueError("Invalid mode count or collocation grid")
        if self.q0 == 0 and self.P == 0:
            raise ValueError("Relative surrogate error is undefined for zero loading")
        x_col = [i * (self.L / (n_collocation - 1)) for i in range(n_collocation)]
        w_exact_col = [self.exact_intact_deflection_at(x) for x in x_col]

        # 15 sparse virtual LVDT sensors
        sensor_idx = [int(round(2 + i * (n_collocation - 5) / 14.0)) for i in range(15)]
        x_sens = [x_col[idx] for idx in sensor_idx]
        w_sens = [w_exact_col[idx] for idx in sensor_idx]

        modes = [m for m in range(1, n_modes + 1)]
        k_m = [m * math.pi / self.L for m in modes]

        # Fourier modal load projection for Euler-Bernoulli PDE residual
        q_fourier = []
        for m in modes:
            if m % 2 == 1:
                term_udl = (4.0 * self.q0) / (m * math.pi)
                term_pt = (2.0 * self.P / self.L) * math.sin(m * math.pi / 2.0)
                q_fourier.append(term_udl + term_pt)
            else:
                q_fourier.append(0.0)

        q_eff_col = [
            sum(q_fourier[j] * math.sin(k_m[j] * x) for j in range(n_modes))
            for x in x_col
        ]

        # Normal equations (A^T A) c = A^T b combining Physics PDE residual + Sensor Data loss
        alpha_pde = 1.0 / (max(self.q0, self.P/self.L) ** 2)
        alpha_dat = 1.0e6

        ATA = [[0.0] * n_modes for _ in range(n_modes)]
        ATb = [0.0] * n_modes

        for r, x in enumerate(x_col):
            row_pde = [self.ei0 * (k_m[j] ** 4) * math.sin(k_m[j] * x) for j in range(n_modes)]
            rhs_pde = q_eff_col[r]
            for i in range(n_modes):
                ATb[i] += alpha_pde * row_pde[i] * rhs_pde
                for j in range(n_modes):
                    ATA[i][j] += alpha_pde * row_pde[i] * row_pde[j]

        for r, x in enumerate(x_sens):
            row_dat = [math.sin(k_m[j] * x) for j in range(n_modes)]
            rhs_dat = w_sens[r]
            for i in range(n_modes):
                ATb[i] += alpha_dat * row_dat[i] * rhs_dat
                for j in range(n_modes):
                    ATA[i][j] += alpha_dat * row_dat[i] * row_dat[j]

        c_opt = solve_dense_linear(ATA, ATb)

        w_pred_col = [
            sum(c_opt[j] * math.sin(k_m[j] * x) for j in range(n_modes))
            for x in x_col
        ]
        err_sq = sum((wp - we) ** 2 for wp, we in zip(w_pred_col, w_exact_col))
        norm_sq = sum(we ** 2 for we in w_exact_col)
        l2_rel_err_pct = math.sqrt(err_sq / norm_sq) * 100.0

        pde_res_sq = 0.0
        q_norm_sq = sum(q ** 2 for q in q_eff_col)
        for r, x in enumerate(x_col):
            lhs_val = sum(self.ei0 * (k_m[j] ** 4) * c_opt[j] * math.sin(k_m[j] * x) for j in range(n_modes))
            pde_res_sq += (lhs_val - q_eff_col[r]) ** 2
        pde_residual_rel = math.sqrt(pde_res_sq / q_norm_sq)

        setup_ms = (time.perf_counter()-setup_started)*1000
        Phi_sens = [[math.sin(k_m[j] * x) for j in range(n_modes)] for x in x_sens]
        ei_uniform = [self.ei0] * self.ne
        inference = lambda: [sum(row[j] * c_opt[j] for j in range(n_modes)) for row in Phi_sens]
        forward = lambda: self.solve_fem(ei_uniform)
        timing = {}
        for label, operation in (("inference", inference), ("fem", forward)):
            operation()
            batches = []
            for _ in range(7):
                started = time.perf_counter()
                for _ in range(20):
                    operation()
                batches.append((time.perf_counter()-started)*1000/20)
            timing[label] = batches
        t_surrogate_ms = statistics.median(timing["inference"])
        t_fem_ms = statistics.median(timing["fem"])

        return {
            "num_spectral_modes": n_modes,
            "num_collocation_points": n_collocation,
            "num_virtual_sensors": len(sensor_idx),
            "setup_ms": setup_ms,
            "timing_batch_means_ms": timing,
            "timing_caveat": "Cached 15-sensor inference versus complete FEM assembly, solve and equilibrium diagnostics; not like-for-like end-to-end speedup",
            "l2_relative_error_pct": round(l2_rel_err_pct, 5),
            "relative_pde_residual": round(pde_residual_rel, 8),
            "surrogate_inference_ms": round(t_surrogate_ms, 5),
            "fem_solve_ms": round(t_fem_ms, 5),
            "speedup_factor_x": round(t_fem_ms / max(t_surrogate_ms, 1e-9), 1),
        }

    def solve_inverse_damage_identification(self, seed=42, noise_fraction=0.01, sensor_indices=None, damage_center=11.0, damage_severity=0.32, moment_bias=0.0, regularization=1.2e-6) -> Dict[str, object]:
        if not all(math.isfinite(value) for value in (noise_fraction, damage_center, damage_severity, moment_bias, regularization)):
            raise ValueError("Inverse parameters must be finite")
        if noise_fraction < 0 or not 0 <= damage_severity < 1 or not 0 <= damage_center <= self.L or regularization <= 0 or moment_bias <= -1:
            raise ValueError("Invalid inverse parameters")
        sensors = list(range(self.ne)) if sensor_indices is None else list(sensor_indices)
        if len(sensors) < 2 or len(set(sensors)) != len(sensors) or any(not isinstance(index, int) or not 0 <= index < self.ne for index in sensors):
            raise ValueError("At least two distinct valid sensor indices are required")
        rng = random.Random(seed)
        x_mid = [(e + 0.5) * self.dx for e in range(self.ne)]

        # True damaged stiffness profile: 32% smooth Gaussian stiffness drop centered at x = 11.0m
        damage_fraction_true = [
            damage_severity * math.exp(-0.5 * ((xm - damage_center) / 1.6) ** 2) for xm in x_mid
        ]
        ei_true = [self.ei0 * (1.0 - df) for df in damage_fraction_true]

        w_damaged, _, moment_elem = self.solve_fem(ei_true)

        # Curvature kappa_e = M_e / EI_true_e with 1.0% Gaussian sensor noise
        kappa_true = [(1+moment_bias)*moment / stiffness for moment, stiffness in zip(moment_elem, ei_true)]
        mean_k = sum(kappa_true) / self.ne
        std_k = math.sqrt(sum((k - mean_k) ** 2 for k in kappa_true) / self.ne)
        kappa_obs = [value + rng.gauss(0.0, noise_fraction * std_k) for value in kappa_true]

        # Tikhonov-regularized inverse system for compliance ratio s_e = EI0 / EI_e:
        #   (M_hat^2 + reg_lambda * L^T L) s = M_hat * kappa_obs
        m_hat = [m / self.ei0 for m in moment_elem]
        reg_lambda = regularization
        A_inv = [[0.0] * self.ne for _ in range(self.ne)]
        b_inv = [0.0]*self.ne

        for i in sensors:
            b_inv[i] = m_hat[i]*kappa_obs[i]
            A_inv[i][i] += m_hat[i] ** 2

        # Add 2nd-order Laplacian smoothing L^T L
        for r in range(self.ne - 2):
            idxs = (r, r + 1, r + 2)
            coeffs = (1.0, -2.0, 1.0)
            for i_loc in range(3):
                for j_loc in range(3):
                    A_inv[idxs[i_loc]][idxs[j_loc]] += reg_lambda * coeffs[i_loc] * coeffs[j_loc]

        s_recon = solve_dense_linear(A_inv, b_inv)
        ei_recon = [self.ei0 / min(2.5, max(0.5, s)) for s in s_recon]

        err_sq = sum((er - et) ** 2 for er, et in zip(ei_recon, ei_true))
        norm_sq = sum(et ** 2 for et in ei_true)
        l2_ei_err_pct = math.sqrt(err_sq / norm_sq) * 100.0

        peak_damage_true_pct = max(damage_fraction_true) * 100.0
        peak_damage_recon_pct = max(1.0 - er / self.ei0 for er in ei_recon) * 100.0
        idx_true = damage_fraction_true.index(max(damage_fraction_true))
        idx_recon = ei_recon.index(min(ei_recon))

        return {
            "seed": seed, "noise_fraction_of_curvature_std": noise_fraction,
            "sensor_indices": sensors, "moment_bias_fraction": moment_bias,
            "regularization": regularization,
            "baseline_intact_profile_l2_error_pct": 100*math.sqrt(sum((self.ei0-value)**2 for value in ei_true)/norm_sq),
            "clipped_compliance_values": sum(not 0.5 <= value <= 2.5 for value in s_recon),
            "damage_location_error_m": abs(x_mid[idx_true]-x_mid[idx_recon]),
            "observation_fit_rmse_curvature": math.sqrt(sum((moment_elem[index]/ei_recon[index]-kappa_obs[index])**2 for index in sensors)/len(sensors)),
            "true_peak_stiffness_loss_pct": round(peak_damage_true_pct, 2),
            "reconstructed_peak_stiffness_loss_pct": round(peak_damage_recon_pct, 2),
            "true_damage_center_m": round(x_mid[idx_true], 2),
            "reconstructed_damage_center_m": round(x_mid[idx_recon], 2),
            "stiffness_profile_l2_error_pct": round(l2_ei_err_pct, 3),
            "midspan_deflection_intact_mm": round(self.exact_intact_midspan_deflection_m() * 1000.0, 3),
            "midspan_deflection_damaged_mm": round(max(w_damaged) * 1000.0, 3),
            "x_mid_m": [round(v, 3) for v in x_mid],
            "ei_true_gnm2": [round(v / 1e9, 4) for v in ei_true],
            "ei_recon_gnm2": [round(v / 1e9, 4) for v in ei_recon],
        }


# ==============================================================================
# PART 2: 100,000-TRIAL MONTE CARLO & FORM BRIDGE RELIABILITY ENGINE
# ==============================================================================

def run_monte_carlo_and_form_reliability(
    span_m: float = 24.0,
    num_trials: int = 100_000,
) -> Dict[str, float]:
    if not math.isfinite(span_m) or span_m <= 0 or not isinstance(num_trials, int) or num_trials < 1:
        raise ValueError("Positive finite span and positive integer trial count required")
    rng = random.Random(2026)
    w_allow_m = span_m / 600.0  # 0.040 m = 40.0 mm

    mu_ei, sig_ei = 8.40e9, 0.10 * 8.40e9
    mu_q, sig_q = 45000.0, 0.08 * 45000.0
    mu_p, sig_p = 240000.0, 0.18 * 240000.0

    c_udl = 5.0 * (span_m ** 4) / 384.0
    c_pt = (span_m ** 3) / 48.0

    failures = 0
    clipped_stiffness_samples = 0
    sum_w = 0.0
    sum_g = 0.0
    sum_g2 = 0.0
    w_list: List[float] = []

    gauss = rng.gauss
    for _ in range(num_trials):
        ei_s = gauss(mu_ei, sig_ei)
        if ei_s < 2.0e9:
            clipped_stiffness_samples += 1
            ei_s = 2.0e9
        q_s = gauss(mu_q, sig_q)
        p_s = gauss(mu_p, sig_p)
        w_s = (c_udl * q_s + c_pt * p_s) / ei_s
        w_list.append(w_s)
        sum_w += w_s
        g_s = w_allow_m - w_s
        sum_g += g_s
        sum_g2 += g_s * g_s
        if g_s < 0.0:
            failures += 1

    pf_mc = failures / float(num_trials)
    mean_w = sum_w / num_trials
    mean_g = sum_g / num_trials
    std_g = math.sqrt(max(1e-18, sum_g2 / num_trials - mean_g * mean_g))
    beta_mvfos = mean_g / std_g

    w_list.sort()
    p99_w = w_list[int(0.99 * num_trials)]

    # Hasofer-Lind FORM iterative solver in standard normal space U = [u_EI, u_q, u_P]
    u = [0.0, 0.0, 0.0]
    sigma = [sig_ei, sig_q, sig_p]
    mu = [mu_ei, mu_q, mu_p]

    for _ in range(25):
        ei_v = mu[0] + sigma[0] * u[0]
        q_v = mu[1] + sigma[1] * u[1]
        p_v = mu[2] + sigma[2] * u[2]
        w_v = (c_udl * q_v + c_pt * p_v) / ei_v
        g_val = w_allow_m - w_v

        grad_u = [
            (w_v / ei_v) * sigma[0],
            (-c_udl / ei_v) * sigma[1],
            (-c_pt / ei_v) * sigma[2],
        ]
        norm_grad = math.sqrt(sum(g * g for g in grad_u))
        dot_gu = sum(grad_u[i] * u[i] for i in range(3))
        beta_hl = (g_val - dot_gu) / norm_grad
        u = [(-grad_u[i] / norm_grad) * beta_hl for i in range(3)]

    beta_form = math.copysign(math.sqrt(sum(ui * ui for ui in u)), w_allow_m-(c_udl*mu_q+c_pt*mu_p)/mu_ei)
    pf_form = 0.5 * math.erfc(beta_form / math.sqrt(2.0))
    exact_beta = (w_allow_m*mu_ei-c_udl*mu_q-c_pt*mu_p) / math.sqrt((w_allow_m*sig_ei)**2+(c_udl*sig_q)**2+(c_pt*sig_p)**2)
    exact_pf = 0.5*math.erfc(exact_beta/math.sqrt(2))
    denominator = 1+1.96**2/num_trials
    centre = (pf_mc+1.96**2/(2*num_trials))/denominator
    half_width = 1.96*math.sqrt(pf_mc*(1-pf_mc)/num_trials+1.96**2/(4*num_trials**2))/denominator

    return {
        "num_monte_carlo_trials": num_trials,
        "failure_count": failures,
        "wilson_95_interval": [max(0.0, centre-half_width), min(1.0, centre+half_width)],
        "zero_failure_95_upper_bound": 1-0.05**(1/num_trials) if failures == 0 else None,
        "clipped_stiffness_samples": clipped_stiffness_samples,
        "linear_gaussian_limit_state_exact_probability": exact_pf,
        "reference_caveat": "Exact for linearized event w_allow*EI-c_udl*q-c_pt*P < 0 with independent Gaussians; MC clips EI below 2e9",
        "allowable_deflection_mm": round(w_allow_m * 1000.0, 2),
        "mean_deflection_mm": round(mean_w * 1000.0, 3),
        "p99_deflection_mm": round(p99_w * 1000.0, 3),
        "monte_carlo_failure_probability": round(pf_mc, 6),
        "form_failure_probability": round(pf_form, 6),
        "form_reliability_index_beta": round(beta_form, 4),
        "fos_mean_std_beta": round(beta_mvfos, 4),
    }


# ==============================================================================
# PART 3: SIGMUND (2001) SIMP 2D BRIDGE TRUSS TOPOLOGY OPTIMIZER (CVL2502)
# ==============================================================================

def lk_plane_stress_q4(nu: float = 0.30) -> List[List[float]]:
    """Exact 8x8 element stiffness matrix for a unit Q4 plane-stress element (Sigmund 2001)."""
    k = [
        1.0 / 2.0 - nu / 6.0,
        1.0 / 8.0 + nu / 8.0,
        -1.0 / 4.0 - nu / 12.0,
        -1.0 / 8.0 + 3.0 * nu / 8.0,
        -1.0 / 4.0 + nu / 12.0,
        -1.0 / 8.0 - nu / 8.0,
        nu / 6.0,
        1.0 / 8.0 - 3.0 * nu / 8.0,
    ]
    scale = 1.0 / (1.0 - nu * nu)
    idx_map = [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1, 0, 7, 6, 5, 4, 3, 2],
        [2, 7, 0, 5, 6, 3, 4, 1],
        [3, 6, 5, 0, 7, 2, 1, 4],
        [4, 5, 6, 7, 0, 1, 2, 3],
        [5, 4, 3, 2, 1, 0, 7, 6],
        [6, 3, 4, 1, 2, 7, 0, 5],
        [7, 2, 1, 4, 3, 6, 5, 0],
    ]
    return [[scale * k[idx] for idx in row] for row in idx_map]


def run_simp_bridge_topology_optimization(
    nelx: int = 30,
    nely: int = 10,
    volfrac: float = 0.45,
    penal: float = 3.0,
    rmin: float = 1.4,
    max_iter: int = 35,
) -> Dict[str, object]:
    """
    Implements a Sigmund-inspired SIMP minimum-compliance experiment
    for a symmetric half-MBB / Pratt bridge girder domain using banded Cholesky FEM.
    """
    if any(not isinstance(value, int) or value < 1 for value in (nelx, nely, max_iter)):
        raise ValueError("Positive mesh sizes and iteration count required")
    if not all(math.isfinite(value) for value in (volfrac, penal, rmin)) or not 0.001 < volfrac <= 1 or penal < 1 or rmin <= 0:
        raise ValueError("Invalid SIMP parameters")
    KE = lk_plane_stress_q4(nu=0.30)
    ndof = 2 * (nelx + 1) * (nely + 1)
    bw = 2 * (nely + 2) + 1  # exact half-bandwidth for column-major node numbering
    x = [[volfrac for _ in range(nelx)] for _ in range(nely)]

    edof_mat: List[List[int]] = []
    for elx in range(nelx):
        for ely in range(nely):
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            edof_mat.append([
                2 * n1, 2 * n1 + 1,
                2 * n2, 2 * n2 + 1,
                2 * n2 + 2, 2 * n2 + 3,
                2 * n1 + 2, 2 * n1 + 3,
            ])

    # Precompute mesh-independence filter neighbors
    r_ceil = int(math.ceil(rmin))
    H_rows: List[List[Tuple[int, int, float]]] = [[] for _ in range(nelx * nely)]
    for i in range(nelx):
        for j in range(nely):
            e1 = i * nely + j
            for k in range(max(0, i - r_ceil), min(nelx, i + r_ceil + 1)):
                for l in range(max(0, j - r_ceil), min(nely, j + r_ceil + 1)):
                    fac = rmin - math.sqrt((i - k) ** 2 + (j - l) ** 2)
                    if fac > 0.0:
                        H_rows[e1].append((k, l, fac))

    # Fixed DOFs: left symmetry edge x-DOFs + bottom-right bridge pier y-DOF
    fixed_dofs = list(range(0, 2 * (nely + 1), 2)) + [ndof - 1]
    Emin = 1e-3
    E0 = 1.0

    compliance_history: List[float] = []
    converged = False
    max_diff = math.inf

    for loop in range(1, max_iter + 2):
        K_band = [[0.0] * (bw + 1) for _ in range(ndof)]
        F = [0.0] * ndof
        F[1] = -1.0

        for elx in range(nelx):
            for ely in range(nely):
                el = elx * nely + ely
                dofs = edof_mat[el]
                stiffness_factor = Emin + (x[ely][elx] ** penal) * (E0 - Emin)
                for i_loc in range(8):
                    ig = dofs[i_loc]
                    ke_row = KE[i_loc]
                    for j_loc in range(8):
                        jg = dofs[j_loc]
                        if jg >= ig:
                            K_band[ig][jg - ig] += stiffness_factor * ke_row[j_loc]

        for fd in fixed_dofs:
            for row in range(max(0, fd-bw), fd):
                K_band[row][fd-row] = 0.0
            K_band[fd] = [1.0]+[0.0]*bw
            F[fd] = 0.0

        U = solve_banded_spd(K_band, F, bw)

        c = 0.0
        dc = [[0.0] * nelx for _ in range(nely)]
        for elx in range(nelx):
            for ely in range(nely):
                el = elx * nely + ely
                dofs = edof_mat[el]
                Ue = [U[d] for d in dofs]
                ce = 0.0
                for i_loc in range(8):
                    ui = Ue[i_loc]
                    ke_row = KE[i_loc]
                    s_row = 0.0
                    for j_loc in range(8):
                        s_row += ke_row[j_loc] * Ue[j_loc]
                    ce += ui * s_row
                rho = x[ely][elx]
                c += (Emin + (rho ** penal) * (E0 - Emin)) * ce
                dc[ely][elx] = -penal * (rho ** (penal - 1.0)) * (E0 - Emin) * ce

        compliance_history.append(round(c, 4))
        if converged or loop > max_iter:
            break

        # Mesh-independence filter
        dcn = [[0.0] * nelx for _ in range(nely)]
        for i in range(nelx):
            for j in range(nely):
                e1 = i * nely + j
                sum_w = 0.0
                val = 0.0
                for k, l, fac in H_rows[e1]:
                    sum_w += fac
                    val += fac * x[l][k] * dc[l][k]
                dcn[j][i] = val / (x[j][i] * sum_w)

        # Optimality Criteria (OC) bisection update
        l1, l2, move = 0.0, 100000.0, 0.2
        xnew = [[0.0] * nelx for _ in range(nely)]
        while (l2 - l1) / (l2 + l1 + 1e-12) > 1e-4:
            lmid = 0.5 * (l2 + l1)
            vol_sum = 0.0
            for ely in range(nely):
                for elx in range(nelx):
                    rho = x[ely][elx]
                    cand = rho * math.sqrt(max(0.0, -dcn[ely][elx] / lmid))
                    val = max(0.001, max(rho - move, min(1.0, min(rho + move, cand))))
                    xnew[ely][elx] = val
                    vol_sum += val
            if vol_sum - volfrac * nelx * nely > 0.0:
                l1 = lmid
            else:
                l2 = lmid

        max_diff = max(
            abs(xnew[ely][elx] - x[ely][elx])
            for ely in range(nely)
            for elx in range(nelx)
        )
        x = xnew
        if max_diff < 0.01 and loop >= 20:
            converged = True

    initial_c = compliance_history[0]
    final_c = compliance_history[-1]
    compliance_reduction_pct = (initial_c - final_c) / initial_c * 100.0
    achieved_vol = sum(sum(row) for row in x) / (nelx * nely)
    binary_count = sum(
        1 for ely in range(nely) for elx in range(nelx)
        if x[ely][elx] > 0.8 or x[ely][elx] < 0.2
    )

    ascii_rows: List[str] = []
    for ely in range(nely):
        row_chars = []
        for elx in range(nelx):
            val = x[ely][elx]
            if val > 0.75:
                row_chars.append("#")
            elif val > 0.40:
                row_chars.append("+")
            elif val > 0.15:
                row_chars.append(".")
            else:
                row_chars.append(" ")
        ascii_rows.append("".join(row_chars))

    return {
        "mesh_dimensions": f"{nelx}x{nely}",
        "num_elements": nelx * nely,
        "target_volume_fraction": volfrac,
        "achieved_volume_fraction": round(achieved_vol, 4),
        "simp_penalization_p": penal,
        "filter_radius_rmin": rmin,
        "iterations_executed": len(compliance_history)-1,
        "converged": converged,
        "last_max_density_change": max_diff,
        "termination_reason": "density_change_tolerance" if converged else "iteration_limit",
        "final_compliance": round(final_c, 4),
        "initial_compliance": round(initial_c, 4),
        "final_optimal_compliance": round(final_c, 4),
        "compliance_reduction_pct": round(compliance_reduction_pct, 2),
        "binary_solid_void_fraction_pct": round(binary_count / (nelx * nely) * 100.0, 2),
        "compliance_history": compliance_history,
        "ascii_truss_topology": ascii_rows,
        "density_grid": x,
    }


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    reports_dir = base_dir / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    solver = EulerBernoulliBridgeSolver()

    # 1. Forward FEM vs Exact Analytical Macaulay Verification
    w_exact_mid = solver.exact_intact_midspan_deflection_m()
    ei_intact = [solver.ei0] * solver.ne
    w_fem, _, _ = solver.solve_fem(ei_intact)
    w_fem_mid = float(w_fem[solver.ne // 2])
    fem_err_pct = abs(w_fem_mid - w_exact_mid) / w_exact_mid * 100.0

    # 2. Physics-Informed Spectral Surrogate Benchmark
    surrogate_res = solver.train_physics_informed_spectral_surrogate()

    # 3. Inverse Stiffness Degradation Identification
    inverse_res = solver.solve_inverse_damage_identification()

    # 4. 100,000-Trial Monte Carlo + FORM Reliability Analysis
    reliability_res = run_monte_carlo_and_form_reliability()

    # 5. Sigmund (2001) SIMP 2D Bridge Truss Topology Optimization
    topopt_res = run_simp_bridge_topology_optimization()

    # Export inverse stiffness profile CSV
    inv_csv_path = data_dir / "inverse_stiffness_profile.csv"
    with inv_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x_mid_m", "ei_true_GNm2", "ei_reconstructed_GNm2"])
        for xm, eit, eir in zip(
            inverse_res["x_mid_m"],
            inverse_res["ei_true_gnm2"],
            inverse_res["ei_recon_gnm2"],
        ):
            writer.writerow([xm, eit, eir])

    # Export SIMP topology density matrix CSV
    top_csv_path = data_dir / "simp_bridge_truss_density.csv"
    with top_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in topopt_res["density_grid"]:
            writer.writerow([f"{val:.4f}" for val in row])

    tests = [
        {
            "test_id": "STRUCT_01_HERMITE_FEM_EXACT_MATCH",
            "description": "Hermite cubic FEM midspan deflection matches exact Euler-Bernoulli Macaulay solution within 0.01%",
            "computed_value": f"Exact={w_exact_mid*1000:.4f} mm, FEM={w_fem_mid*1000:.4f} mm (err={fem_err_pct:.6f}%)",
            "passed": fem_err_pct < 0.01,
        },
        {
            "test_id": "STRUCT_02_PHYSICS_INFORMED_SURROGATE_ACCURACY",
            "description": "Spectral surrogate has < 0.25% L2 deflection error; timing is descriptive, not a correctness criterion",
            "computed_value": f"L2_err={surrogate_res['l2_relative_error_pct']:.4f}%, Speedup={surrogate_res['speedup_factor_x']}x",
            "passed": surrogate_res["l2_relative_error_pct"] < 0.25,
        },
        {
            "test_id": "STRUCT_03_INVERSE_DAMAGE_LOCALIZATION",
            "description": "Regularized physics-constrained inverse solver reconstructs EI(x) within 2.5% L2 error under 1% noise",
            "computed_value": f"L2_err={inverse_res['stiffness_profile_l2_error_pct']:.3f}%, Center={inverse_res['reconstructed_damage_center_m']}m",
            "passed": inverse_res["stiffness_profile_l2_error_pct"] < 2.5,
        },
        {
            "test_id": "STRUCT_04_MONTE_CARLO_FORM_CONSISTENCY",
            "description": "100,000-trial Monte Carlo failure probability agrees closely with Hasofer-Lind FORM reliability index",
            "computed_value": f"P_f(MC)={reliability_res['monte_carlo_failure_probability']:.5f}, P_f(FORM)={reliability_res['form_failure_probability']:.5f}, beta={reliability_res['form_reliability_index_beta']:.3f}",
            "passed": abs(reliability_res["monte_carlo_failure_probability"] - reliability_res["form_failure_probability"]) < 0.005,
        },
        {
            "test_id": "STRUCT_05_SIMP_FEASIBILITY_AND_IMPROVEMENT",
            "description": "SIMP experiment approximately satisfies volume constraint and reduces compliance; convergence reported separately",
            "computed_value": f"Vol={topopt_res['achieved_volume_fraction']:.4f}, Compliance Reduction={topopt_res['compliance_reduction_pct']:.2f}%",
            "passed": abs(topopt_res["achieved_volume_fraction"] - 0.45) < 0.005 and topopt_res["final_compliance"] < topopt_res["initial_compliance"],
        },
    ]

    topopt_summary = {k: v for k, v in topopt_res.items() if k != "density_grid"}
    inverse_summary = {k: v for k, v in inverse_res.items() if k not in ("x_mid_m", "ei_true_gnm2", "ei_recon_gnm2")}

    report = {
        "project": "StructSurrogate-TopOpt: Physics-Informed Beam Inverse Solver, Reliability Engine & SIMP Topology Optimizer",
        "course_grounding": "CVL2502 (Engineering Mechanics of Determinate Structures, IIT Delhi)",
        "method_references_not_exact_reproductions": [
            "Sigmund, O. (2001), Structural and Multidisciplinary Optimization, 21(2), 120-127",
        ],
        "forward_fem_verification": {
            "span_m": solver.L,
            "num_elements": solver.ne,
            "exact_midspan_deflection_mm": round(w_exact_mid * 1000.0, 4),
            "fem_midspan_deflection_mm": round(w_fem_mid * 1000.0, 4),
            "relative_error_pct": round(fem_err_pct, 6),
        },
        "physics_informed_surrogate": surrogate_res,
        "inverse_damage_identification": inverse_summary,
        "monte_carlo_form_reliability": reliability_res,
        "simp_topology_optimization": topopt_summary,
        "verification_tests": tests,
        "all_tests_passed": all(t["passed"] for t in tests),
    }

    report_path = reports_dir / "structural_benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 84)
    print("StructSurrogate-TopOpt (CVL2502) — Benchmark & Verification Suite")
    print("=" * 84)
    for t in tests:
        status = "PASS" if t["passed"] else "FAIL"
        print(f"[{status}] {t['test_id']}: {t['computed_value']}")
    print("-" * 84)
    print("Optimized Half-MBB Bridge Truss Topology (Sigmund 2001 SIMP, 30x10 mesh):")
    for row in topopt_res["ascii_truss_topology"]:
        print("  |" + row + "|")
    print("=" * 84)


if __name__ == "__main__":
    main()
