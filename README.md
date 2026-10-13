# StructSurrogate-TopOpt: Physics-Informed Beam Inverse Solver, Reliability Engine & SIMP Topology Optimizer

**Author:** Kshitija Karmore (B.Tech Civil Engineering, IIT Delhi)  
**Course Grounding:** `CVL2502` (Engineering Mechanics of Determinate Structures)  
**Paper Reproductions:**
1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *"Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations."* Journal of Computational Physics, 378, 686–707.
2. Sigmund, O. (2001). *"A 99 line topology optimization code written in Matlab."* Structural and Multidisciplinary Optimization, 21(2), 120–127.

---

## 1. Overview & Mathematical Formulation

`StructSurrogate-TopOpt` is a zero-dependency computational mechanics and structural optimization suite implementing three coupled solvers for a $24.0\text{ m}$ highway bridge girder (`CVL2502`):

### A. Forward Hermite Cubic FEM & Physics-Informed Spectral Surrogate
Solves the 4th-order Euler–Bernoulli flexural differential equation under distributed dead load $q_0 = 45\text{ kN/m}$ and concentrated midspan axle load $P = 240\text{ kN}$ ($EI_0 = 8.40 \times 10^9\text{ N}\cdot\text{m}^2$):

$$\frac{d^2}{dx^2}\left(EI(x)\frac{d^2 w(x)}{dx^2}\right) = q(x)$$

and trains a physics-informed sine-basis spectral collocation surrogate $\hat{w}(x; \mathbf{c}) = \sum_{m=1}^{9} c_m \sin(m\pi x / L)$ over $N_c = 96$ PDE collocation points and $N_s = 15$ virtual LVDT sensors.

### B. Physics-Constrained Inverse Damage Localization ($EI(x)$ Reconstruction)
Given noisy curvature/deflection sensor observations ($\sigma_{\text{noise}} = 1.0\%$) from a girder with a localized $32\%$ flexural stiffness loss centered near $x = 11.0\text{ m}$, solves the Tikhonov-regularized inverse problem:

$$\min_{\mathbf{s}} \; \left\|\operatorname{diag}(\hat{\mathbf{M}})\mathbf{s} - \boldsymbol{\kappa}_{\text{obs}}\right\|_2^2 + \lambda \left\|\mathbf{D}_2 \mathbf{s}\right\|_2^2, \qquad s_e = \frac{EI_0}{EI_e}$$

### C. $10^5$-Trial Monte Carlo & Hasofer–Lind FORM Reliability Engine
Evaluates the Serviceability Limit State $g(EI, q, P) = w_{\text{allow}} - w_{\max}(EI, q, P)$ ($w_{\text{allow}} = L/600 = 40.0\text{ mm}$) across $100,000$ stochastic realizations and compares against iterative First-Order Reliability Method (FORM) in standard normal space $\mathbf{U}$.

### D. Sigmund (2001) 2D SIMP Bridge Truss Topology Optimization
Solves the minimum-compliance topology optimization problem on a 2D Q4 plane-stress mesh ($30 \times 10$ elements, volume fraction $f_v = 0.45$, penalization $p = 3.0$, mesh-independence filter $r_{\min} = 1.4$) using a custom banded $LDL^T$ Cholesky finite-element solver and Optimality Criteria (OC) updates:

$$\min_{\boldsymbol{\rho}} \; c(\boldsymbol{\rho}) = \mathbf{U}^T \mathbf{K}(\boldsymbol{\rho}) \mathbf{U} = \sum_{e=1}^{N_e} \left(E_{\min} + \rho_e^p (E_0 - E_{\min})\right) \mathbf{u}_e^T \mathbf{k}_0 \mathbf{u}_e \quad \text{s.t.} \quad \mathbf{K}\mathbf{U} = \mathbf{F}, \;\; \frac{V(\boldsymbol{\rho})}{V_0} = 0.45$$

---

## 2. Verified Benchmark Results (`run_structural_sim_and_tests.py`)

| Verification Test | Target / Analytical Reference | Computed Benchmark Result | Status |
| :--- | :--- | :--- | :--- |
| **`STRUCT_01` Forward Hermite FEM vs. Macaulay** | $w_{\text{exact}}(L/2) = 31.3714\text{ mm}$ | **`31.3714 mm` (`0.000000%` error)** | PASS |
| **`STRUCT_02` Physics-Informed Surrogate** | $L_2$ error $< 0.25\%$, Speedup $> 10\times$ | **`0.00182%` $L_2$ error, `22.2x` speedup vs. FEM** | PASS |
| **`STRUCT_03` Inverse $EI(x)$ Damage Identification** | $L_2$ error $< 2.5\%$ under $1.0\%$ noise | **`1.038%` $L_2$ error, localized at `11.25 m`** | PASS |
| **`STRUCT_04` $10^5$ Monte Carlo vs. FORM Reliability** | $\left|P_f^{\text{MC}} - P_f^{\text{FORM}}\right| < 0.005$ | **$P_f^{\text{MC}} = 0.03157$, $P_f^{\text{FORM}} = 0.03175$, $\beta = 1.8556$** | PASS |
| **`STRUCT_05` Sigmund (2001) SIMP Topology Opt** | $V/V_0 = 0.4500$, Compliance drop $> 55\%$ | **`80.62%` compliance reduction ($1337.2 \to 259.1$)** | PASS |

### Converged Half-MBB / Pratt Bridge Truss Topology (`30x10` Q4 Mesh, $f_v = 0.45$)
```text
  |#####################+        |
  |####+++++++++++++++###+       |
  | .+#+.          .+#+++#+.     |
  |  .+#+.        .+#+.  +#+.    |
  |    +##.      .+#+.    +#+.   |
  |     +##+    .##+       +##.  |
  |      .##+. .##+         +##. |
  |.......++#++##.           .##+|
  |#############..        ....+##|
  |##############################|
```

---

## 3. Repository Structure & Reproduction

```text
struct_surrogate_pinn_topopt/
├── run_structural_sim_and_tests.py      # Hermite FEM, PINN surrogate, inverse solver, MC/FORM & SIMP TopOpt
├── README.md                            # Mathematical formulation & verified benchmark results
├── data/
│   ├── inverse_stiffness_profile.csv    # True vs. reconstructed EI(x) flexural stiffness across 48 elements
│   └── simp_bridge_truss_density.csv    # Converged 2D SIMP element density matrix rho(x,y)
└── reports/
    └── structural_benchmark_report.json # Machine-readable verification & benchmark report
```

### Quick Start
```bash
python run_structural_sim_and_tests.py
```
