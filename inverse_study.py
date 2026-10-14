"""Predeclared synthetic inverse sensitivity cases; includes unsuccessful fits."""

import json
from pathlib import Path

from run_structural_sim_and_tests import EulerBernoulliBridgeSolver


def main():
    solver = EulerBernoulliBridgeSolver()
    cases = []
    for layout, sensors in (("distributed_8", list(range(2, 48, 6))), ("left_cluster_8", list(range(4, 12)))):
        for noise in (0.0, 0.01, 0.05):
            for centre in (6.0, 17.0):
                for moment_bias in (0.0, 0.03):
                    for seed in range(3):
                        result = solver.solve_inverse_damage_identification(seed=seed, noise_fraction=noise, sensor_indices=sensors, damage_center=centre, moment_bias=moment_bias)
                        result["layout"] = layout
                        result["requested_damage_center_m"] = centre
                        cases.append({key: value for key, value in result.items() if key not in ("x_mid_m", "ei_true_gnm2", "ei_recon_gnm2")})
    spectral = [solver.train_physics_informed_spectral_surrogate(n_modes=modes) for modes in (3, 9, 15)]
    report = {
        "scope": "72 synthetic curvature-inversion cases; no field sensors or neural network",
        "regularization_policy": "Fixed 1.2e-6 inherited from original exercise; not tuned on this sweep. Not claimed optimal.",
        "noise_definition": "Gaussian sigma is fraction of spatial true curvature standard deviation",
        "mismatch_definition": "3% true bending-moment scaling is omitted from inverse forward model",
        "baseline": "Assume intact stiffness everywhere; no observations or damage truth used",
        "cases": cases, "spectral_mode_study": spectral,
        "cases_worse_than_intact_baseline": sum(case["stiffness_profile_l2_error_pct"] > case["baseline_intact_profile_l2_error_pct"] for case in cases),
    }
    output = Path(__file__).resolve().parent / "reports" / "inverse_sensitivity.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Completed {len(cases)} cases; {report['cases_worse_than_intact_baseline']} worse than intact baseline")


if __name__ == "__main__":
    main()
