# Structures: beam verification and inverse-stiffness sensitivity

This standard-library Python project contains Hermite beam FEM, a linear spectral approximation, synthetic curvature inversion, reliability exercises and a SIMP demonstration. It is not a PINN and does not claim an exact paper reproduction.

## What is checked

- Simply supported beam deflection, reactions, equilibrium, mesh choices and zero loads.
- Singular/non-SPD systems and invalid stiffness inputs fail explicitly.
- Seventy-two sparse-sensor inverse cases vary location, noise, sensor placement and a 3% moment mismatch. Regularization is fixed; 21 cases are worse than the intact-stiffness baseline.
- Monte Carlo/FORM outputs include an analytical linear-Gaussian event reference, a Wilson interval and a zero-failure upper bound.
- Spectral setup and inference timings are separate. Cached sensor inference is not equivalent to a complete FEM solve and timing is not a correctness threshold.
- SIMP final compliance corresponds to exported density. Termination reason and density change are reported; reaching an iteration limit is not convergence.

## Run

```text
python -B -m unittest discover -s tests -q
python -B run_structural_sim_and_tests.py
python -B inverse_study.py
```

From this directory, run `python -B build_research_reports.py` to generate reports and figures.

Read `reports/RESEARCH_REPORT.md`, `reports/structural_benchmark_report.json` and `reports/inverse_sensitivity.json`. Figures are `inverse_profile.svg`, `inverse_failures.svg` and `topology_history.svg` in `reports/`.

All measurements are synthetic. The inverse solver uses curvature, not measured deflection. Noise is a fraction of spatial curvature standard deviation. Compliance clipping counts are retained. No experimental damage validation, robust localization guarantee, optimized real bridge design or authorship certification is implied.

Method references: [TU Delft beam formulation](https://teachbooks.tudelft.nl/computational-modelling/structural_linear/euler_bernouilli.html), [DTU educational SIMP](https://www.topopt.mek.dtu.dk/apps-and-software/a-99-line-topology-optimization-code-written-in-matlab). Student explanation and independent reproduction remain pending.
