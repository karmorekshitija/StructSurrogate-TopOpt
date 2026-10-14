# Structures research report

## Question and method
How sensitive is stiffness recovery to sensor placement, noise and incorrect loading? The supporting forward model is a simply supported Euler-Bernoulli beam with Hermite cubic elements, uniform load and a midspan point load. Even meshes are required to place that point load exactly. Essential boundary conditions are eliminated, not imposed through a large penalty. Linear solvers reject singular/non-SPD systems rather than hide them with pivot clipping. Beam reference and force/reaction checks cover multiple meshes and zero load.

## Inverse results and failures
The original 48-sensor same-model curvature example has 1.038% stiffness-profile error. The new eight-sensor study runs 72 cases across distributed/clustered sensors, two damage locations, three noise levels, three seeds and correct/3%-biased bending moments. Regularization is fixed at the inherited 1.2e-6; no held-out truth is used to choose it. The intact-stiffness baseline uses no damage information.

**21 of 72 cases are worse than the intact baseline.** Thus successful dense-sensor recovery does not imply reliable sparse damage localization. Compliance bounds are reported as clipping counts, not hidden. Noise sigma is relative to spatial curvature standard deviation, not a percentage of each sensor's reading. Synthetic curvature is algebraically generated; there is no independent experimental damage benchmark.

## Supporting experiments
- Spectral approximation is a fitted linear sine basis, not a PINN. The 3/9/15-mode study reports errors, setup time and seven timing batches after warmup. Cached 15-sensor inference and full FEM assembly/solve/diagnostics are different workloads; their ratio is not an end-to-end acceleration claim or correctness criterion.
- Monte Carlo probability is 0.03157; Wilson 95% interval is [0.03050412001837836, 0.03267186901282701]. FORM is 0.031752. A linear-Gaussian event reference is 0.0317520; Monte Carlo's stiffness floor is separately disclosed. Zero-failure tests return an upper bound rather than a proven zero probability.
- SIMP uses unit point load, left-edge horizontal restraints and a bottom-right vertical restraint on a 30x10 Q4 mesh, volume 0.45, penalty 3 and filter radius 1.4. Compliance is evaluated on the exported final density, fixing an off-by-one mismatch. Final compliance is 258.9393, achieved volume 0.45, termination `iteration_limit`, last density change 0.01546. This is not an independently matched published optimization result or design certification.

## Reproduce and references
Run `python -B run_structural_sim_and_tests.py`, `python -B inverse_study.py` and `python -B -m unittest discover -s tests -q`; `build_research_reports.py` in this folder builds figures. Evidence: `structural_benchmark_report.json`, `inverse_sensitivity.json`, `inverse_profile.svg`, `inverse_failures.svg`, `topology_history.svg`.

[TU Delft beam formulation](https://teachbooks.tudelft.nl/computational-modelling/structural_linear/euler_bernouilli.html) and [DTU's educational SIMP reference](https://www.topopt.mek.dtu.dk/apps-and-software/a-99-line-topology-optimization-code-written-in-matlab) ground methods, not empirical accuracy. Student ownership/reproduction remains pending; these are agent-assisted educational numerical studies.
