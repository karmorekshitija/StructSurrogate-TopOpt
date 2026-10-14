"""Generate concise reports and SVG figures from saved numerical evidence."""

import csv
import html
import json
import math
from pathlib import Path


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
BASE = SCRIPT_DIRECTORY.parent if SCRIPT_DIRECTORY.name in ("macroflow_lwr_traffic_simulator", "flood_swe_hydraulics_simulator", "struct_surrogate_pinn_topopt") else SCRIPT_DIRECTORY


def read_report(project, filename):
    return json.loads((BASE / project / "reports" / filename).read_text(encoding="utf-8"))


def plot(path, title, xlabel, ylabel, series):
    all_points = [point for _, points in series for point in points]
    if not all_points or any(not math.isfinite(value) for point in all_points for value in point):
        raise ValueError("Plot needs finite points")
    xmin, xmax = min(point[0] for point in all_points), max(point[0] for point in all_points)
    ymin, ymax = min(0, min(point[1] for point in all_points)), max(point[1] for point in all_points)
    if xmax == xmin:
        xmax = xmin+1
    if ymax == ymin:
        ymax = ymin+1
    ymax += (ymax-ymin)*0.08
    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="550" viewBox="0 0 900 550"><rect width="900" height="550" fill="white"/><g font-family="Arial,sans-serif" font-size="14" fill="#172b4d">',
                f'<text x="450" y="30" text-anchor="middle" font-size="20">{html.escape(title)}</text>']
    for tick in range(6):
        horizontal = 90+tick*150
        vertical = 430-tick*68
        elements.extend([
            f'<path d="M90 {vertical} H840" stroke="#dde2e8"/>',
            f'<text x="78" y="{vertical+5}" text-anchor="end">{ymin+(ymax-ymin)*tick/5:.3g}</text>',
            f'<text x="{horizontal}" y="455" text-anchor="middle">{xmin+(xmax-xmin)*tick/5:.4g}</text>',
        ])
    elements.append('<path d="M90 90 V430 H840" fill="none" stroke="#172b4d"/>')
    colors = ("#1768ac", "#c44536", "#23845c", "#7d4aa6")
    for index, (label, points) in enumerate(series):
        color = colors[index % len(colors)]
        scaled = [(90+750*(xvalue-xmin)/(xmax-xmin), 430-340*(yvalue-ymin)/(ymax-ymin)) for xvalue, yvalue in points]
        coordinates = " ".join(f"{xvalue:.2f},{yvalue:.2f}" for xvalue, yvalue in scaled)
        elements.append(f'<polyline points="{coordinates}" stroke="{color}" stroke-width="2" fill="none"/>')
        if len(points) < 100:
            elements.extend(f'<circle cx="{xvalue:.2f}" cy="{yvalue:.2f}" r="3" fill="{color}"/>' for xvalue, yvalue in scaled)
        elements.append(f'<text x="{100+index*240}" y="65" fill="{color}">{html.escape(label)}</text>')
    elements.extend([f'<text x="450" y="495" text-anchor="middle">{html.escape(xlabel)}</text>',
                     f'<text transform="translate(20,270) rotate(-90)" text-anchor="middle">{html.escape(ylabel)}</text>',
                     '<text x="450" y="530" text-anchor="middle" fill="#555">Reproducible numerical evidence; not field-performance validation</text></g></svg>'])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements), encoding="utf-8")


def transportation():
    project = "macroflow_lwr_traffic_simulator"
    report = read_report(project, "traffic_simulation_report.json")
    ingestion = read_report(project, "lankershim_ingestion.json")
    output = BASE / project / "reports"
    refinement = report["shock_grid_refinement"]
    fair = report["prospective_equal_budget_comparison"]
    plot(output/"shock_refinement.svg", "Analytical shock-front verification", "Number of cells", "Shock speed error (%)",
         [("Tracked front", [(row["num_cells"], row["relative_error_pct"]) for row in refinement])])
    plot(output/"equal_budget_delay.svg", "Synthetic equal-green comparison", "Demand multiplier", "Window delay (vehicle-hours)",
         [(label, [(row["demand_scale"], row[key]["total_delay_veh_h"]) for row in fair]) for label, key in (("Fixed phase", "fixed"), ("Causal phase", "causal"))])
    sensitivity = report["demand_sensitivity"]
    plot(output/"demand_sensitivity.svg", "Synthetic fixed-signal sensitivity", "Realized green fraction", "Window delay (vehicle-hours)",
         [(f"Demand {scale}x", [(row["green_split_ratio"], row["total_delay_veh_h"]) for row in sensitivity if row["demand_scale"] == scale]) for scale in (0.75, 1.0, 1.25)])
    audit = ingestion["audit"]
    text = f"""# Transportation research report

## Question and positioning
Can a compact CTM reproduce queue growth and discharge on an observed approach? The implemented numerical study is a synthetic precursor; observed calibration is not completed. Earlier CTM/Lankershim work means the defensible direction is replication and failure analysis, not a new model.

## Method
The current synthetic corridor uses 40 cells over 2 km, a 2-second step and a 90-second signal cycle. Supply-demand fluxes conserve vehicles; rejected arrivals remain in an external queue. Queue extent is the summed length of cells above 1.15 times diagram-specific critical density, not necessarily one connected stopped queue. Greenshields uses its own capacity and critical density. Equal-budget comparisons give both policies the same realized green seconds. No cross-street, amber, turning or spillback model is validated.

Window delay is occupancy-time plus external waiting minus free-flow time for distance actually covered. It is not completed-trip delay. The historical adaptive-versus-fixed percentage improvements spend different green budgets and must not be used as fair controller-performance claims.

## Results
- Tracked shock-speed errors for 40/80/160 cells: {', '.join(str(row['relative_error_pct'])+'%' for row in refinement)}. This supports numerical refinement, not a claimed formal convergence order.
- Causal-minus-fixed delay at demand 0.75/1/1.25: {', '.join(format(row['causal_minus_fixed_delay_veh_h'], '.4f') for row in fair)} vehicle-hours. These are small scenario-specific differences, not evidence of general superiority.
- Acquired {audit['rows']:,} northbound section-3 records and verified the official count. Temporal filtering retains {audit['partitions']['calibration']:,} calibration rows and {audit['partitions']['evaluation']:,} evaluation rows, excluding {audit['partitions']['excluded']:,}. There are {audit['duplicate_track_time_keys']} duplicate track/time keys and {audit['within_subset_gaps_over_100ms']} within-subset gaps over 100 ms. Gaps may reflect leaving/re-entering the section; completeness is not established by counts alone.

## Unfinished observed-data gate
The official support archive is unreadable. Exact stop-line geometry and actual RSM phase histories remain unavailable. The metadata suggests a 26-second clock offset, not yet checked against timing files. No signal schedule is invented from held-out queues. The selected section is an ingestion candidate, not a verified full approach. Calibration, baseline scoring, observed residual plots and empirical conclusions are blocked by these inputs. File-specific licensing remains unresolved; raw data stay ignored and local.

## Reproduce and inspect
Run `python -B run_traffic_sim_and_tests.py`, `python -B -m unittest discover -s tests -q`, and from this folder `python -B build_research_reports.py`. Network acquisition is a separate `python -B ingest_lankershim.py` command. See `lankershim_ingestion.json`, `RESEARCH_PROTOCOL.md`, `DATA_FEASIBILITY.md` and `RESEARCH_OWNERSHIP.md`.

Figures: `shock_refinement.svg`, `equal_budget_delay.svg`, `demand_sensitivity.svg`. These show analytical/synthetic evidence only.

## Reference and contribution boundary
[Su, Kurzhanskiy and Horowitz (2013)](https://horowitz.me.berkeley.edu/Publications_files/All_papers_numbered/Su_TRB_2013_Simulation_of_Cell_Transmission_Model_Using_NGSIM_data.pdf) motivates replication. [FHWA source description](https://www.fhwa.dot.gov/publications/research/operations/07029/index.cfm) identifies supporting data. The agent implemented substantial changes; student derivation, reproduction and interpretation exercises remain unconfirmed. Do not present this as independently completed student research yet.
"""
    (output/"RESEARCH_REPORT.md").write_text(text, encoding="utf-8")


def hydraulics():
    project = "flood_swe_hydraulics_simulator"
    report = read_report(project, "hydraulics_benchmark_report.json")
    uncertainty = read_report(project, "roughness_uncertainty.json")
    output = BASE/project/"reports"
    with (BASE/project/"data"/"saint_venant_dambreak_profile.csv").open(encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    plot(output/"dam_break_profile.svg", "Wet-bed dam break: analytical verification", "Position (m)", "Water depth (m)",
         [(label, [(float(row["x_m"]), float(row[key])) for row in rows]) for label, key in (("Rusanov", "h_rusanov_fvm_m"), ("Stoker", "h_stoker_exact_m"))])
    series = []
    for factor in (1.0, 1.1):
        for layout in ("distributed", "near_boundary"):
            points = []
            for noise in (0.0, 0.002, 0.01):
                values = [abs(row["signed_n_error_pct"]) for row in uncertainty["cases"] if row["true_slope_factor"] == factor and row["sensor_layout"] == layout and row["noise_std_m"] == noise]
                points.append((noise*1000, sum(values)/len(values)))
            series.append((f"{layout}, slope {factor}x", points))
    plot(output/"roughness_uncertainty.svg", "Synthetic roughness error: four-seed means", "Observation noise standard deviation (mm)", "Mean absolute roughness error (%)", series[:2])
    plot(output/"roughness_mismatch.svg", "Synthetic roughness error: 10% slope mismatch", "Observation noise standard deviation (mm)", "Mean absolute roughness error (%)", series[2:])
    swe = report["saint_venant_dambreak"]
    errors = [abs(row["signed_n_error_pct"]) for row in uncertainty["cases"]]
    text = f"""# Hydraulics research report

## Question and method
How do discretization and imperfect observations affect wet-bed flow verification and roughness recovery? A first-order Rusanov solver uses current-state characteristic speeds to select every timestep (target CFL 0.45), clips the last step to the end time, and accumulates boundary mass flux. Supported inputs require h_left >= h_right >= 1e-6 m. Dry beds and critical/supercritical GVF are explicitly unsupported rather than silently clipped.

## Numerical evidence
The 1 km, 4 m/0.4 m wet-bed dam break at 30 seconds has {swe['l2_depth_error_vs_stoker_pct']}% relative depth-profile error at 500 cells and {swe['shock_speed_relative_error_pct']}% measured shock-speed error. Profile errors at 125/250/500 cells are {', '.join(str(row['l2_depth_error_pct'])+'%' for row in report['grid_convergence'])}. Maximum CFL is {swe['max_cfl_number']}; the mass-balance residual including boundary flux is {swe['volume_conservation_error_m3_per_m']:.3g} m3 per metre width. Open-boundary and rest-state regression cases also pass. The infinite-domain analytical comparison is flagged inapplicable after waves reach boundaries.

## Roughness sensitivity
The original near-exact recovery is only one same-family synthetic exercise. The new study contains {len(uncertainty['cases'])} cases: four seeds, three noise levels, two sensor layouts and two slope assumptions. Truth uses 600 integration steps; fitting uses 150. The optional 10% slope mismatch is deliberately not fitted. Absolute roughness error ranges from {min(errors):.4g}% to {max(errors):.4g}%. Reported observation-fit RMSE and synthetic true-profile RMSE are distinct. Four seeds are a sensitivity check, not a calibrated confidence interval or proof of identifiability. Poor sensor placement and slope uncertainty can dominate measurement noise.

## Supporting detention exercise
An invented rainfall stress scenario uses midpoint RK2 level-pool routing, not Modified Puls. Outflow is capped by available storage and inflow. The saved budget residual is {report['synthetic_storm_and_detention']['storage_balance_error_m3']:.3g} m3. No observed storm, climate projection or engineering design adequacy is established.

## Evidence and reproduction
Reference equations and wet-bed wave structure: [Clawpack Riemann book](https://www.clawpack.org/riemann_book/html/Shallow_water.html); classic analytical benchmark: Stoker, as discussed in Toro (2001). This is analytical verification, not an independent experimental dataset or externally executed solver comparison.

Run `python -B run_hydraulics_sim_and_tests.py`, `python -B roughness_study.py` and `python -B -m unittest discover -s tests -q`. Generate figures with the `build_research_reports.py` in this folder. Inputs and all 48 outcomes are retained in `roughness_uncertainty.json`. Figures: `dam_break_profile.svg`, `roughness_uncertainty.svg`, `roughness_mismatch.svg`.

Agent-assisted implementation; student explanation and reproduction remain to be demonstrated. No field-validation CV wording is supported.
"""
    (output/"RESEARCH_REPORT.md").write_text(text, encoding="utf-8")


def structures():
    project = "struct_surrogate_pinn_topopt"
    report = read_report(project, "structural_benchmark_report.json")
    study = read_report(project, "inverse_sensitivity.json")
    output = BASE/project/"reports"
    with (BASE/project/"data"/"inverse_stiffness_profile.csv").open(encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    plot(output/"inverse_profile.svg", "Original dense-sensor synthetic example", "Beam position (m)", "Flexural stiffness (GN m2)",
         [(label, [(float(row["x_mid_m"]), float(row[key])) for row in rows]) for label, key in (("Synthetic truth", "ei_true_GNm2"), ("Reconstructed", "ei_reconstructed_GNm2"))])
    plot(output/"inverse_failures.svg", "Sparse-sensor synthetic cases (all retained)", "Predeclared case index", "Stiffness profile error (%)",
         [(label, [(index+1, row[key]) for index, row in enumerate(study["cases"])]) for label, key in (("Inverse estimate", "stiffness_profile_l2_error_pct"), ("Intact baseline", "baseline_intact_profile_l2_error_pct"))])
    topology = report["simp_topology_optimization"]
    plot(output/"topology_history.svg", "SIMP objective history (not proof of convergence)", "Density update count", "Nondimensional compliance",
         [("Evaluated density state", list(enumerate(topology["compliance_history"])))])
    reliability = report["monte_carlo_form_reliability"]
    text = f"""# Structures research report

## Question and method
How sensitive is stiffness recovery to sensor placement, noise and incorrect loading? The supporting forward model is a simply supported Euler-Bernoulli beam with Hermite cubic elements, uniform load and a midspan point load. Even meshes are required to place that point load exactly. Essential boundary conditions are eliminated, not imposed through a large penalty. Linear solvers reject singular/non-SPD systems rather than hide them with pivot clipping. Beam reference and force/reaction checks cover multiple meshes and zero load.

## Inverse results and failures
The original 48-sensor same-model curvature example has {report['inverse_damage_identification']['stiffness_profile_l2_error_pct']}% stiffness-profile error. The new eight-sensor study runs {len(study['cases'])} cases across distributed/clustered sensors, two damage locations, three noise levels, three seeds and correct/3%-biased bending moments. Regularization is fixed at the inherited 1.2e-6; no held-out truth is used to choose it. The intact-stiffness baseline uses no damage information.

**{study['cases_worse_than_intact_baseline']} of {len(study['cases'])} cases are worse than the intact baseline.** Thus successful dense-sensor recovery does not imply reliable sparse damage localization. Compliance bounds are reported as clipping counts, not hidden. Noise sigma is relative to spatial curvature standard deviation, not a percentage of each sensor's reading. Synthetic curvature is algebraically generated; there is no independent experimental damage benchmark.

## Supporting experiments
- Spectral approximation is a fitted linear sine basis, not a PINN. The 3/9/15-mode study reports errors, setup time and seven timing batches after warmup. Cached 15-sensor inference and full FEM assembly/solve/diagnostics are different workloads; their ratio is not an end-to-end acceleration claim or correctness criterion.
- Monte Carlo probability is {reliability['monte_carlo_failure_probability']}; Wilson 95% interval is {reliability['wilson_95_interval']}. FORM is {reliability['form_failure_probability']}. A linear-Gaussian event reference is {reliability['linear_gaussian_limit_state_exact_probability']:.7f}; Monte Carlo's stiffness floor is separately disclosed. Zero-failure tests return an upper bound rather than a proven zero probability.
- SIMP uses unit point load, left-edge horizontal restraints and a bottom-right vertical restraint on a 30x10 Q4 mesh, volume 0.45, penalty 3 and filter radius 1.4. Compliance is evaluated on the exported final density, fixing an off-by-one mismatch. Final compliance is {topology['final_compliance']}, achieved volume {topology['achieved_volume_fraction']}, termination `{topology['termination_reason']}`, last density change {topology['last_max_density_change']:.4g}. This is not an independently matched published optimization result or design certification.

## Reproduce and references
Run `python -B run_structural_sim_and_tests.py`, `python -B inverse_study.py` and `python -B -m unittest discover -s tests -q`; `build_research_reports.py` in this folder builds figures. Evidence: `structural_benchmark_report.json`, `inverse_sensitivity.json`, `inverse_profile.svg`, `inverse_failures.svg`, `topology_history.svg`.

[TU Delft beam formulation](https://teachbooks.tudelft.nl/computational-modelling/structural_linear/euler_bernouilli.html) and [DTU's educational SIMP reference](https://www.topopt.mek.dtu.dk/apps-and-software/a-99-line-topology-optimization-code-written-in-matlab) ground methods, not empirical accuracy. Student ownership/reproduction remains pending; these are agent-assisted educational numerical studies.
"""
    (output/"RESEARCH_REPORT.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    builders = {"macroflow_lwr_traffic_simulator": transportation, "flood_swe_hydraulics_simulator": hydraulics, "struct_surrogate_pinn_topopt": structures}
    selected = [builders[SCRIPT_DIRECTORY.name]] if SCRIPT_DIRECTORY.name in builders else list(builders.values())
    for builder in selected:
        builder()
    print(f"Generated {len(selected)} research reports and {3*len(selected)} SVG figures")
