import math
import unittest

from run_structural_sim_and_tests import (
    EulerBernoulliBridgeSolver, run_monte_carlo_and_form_reliability,
    run_simp_bridge_topology_optimization, solve_banded_spd, solve_dense_linear,
)


class StructureTests(unittest.TestCase):
    def test_beam_reference_and_equilibrium(self):
        for elements in (4, 12, 48):
            solver = EulerBernoulliBridgeSolver(num_elements=elements)
            displacement, _, _ = solver.solve_fem([solver.ei0]*elements)
            self.assertAlmostEqual(displacement[elements//2], solver.exact_intact_midspan_deflection_m(), places=9)
            self.assertEqual(displacement[0], 0)
            self.assertEqual(displacement[-1], 0)
            self.assertLess(abs(solver.last_equilibrium["force_balance_residual_n"]), 0.01)
            for reaction in solver.last_equilibrium["support_reactions_n"]:
                self.assertAlmostEqual(reaction, (solver.q0*solver.L+solver.P)/2, delta=0.01)

    def test_zero_loading(self):
        solver = EulerBernoulliBridgeSolver(num_elements=4, q_dead_n_per_m=0, point_load_n=0)
        displacement, _, _ = solver.solve_fem([solver.ei0]*4)
        self.assertEqual(displacement, [0]*5)

    def test_invalid_models_and_singular_systems(self):
        with self.assertRaises(ValueError):
            EulerBernoulliBridgeSolver(num_elements=5)
        with self.assertRaises(ValueError):
            EulerBernoulliBridgeSolver().solve_fem([0]*48)
        with self.assertRaises(ValueError):
            solve_dense_linear([[1, 1], [1, 1]], [1, 1])
        with self.assertRaises(ValueError):
            solve_banded_spd([[0]], [1], 0)

    def test_linear_solvers_match_hand_solution(self):
        self.assertEqual(solve_dense_linear([[2, 0], [0, 4]], [6, 8]), [3, 2])
        result = solve_banded_spd([[2, 1], [2, 0]], [3, 3], 1)
        self.assertAlmostEqual(result[0], 1)
        self.assertAlmostEqual(result[1], 1)

    def test_reliability_reference_and_zero_failure_bound(self):
        result = run_monte_carlo_and_form_reliability(num_trials=10000)
        self.assertAlmostEqual(result["form_failure_probability"], result["linear_gaussian_limit_state_exact_probability"], places=5)
        zero = run_monte_carlo_and_form_reliability(span_m=1, num_trials=100)
        self.assertEqual(zero["failure_count"], 0)
        self.assertAlmostEqual(zero["zero_failure_95_upper_bound"], 1-0.05**0.01)

    def test_simp_reports_iteration_limit_and_matching_state(self):
        result = run_simp_bridge_topology_optimization(nelx=6, nely=2, max_iter=2)
        self.assertEqual(result["iterations_executed"], 2)
        self.assertEqual(len(result["compliance_history"]), 3)
        self.assertFalse(result["converged"])
        self.assertLess(abs(result["achieved_volume_fraction"]-0.45), 0.001)
        self.assertTrue(all(math.isfinite(value) for row in result["density_grid"] for value in row))

    def test_sparse_inverse_records_baseline(self):
        solver = EulerBernoulliBridgeSolver()
        result = solver.solve_inverse_damage_identification(sensor_indices=[4, 12, 24, 36, 44])
        self.assertEqual(len(result["sensor_indices"]), 5)
        self.assertGreater(result["baseline_intact_profile_l2_error_pct"], 0)
        with self.assertRaises(ValueError):
            solver.solve_inverse_damage_identification(sensor_indices=[1, 1])
