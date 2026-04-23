import unittest
import numpy as np
from scipy.optimize import approx_fprime

from optimizer import MIOptimizer 

class TestMIOptimizer(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.n = 3
        self.n_samples = 10 # Fewer samples = faster tests
        
        # Use a strongly diagonal-dominant M to stay far from det=0
        self.M = np.eye(self.n) * 2.0 + 0.1 * np.random.randn(self.n, self.n)
        self.inv_M = np.linalg.inv(self.M)
        
        # Sigmas
        A1 = np.random.randn(self.n, self.n)
        self.Sigmas = [A1 @ A1.T + np.eye(self.n)] # Add identity for PSD stability
        
        # Make C and S clearly positive and smaller to stay away from barriers
        self.C = np.abs(np.random.randn(self.n, self.n_samples)) * 0.5 + 1.0
        self.S = np.abs(np.random.randn(self.n, self.n_samples)) * 0.5 + 1.0
        
        self.mu = 0.1 # Lower mu for gradient tests makes the "cliff" less steep

    def test_main_gradient(self):
        """Proves the analytical gradient of the main objective matches finite differences."""
        # 1. Get analytical gradient
        res = MIOptimizer.main(self.M, self.Sigmas, compute_grad=True)
        grad_analytical = res.grad # type: ignore
        
        # 2. Setup finite difference wrapper
        def f_wrapper(m_flat: np.ndarray) -> float:
            M_reshaped = m_flat.reshape(self.n, self.n)
            return MIOptimizer.main(M_reshaped, self.Sigmas, compute_grad=False).f
            
        # 3. Compute numerical gradient
        grad_numerical = approx_fprime(self.M.flatten(), f_wrapper, epsilon=1e-7)
        grad_numerical = grad_numerical.reshape(self.n, self.n)

        # 4. Compare
        grad_analytical: np.ndarray
        np.testing.assert_allclose(grad_analytical, grad_numerical, rtol=1e-4, atol=1e-5,
                                   err_msg="Main objective gradient is incorrect.")

    def test_right_barrier_gradient(self):
        """Proves the right barrier gradient matches finite differences."""
        res = MIOptimizer.right_barrier(self.M, self.S, self.mu, compute_grad=True)
        grad_analytical = res.grad # type: ignore

        def f_wrapper(m_flat: np.ndarray) -> float:
            M_reshaped = m_flat.reshape(self.n, self.n)
            return MIOptimizer.right_barrier(M_reshaped, self.S, self.mu, compute_grad=False).f
            
        grad_numerical = approx_fprime(self.M.flatten(), f_wrapper, epsilon=1e-7).reshape(self.n, self.n)

        grad_analytical: np.ndarray
        np.testing.assert_allclose(grad_analytical, grad_numerical, rtol=1e-4, atol=1e-5)

    def test_left_barrier_gradient(self):
        """Proves the left barrier gradient matches finite differences."""
        res = MIOptimizer.left_barrier(self.inv_M, self.C, self.mu, compute_grad=True)
        grad_analytical = res.grad # type: ignore

        def f_wrapper(m_flat: np.ndarray) -> float:
            M_reshaped = m_flat.reshape(self.n, self.n)
            inv_M_reshaped = np.linalg.inv(M_reshaped)
            return MIOptimizer.left_barrier(inv_M_reshaped, self.C, self.mu, compute_grad=False).f
            
        grad_numerical = approx_fprime(self.M.flatten(), f_wrapper, epsilon=1e-7).reshape(self.n, self.n)

        grad_analytical: np.ndarray
        np.testing.assert_allclose(grad_analytical, grad_numerical, rtol=1e-4, atol=1e-5)

    def test_barrier_feasibility_rejection(self):
        """Ensures the barriers return np.inf if constraints are violated."""
        # Create a matrix that explicitly violates MS > 0
        bad_M = np.copy(self.M)
        bad_M[0, 0] = -100.0 
        
        res_right = MIOptimizer.right_barrier(bad_M, self.S, self.mu, compute_grad=False)
        self.assertEqual(res_right.f, np.inf, "Right barrier failed to reject negative MS")

    def test_tangent_projection(self):
        """Proves that the gradient Z is projected correctly onto the tangent space of SL(N).
        For SL(N), the trace of (M^-1 @ Z) must be exactly 0.
        """
        res = MIOptimizer.f(self.M, self.inv_M, self.Sigmas, self.C, self.S, self.mu, compute_grad=True)
        Z = res.grad # type: ignore
        Z : np.ndarray
        trace_val = np.trace(self.inv_M @ Z)
        
        # It should be 0 within machine precision
        self.assertAlmostEqual(trace_val, 0.0, places=10, 
                               msg=f"Tangent projection failed. Trace is {trace_val}")

    def test_optimize_step_descent(self):
        """Proves that taking an optimization step strictly decreases the total objective."""
        res_initial = MIOptimizer.f(self.M, self.inv_M, self.Sigmas, self.C, self.S, self.mu, compute_grad=False)
        f_old = res_initial.f
        
        success, M_new, inv_M_new = MIOptimizer.optimize_step(
            self.M, self.inv_M, self.Sigmas, self.C, self.S, self.mu, 
            initial_lr=0.1, max_line_search_iters=20
        )
        
        self.assertTrue(success, "Optimization step failed to find a valid learning rate.")
        
        res_new = MIOptimizer.f(M_new, inv_M_new, self.Sigmas, self.C, self.S, self.mu, compute_grad=False)
        f_new = res_new.f

        self.assertLess(f_new, f_old, "Objective function did not decrease after optimize_step.")


if __name__ == '__main__':
    unittest.main()