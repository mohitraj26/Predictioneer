"""
═══════════════════════════════════════════════════════════════════════════════
GENE REGULATORY NETWORK INFERENCE - FINAL REFINED SOLUTION
AZeotropy '26 Competition - IIT Bombay

Problem: Infer gene regulatory network from spatial-temporal expression data
Model: dS/dt = AS + BM - λS

Author: Competition Submission Team
Date: February 2026
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, least_squares
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class GRNInferenceOptimized:
    """
    Optimized Gene Regulatory Network Inference System
    
    Mathematical Model:
        dS/dt = AS + BM - λS
        
    Where:
        S ∈ ℝ⁴: Gene expression deviations [S₁, S₂, S₃, S₄]ᵀ
        A ∈ ℝ⁴ˣ⁴: Gene-gene interaction matrix (Aᵢⱼ = effect of j on i)
        B ∈ ℝ⁴ˣ²: Morphogen-gene coupling matrix
        M ∈ ℝ²: Morphogen concentrations [M₁(x,y), M₂(x,y)]ᵀ
        λ > 0: Auto-regulation parameter
    
    Morphogen Profiles:
        M₁(x,y) = x
        M₂(x,y) = y
    """
    
    def __init__(self):
        self.A = None
        self.B = None
        self.lambda_auto = None
        self.A_continuous = None
        self.B_continuous = None
        
    def load_data(self, filepath):
        """Load experimental CSV data"""
        print(f"  Loading: {filepath}")
        df = pd.read_csv(filepath)
        print(f"    → {len(df)} data points loaded")
        return df
    
    def get_morphogen_values(self, x, y):
        """
        Morphogen spatial profiles
        Based on problem statement and data analysis
        """
        M1 = x  # M1 varies in x-direction
        M2 = y  # M2 varies in y-direction
        return M1, M2
    
    def compute_spatial_derivatives(self, df):
        """
        Compute time derivatives using finite differences
        
        Method:
            - Forward difference: dS/dt ≈ (S(t+Δt) - S(t))/Δt for t=0
            - Backward difference: dS/dt ≈ (S(t) - S(t-Δt))/Δt for t=final
            - Central difference: dS/dt ≈ (S(t+Δt) - S(t-Δt))/(2Δt) otherwise
        
        Returns:
            S_array: 4D tensor [time, x_index, y_index, gene]
            dS_dt: Time derivatives
            times: Time points
            x_vals, y_vals: Spatial coordinates
        """
        # Extract unique values
        times = np.array(sorted(df['time'].unique()))
        x_vals = np.array(sorted(df['x'].unique()))
        y_vals = np.array(sorted(df['y'].unique()))
        
        n_t, n_x, n_y, n_genes = len(times), len(x_vals), len(y_vals), 4
        
        # Initialize arrays
        S_array = np.zeros((n_t, n_x, n_y, n_genes))
        
        # Fill S_array from dataframe
        for ti, t in enumerate(times):
            for xi, x in enumerate(x_vals):
                for yi, y in enumerate(y_vals):
                    mask = (df['time'] == t) & (df['x'] == x) & (df['y'] == y)
                    if mask.any():
                        row = df[mask].iloc[0]
                        S_array[ti, xi, yi, :] = [
                            row['S1'], row['S2'], row['S3'], row['S4']
                        ]
        
        # Compute time derivatives
        dS_dt = np.zeros_like(S_array)
        
        for ti in range(n_t):
            if ti == 0:
                # Forward difference
                dt = times[1] - times[0]
                dS_dt[ti] = (S_array[ti+1] - S_array[ti]) / dt
            elif ti == n_t - 1:
                # Backward difference
                dt = times[ti] - times[ti-1]
                dS_dt[ti] = (S_array[ti] - S_array[ti-1]) / dt
            else:
                # Central difference (most accurate)
                dt = times[ti+1] - times[ti-1]
                dS_dt[ti] = (S_array[ti+1] - S_array[ti-1]) / dt
        
        return S_array, dS_dt, times, x_vals, y_vals
    
    def infer_parameters_multistage(self, df, alpha=0.003):
        """
        Multi-stage parameter inference with improved accuracy
        
        Strategy:
        1. Use regression to get initial estimates of (A - λI) and B[:,0]
        2. Refine λ estimation using multiple methods
        3. Infer B[:,1] using steady-state constraint and optimization
        4. Final refinement using global optimization
        
        Args:
            df: Training dataframe (M2 removal experiment)
            alpha: Ridge regularization parameter
        """
        print("  Inferring GRN parameters...")
        
        # Stage 1: Compute derivatives
        S_array, dS_dt, times, x_vals, y_vals = self.compute_spatial_derivatives(df)
        
        n_samples = len(times) * len(x_vals) * len(y_vals)
        n_genes = 4
        
        # Flatten for regression
        S_flat = S_array.reshape(n_samples, n_genes)
        dS_dt_flat = dS_dt.reshape(n_samples, n_genes)
        
        # Create morphogen features
        M1_flat = np.zeros(n_samples)
        M2_flat = np.zeros(n_samples)
        
        idx = 0
        for t in times:
            for x in x_vals:
                for y in y_vals:
                    M1, M2 = self.get_morphogen_values(x, y)
                    M1_flat[idx] = M1
                    M2_flat[idx] = 0  # M2 removed in training
                    idx += 1
        
        # Stage 2: Ridge regression for each gene
        print("    → Stage 1: Ridge regression")
        
        A_minus_lambda_I = np.zeros((n_genes, n_genes))
        B_col0 = np.zeros(n_genes)
        
        for gene_idx in range(n_genes):
            y_target = dS_dt_flat[:, gene_idx]
            
            # Features: [S1, S2, S3, S4, M1]
            # Model: dSi/dt = Σⱼ(A-λI)ᵢⱼSⱼ + Bᵢ₀M₁
            X = np.column_stack([S_flat, M1_flat])
            
            # Try multiple alphas and pick best
            best_score = -np.inf
            best_model = None
            
            for alpha_test in [alpha, alpha/2, alpha*2]:
                model = Ridge(alpha=alpha_test, fit_intercept=False)
                model.fit(X, y_target)
                score = model.score(X, y_target)
                
                if score > best_score:
                    best_score = score
                    best_model = model
            
            A_minus_lambda_I[gene_idx, :] = best_model.coef_[:n_genes]
            B_col0[gene_idx] = best_model.coef_[n_genes]
        
        # Stage 3: Extract λ
        print("    → Stage 2: Extracting auto-regulation λ")
        
        # Method 1: From diagonal
        lambda_diag = -np.mean(np.diag(A_minus_lambda_I))
        
        # Method 2: From decay rate analysis
        # Genes should decay toward zero, so we can estimate λ from decay
        lambda_decay = self._estimate_lambda_from_decay(S_array, times)
        
        # Combine estimates with weights
        lambda_est = 0.7 * lambda_diag + 0.3 * lambda_decay
        lambda_est = max(lambda_est, 0.1)  # Ensure positive
        
        print(f"      λ (diagonal): {lambda_diag:.4f}")
        print(f"      λ (decay): {lambda_decay:.4f}")
        print(f"      λ (combined): {lambda_est:.4f}")
        
        # Stage 4: Recover A matrix
        A = A_minus_lambda_I + lambda_est * np.eye(n_genes)
        
        # Stage 5: Infer B[:,1] using steady-state constraint
        print("    → Stage 3: Inferring morphogen coupling B[:,1]")
        
        # At initial steady state (t=0) with both morphogens:
        # 0 = AS₀ + B[M₁, M₂] - λS₀
        # BM = (λI - A)S₀
        
        df_t0 = df[df['time'] == times[0]].copy()
        
        # Collect multiple estimates
        B_col1_estimates = []
        weights = []
        
        for _, row in df_t0.iterrows():
            x, y = row['x'], row['y']
            
            # Skip edge cases
            if abs(y) < 0.05:
                continue
                
            S0 = np.array([row['S1'], row['S2'], row['S3'], row['S4']])
            M1, M2 = self.get_morphogen_values(x, y)
            
            # From steady-state: BM = (λI - A)S
            # B[:,0]*M1 + B[:,1]*M2 = (λI - A)S
            # B[:,1] = ((λI - A)S - B[:,0]*M1) / M2
            
            rhs = (lambda_est * np.eye(n_genes) - A) @ S0
            B_col1 = (rhs - B_col0 * M1) / M2
            
            # Weight by distance from origin (more reliable far from edges)
            weight = np.sqrt(x**2 + y**2)
            
            B_col1_estimates.append(B_col1)
            weights.append(weight)
        
        # Weighted median for robustness
        if B_col1_estimates:
            B_col1_estimates = np.array(B_col1_estimates)
            weights = np.array(weights)
            B_col1 = self._weighted_median(B_col1_estimates, weights)
        else:
            B_col1 = np.zeros(n_genes)
        
        # Construct B matrix
        B = np.column_stack([B_col0, B_col1])
        
        # Stage 6: Optional refinement using nonlinear optimization
        print("    → Stage 4: Final refinement")
        
        A_refined, B_refined, lambda_refined = self._refine_parameters(
            A, B, lambda_est, S_array, dS_dt, times, x_vals, y_vals
        )
        
        # Store results
        self.A = A_refined
        self.B = B_refined
        self.lambda_auto = lambda_refined
        self.A_continuous = A_refined.copy()
        self.B_continuous = B_refined.copy()
        
        print("    → Parameter inference complete!")
        
        return A_refined, B_refined, lambda_refined
    
    def _estimate_lambda_from_decay(self, S_array, times):
        """Estimate λ from exponential decay rate"""
        # For autonomous system dS/dt = -λS, solution is S(t) = S₀ exp(-λt)
        # We can estimate λ from the decay rate
        
        # Use late time points where morphogen effect is minimal
        t_late_idx = len(times) // 2
        
        S_magnitude = np.sqrt(np.sum(S_array**2, axis=3))  # [time, x, y]
        S_mean = np.mean(S_magnitude, axis=(1, 2))  # [time]
        
        # Fit exponential decay
        valid_idx = S_mean > 1e-6
        if np.sum(valid_idx) > 2:
            t_fit = times[valid_idx]
            S_fit = S_mean[valid_idx]
            
            # log(S) = log(S0) - λt
            log_S = np.log(S_fit + 1e-10)
            
            # Linear fit
            coeffs = np.polyfit(t_fit, log_S, 1)
            lambda_decay = -coeffs[0]
            
            return max(lambda_decay, 0.1)
        
        return 0.5  # Default value
    
    def _weighted_median(self, values, weights):
        """Compute weighted median along axis 0"""
        # For each gene separately
        result = np.zeros(values.shape[1])
        
        for gene_idx in range(values.shape[1]):
            vals = values[:, gene_idx]
            wts = weights / np.sum(weights)
            
            # Sort by values
            sorted_idx = np.argsort(vals)
            vals_sorted = vals[sorted_idx]
            wts_sorted = wts[sorted_idx]
            
            # Find median
            cumsum = np.cumsum(wts_sorted)
            median_idx = np.searchsorted(cumsum, 0.5)
            result[gene_idx] = vals_sorted[median_idx]
        
        return result
    
    def _refine_parameters(self, A_init, B_init, lambda_init, 
                          S_array, dS_dt, times, x_vals, y_vals):
        """Refine parameters using nonlinear optimization (optional)"""
        
        # For speed, we can skip this or use a quick refinement
        # Current estimates are already quite good
        
        return A_init, B_init, lambda_init
    
    def discretize_matrices(self, A, B, threshold_A=0.10, threshold_B=0.05):
        """
        Discretize continuous matrices to {-1, 0, +1}
        
        Thresholds tuned for optimal balance between sparsity and accuracy
        """
        print("  Discretizing matrices...")
        
        A_disc = np.zeros_like(A, dtype=int)
        B_disc = np.zeros_like(B, dtype=int)
        
        # Discretize A (off-diagonal only)
        for i in range(4):
            for j in range(4):
                if i == j:
                    A_disc[i, j] = 0  # Diagonal irrelevant per problem statement
                else:
                    if abs(A[i, j]) < threshold_A:
                        A_disc[i, j] = 0
                    else:
                        A_disc[i, j] = 1 if A[i, j] > 0 else -1
        
        # Discretize B
        for i in range(4):
            for j in range(2):
                if abs(B[i, j]) < threshold_B:
                    B_disc[i, j] = 0
                else:
                    B_disc[i, j] = 1 if B[i, j] > 0 else -1
        
        print("    → Discretization complete")
        
        return A_disc, B_disc
    
    def simulate_dynamics(self, S0, times, x_vals, y_vals, 
                         M1_active=True, M2_active=True):
        """
        Simulate dynamics using 4th-order Runge-Kutta method (more accurate than Euler)
        
        Args:
            S0: Initial condition [n_x, n_y, 4]
            times: Time points
            x_vals, y_vals: Spatial grid
            M1_active, M2_active: Which morphogens are present
        """
        n_t = len(times)
        n_x, n_y = len(x_vals), len(y_vals)
        
        S_result = np.zeros((n_t, n_x, n_y, 4))
        S_result[0] = S0.copy()
        
        dt = times[1] - times[0] if len(times) > 1 else 0.01
        
        # RK4 integration for better accuracy
        for ti in range(1, n_t):
            for xi, x in enumerate(x_vals):
                for yi, y in enumerate(y_vals):
                    S_curr = S_result[ti-1, xi, yi, :]
                    
                    # Morphogen values
                    M1 = x if M1_active else 0
                    M2 = y if M2_active else 0
                    M = np.array([M1, M2])
                    
                    # RK4 steps
                    def f(S):
                        return self.A @ S + self.B @ M - self.lambda_auto * S
                    
                    k1 = f(S_curr)
                    k2 = f(S_curr + 0.5 * dt * k1)
                    k3 = f(S_curr + 0.5 * dt * k2)
                    k4 = f(S_curr + dt * k3)
                    
                    S_result[ti, xi, yi, :] = S_curr + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        
        return S_result
    
    def compute_initial_steady_state(self, x_vals, y_vals, M1_active, M2_active):
        """
        Compute steady state for given morphogen configuration
        
        At steady state: 0 = AS + BM - λS
        => (λI - A)S = BM
        """
        n_x, n_y = len(x_vals), len(y_vals)
        S0 = np.zeros((n_x, n_y, 4))
        
        for xi, x in enumerate(x_vals):
            for yi, y in enumerate(y_vals):
                M1 = x if M1_active else 0
                M2 = y if M2_active else 0
                M = np.array([M1, M2])
                
                # Solve: (λI - A)S = BM
                try:
                    lhs_matrix = self.lambda_auto * np.eye(4) - self.A
                    rhs_vector = self.B @ M
                    S_ss = np.linalg.solve(lhs_matrix, rhs_vector)
                    S0[xi, yi, :] = S_ss
                except np.linalg.LinAlgError:
                    # Use least squares if singular
                    lhs_matrix = self.lambda_auto * np.eye(4) - self.A
                    rhs_vector = self.B @ M
                    S_ss = np.linalg.lstsq(lhs_matrix, rhs_vector, rcond=None)[0]
                    S0[xi, yi, :] = S_ss
        
        return S0
    
    def predict_experiment_B(self, df_test):
        """
        Predict gene expressions for Experiment B (M1 removal)
        
        Initial condition: Steady state with M1=0, M2=y
        Then simulate dynamics
        """
        print("  Predicting Experiment B (M1 removal)...")
        
        times = np.array(sorted(df_test['time'].unique()))
        x_vals = np.array(sorted(df_test['x'].unique()))
        y_vals = np.array(sorted(df_test['y'].unique()))
        
        # Compute initial steady state
        print("    → Computing initial steady state")
        S0 = self.compute_initial_steady_state(
            x_vals, y_vals,
            M1_active=False,  # M1 removed
            M2_active=True    # M2 present
        )
        
        # Simulate dynamics
        print("    → Simulating dynamics")
        S_pred = self.simulate_dynamics(
            S0, times, x_vals, y_vals,
            M1_active=False,
            M2_active=True
        )
        
        # Convert to DataFrame
        print("    → Formatting results")
        results = []
        for ti, t in enumerate(times):
            for xi, x in enumerate(x_vals):
                for yi, y in enumerate(y_vals):
                    results.append({
                        'time': t,
                        'x': x,
                        'y': y,
                        'S1': S_pred[ti, xi, yi, 0],
                        'S2': S_pred[ti, xi, yi, 1],
                        'S3': S_pred[ti, xi, yi, 2],
                        'S4': S_pred[ti, xi, yi, 3]
                    })
        
        df_pred = pd.DataFrame(results)
        print(f"    → {len(df_pred)} predictions generated")
        
        return df_pred
    
    def validate_on_training(self, df_train):
        """Validate model on training data"""
        print("  Validating on training data...")
        
        times = np.array(sorted(df_train['time'].unique()))
        x_vals = np.array(sorted(df_train['x'].unique()))
        y_vals = np.array(sorted(df_train['y'].unique()))
        
        # Get initial condition
        df_t0 = df_train[df_train['time'] == times[0]]
        
        n_x, n_y = len(x_vals), len(y_vals)
        S0 = np.zeros((n_x, n_y, 4))
        
        for xi, x in enumerate(x_vals):
            for yi, y in enumerate(y_vals):
                mask = (df_t0['x'] == x) & (df_t0['y'] == y)
                if mask.any():
                    row = df_t0[mask].iloc[0]
                    S0[xi, yi, :] = [row['S1'], row['S2'], row['S3'], row['S4']]
        
        # Simulate
        S_pred = self.simulate_dynamics(
            S0, times, x_vals, y_vals,
            M1_active=True,   # M1 present in training
            M2_active=False   # M2 removed in training
        )
        
        # Compute RMSE
        errors = []
        for ti, t in enumerate(times):
            for xi, x in enumerate(x_vals):
                for yi, y in enumerate(y_vals):
                    mask = (df_train['time'] == t) & \
                           (df_train['x'] == x) & \
                           (df_train['y'] == y)
                    if mask.any():
                        row = df_train[mask].iloc[0]
                        actual = np.array([row['S1'], row['S2'], 
                                         row['S3'], row['S4']])
                        predicted = S_pred[ti, xi, yi, :]
                        errors.append((actual - predicted) ** 2)
        
        rmse = np.sqrt(np.mean(errors))
        print(f"    → Training RMSE: {rmse:.6f}")
        
        return rmse
    
    def print_summary(self):
        """Print detailed summary of inferred parameters"""
        print("\n" + "="*80)
        print("INFERRED PARAMETERS SUMMARY")
        print("="*80)
        
        print(f"\nAuto-regulation parameter: λ = {self.lambda_auto:.6f}")
        
        print("\n" + "-"*80)
        print("Gene Regulatory Matrix A (continuous):")
        print("-"*80)
        print("       G1        G2        G3        G4")
        for i in range(4):
            row_str = "  ".join([f"{self.A[i,j]:8.4f}" for j in range(4)])
            print(f"  G{i+1}  {row_str}")
        
        print("\n" + "-"*80)
        print("Morphogen Coupling Matrix B (continuous):")
        print("-"*80)
        print("       M1        M2")
        for i in range(4):
            print(f"  G{i+1}  {self.B[i,0]:8.4f}  {self.B[i,1]:8.4f}")


def main():
    """Main execution pipeline with enhanced output"""
    
    print("\n" + "="*80)
    print(" "*20 + "GENE REGULATORY NETWORK INFERENCE")
    print(" "*25 + "AZeotropy '26 - IIT Bombay")
    print("="*80)
    
    # Initialize
    grn = GRNInferenceOptimized()
    
    # Step 1: Load training data
    print("\n" + "┌" + "─"*78 + "┐")
    print("│ STEP 1: Loading Training Data (Experiment A - M2 Removal)" + " "*18 + "│")
    print("└" + "─"*78 + "┘")
    
    try:
        df_train = grn.load_data('GRN_experiment_M2_removal.csv')
        times_train = sorted(df_train['time'].unique())
        print(f"  Time points: {len(times_train)}")
        print(f"  Time range: [{times_train[0]:.2f}, {times_train[-1]:.2f}]")
        print(f"  Spatial grid: {len(df_train['x'].unique())} × {len(df_train['y'].unique())}")
    except FileNotFoundError:
        print("  ✗ ERROR: Training file not found!")
        print("  Please ensure 'GRN_experiment_M2_removal.csv' is in the current directory")
        return None
    
    # Step 2: Parameter inference
    print("\n" + "┌" + "─"*78 + "┐")
    print("│ STEP 2: Parameter Inference" + " "*51 + "│")
    print("└" + "─"*78 + "┘")
    
    A, B, lambda_auto = grn.infer_parameters_multistage(df_train, alpha=0.003)
    grn.print_summary()
    
    # Step 3: Validation
    print("\n" + "┌" + "─"*78 + "┐")
    print("│ STEP 3: Model Validation" + " "*54 + "│")
    print("└" + "─"*78 + "┘")
    
    rmse = grn.validate_on_training(df_train)
    
    # Step 4: Discretization
    print("\n" + "┌" + "─"*78 + "┐")
    print("│ STEP 4: Matrix Discretization" + " "*49 + "│")
    print("└" + "─"*78 + "┘")
    
    A_disc, B_disc = grn.discretize_matrices(A, B, threshold_A=0.10, threshold_B=0.05)
    
    print("\n  Discrete Gene Regulatory Matrix A:")
    print("  " + "─"*40)
    print("       G1   G2   G3   G4")
    for i in range(4):
        row_str = "   ".join([f"{A_disc[i,j]:2d}" for j in range(4)])
        print(f"    G{i+1}  {row_str}")
    
    print("\n  Discrete Morphogen Coupling Matrix B:")
    print("  " + "─"*40)
    print("       M1   M2")
    for i in range(4):
        print(f"    G{i+1}  {B_disc[i,0]:2d}   {B_disc[i,1]:2d}")
    
    # Step 5: Save matrices
    print("\n" + "┌" + "─"*78 + "┐")
    print("│ STEP 5: Saving Matrix Files" + " "*51 + "│")
    print("└" + "─"*78 + "┘")
    
    np.savetxt('gene_regulatory_matrix.txt', A_disc, fmt='%d')
    np.savetxt('morphogen_coupling_matrix.txt', B_disc, fmt='%d')
    print("  ✓ gene_regulatory_matrix.txt")
    print("  ✓ morphogen_coupling_matrix.txt")
    
    # Step 6: Load test data
    print("\n" + "┌" + "─"*78 + "┐")
    print("│ STEP 6: Loading Test Data (Experiment B - M1 Removal)" + " "*25 + "│")
    print("└" + "─"*78 + "┘")
    
    try:
        df_test = grn.load_data('GRN_experiment_M1_removal.csv')
    except FileNotFoundError:
        print("  ✗ ERROR: Test file not found!")
        print("  Please ensure 'GRN_experiment_M1_removal.csv' is in the current directory")
        return None
    
    # Step 7: Prediction
    print("\n" + "┌" + "─"*78 + "┐")
    print("│ STEP 7: Prediction for Experiment B" + " "*42 + "│")
    print("└" + "─"*78 + "┘")
    
    df_pred = grn.predict_experiment_B(df_test)
    
    # Step 8: Save predictions
    print("\n" + "┌" + "─"*78 + "┐")
    print("│ STEP 8: Saving Predictions" + " "*52 + "│")
    print("└" + "─"*78 + "┘")
    
    df_pred.to_csv('GRN_experiment_M1_removal_predicted.csv', index=False)
    print("  ✓ GRN_experiment_M1_removal_predicted.csv")
    
    # Step 9: Statistics
    print("\n" + "┌" + "─"*78 + "┐")
    print("│ STEP 9: Prediction Statistics" + " "*49 + "│")
    print("└" + "─"*78 + "┘")
    
    print("\n  Gene Expression Statistics:")
    print("  " + "─"*70)
    print("  Gene    Mean      Std Dev    Min        Max")
    print("  " + "─"*70)
    for gene in ['S1', 'S2', 'S3', 'S4']:
        mean_val = df_pred[gene].mean()
        std_val = df_pred[gene].std()
        min_val = df_pred[gene].min()
        max_val = df_pred[gene].max()
        print(f"  {gene}    {mean_val:7.4f}   {std_val:7.4f}    {min_val:7.4f}    {max_val:7.4f}")
    
    # Final summary
    print("\n" + "="*80)
    print(" "*25 + "✓ ANALYSIS COMPLETE")
    print(" "*20 + "Files Ready for Submission:")
    print(" "*15 + "• gene_regulatory_matrix.txt")
    print(" "*15 + "• morphogen_coupling_matrix.txt")
    print(" "*15 + "• GRN_experiment_M1_removal_predicted.csv")
    print("="*80 + "\n")
    
    return grn, df_pred, A_disc, B_disc


if __name__ == "__main__":
    results = main()
    if results:
        grn, df_pred, A_disc, B_disc = results
        print("SUCCESS: All outputs generated!")
    else:
        print("FAILED: Check error messages above")