"""
Gene Regulatory Network Inference - AZeotropy '26
Main inference script for GRN parameter estimation and prediction

Author: Competition Submission
Date: February 2026
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')


class GRNInference:
    """
    Gene Regulatory Network Inference System
    
    Infers parameters of the dynamical system: dS/dt = AS + BM - λS
    where:
        S: 4D gene expression deviation vector
        A: 4x4 gene-gene interaction matrix
        B: 4x2 morphogen-gene coupling matrix
        M: 2D morphogen concentration vector
        λ: auto-regulation parameter
    """
    
    def __init__(self):
        self.A = None  # Gene regulatory matrix
        self.B = None  # Morphogen coupling matrix
        self.lambda_auto = None  # Auto-regulation parameter
        
    def load_data(self, filepath):
        """Load experimental data from CSV file"""
        print(f"Loading data from: {filepath}")
        df = pd.read_csv(filepath)
        print(f"  -> Loaded {len(df)} data points")
        return df
    
    def morphogen_profiles(self, x, y):
        """
        Define morphogen spatial profiles
        Based on analysis: M1(x,y) = x, M2(x,y) = y
        """
        return x, y
    
    def compute_time_derivatives(self, df):
        """
        Compute dS/dt using finite differences
        
        Uses:
        - Forward difference for t=0
        - Backward difference for t=final
        - Central difference for intermediate points
        
        Returns:
            S_array: 4D array [time, x, y, gene]
            dS_dt: Time derivatives
            times: Time points
            x_vals, y_vals: Spatial grid points
        """
        times = sorted(df['time'].unique())
        x_vals = sorted(df['x'].unique())
        y_vals = sorted(df['y'].unique())
        
        n_t, n_x, n_y = len(times), len(x_vals), len(y_vals)
        n_genes = 4
        
        # Build 4D array [time, x, y, gene]
        S_array = np.zeros((n_t, n_x, n_y, n_genes))
        
        for ti, t in enumerate(times):
            for xi, x in enumerate(x_vals):
                for yi, y in enumerate(y_vals):
                    mask = (df['time'] == t) & (df['x'] == x) & (df['y'] == y)
                    if mask.any():
                        row = df[mask].iloc[0]
                        S_array[ti, xi, yi, :] = [row['S1'], row['S2'], 
                                                   row['S3'], row['S4']]
        
        # Compute time derivatives
        dS_dt = np.zeros_like(S_array)
        
        for ti in range(n_t):
            if ti == 0:
                # Forward difference for first point
                dt = times[1] - times[0]
                dS_dt[ti] = (S_array[ti + 1] - S_array[ti]) / dt
            elif ti == n_t - 1:
                # Backward difference for last point
                dt = times[ti] - times[ti - 1]
                dS_dt[ti] = (S_array[ti] - S_array[ti - 1]) / dt
            else:
                # Central difference for middle points
                dt = times[ti + 1] - times[ti - 1]
                dS_dt[ti] = (S_array[ti + 1] - S_array[ti - 1]) / dt
        
        return S_array, dS_dt, times, x_vals, y_vals
    
    def infer_parameters(self, df, alpha=0.005):
        """
        Infer GRN parameters using Ridge regression with steady-state constraints
        
        Strategy:
        1. Use dynamics to estimate (A - λI) and B[:,0] via regression
        2. Extract λ from diagonal elements
        3. Use steady-state constraint to estimate B[:,1]
        
        Args:
            df: Training data (M2 removal experiment)
            alpha: Ridge regularization parameter
            
        Returns:
            A, B, lambda_auto: Inferred parameters
        """
        print("\nInferring GRN parameters...")
        
        # Compute derivatives
        S_array, dS_dt, times, x_vals, y_vals = self.compute_time_derivatives(df)
        
        n_samples = len(times) * len(x_vals) * len(y_vals)
        n_genes = 4
        
        # Flatten arrays for regression
        S_flat = S_array.reshape(n_samples, n_genes)
        dS_dt_flat = dS_dt.reshape(n_samples, n_genes)
        
        # Create M1 values (M2 is removed in this experiment)
        M1_flat = np.zeros(n_samples)
        idx = 0
        for t in times:
            for x in x_vals:
                for y in y_vals:
                    M1_flat[idx] = x  # M1(x,y) = x
                    idx += 1
        
        # Stage 1: Estimate (A - λI) and B[:,0] from dynamics
        # Model: dS/dt = (A - λI)S + B[:,0]*M1
        
        A_minus_lambda_I = np.zeros((n_genes, n_genes))
        B_col0 = np.zeros(n_genes)
        
        print("  Stage 1: Ridge regression for (A-λI) and B[:,0]")
        
        for gene_idx in range(n_genes):
            y_target = dS_dt_flat[:, gene_idx]
            
            # Features: [S1, S2, S3, S4, M1]
            X = np.column_stack([S_flat, M1_flat])
            
            # Ridge regression
            model = Ridge(alpha=alpha)
            model.fit(X, y_target)
            
            # Extract coefficients
            A_minus_lambda_I[gene_idx, :] = model.coef_[:4]
            B_col0[gene_idx] = model.coef_[4]
        
        # Stage 2: Extract λ from diagonal
        # Auto-regulation should stabilize, so diagonal of (A-λI) is negative
        lambda_est = -np.mean(np.diag(A_minus_lambda_I))
        lambda_est = max(lambda_est, 0.1)  # Ensure positive
        
        print(f"  Stage 2: Estimated λ = {lambda_est:.4f}")
        
        # Stage 3: Recover A matrix
        A = A_minus_lambda_I + lambda_est * np.eye(n_genes)
        
        # Stage 4: Infer B[:,1] using steady-state constraint
        # At steady state (before M2 removal): 0 = AS + B*[M1, M2] - λS
        # => BM = (λI - A)S
        
        print("  Stage 3: Inferring B[:,1] from steady-state constraints")
        
        df_t0 = df[df['time'] == 0].copy()
        B_col1_estimates = []
        
        for _, row in df_t0.iterrows():
            x, y = row['x'], row['y']
            if abs(y) > 0.01:  # Avoid division by small numbers
                S_vec = np.array([row['S1'], row['S2'], row['S3'], row['S4']])
                
                # Right-hand side: (λI - A)S
                rhs = lambda_est * S_vec - A @ S_vec
                
                # Solve: B[:,0]*x + B[:,1]*y = rhs
                # => B[:,1] = (rhs - B[:,0]*x) / y
                B_col1 = (rhs - B_col0 * x) / y
                B_col1_estimates.append(B_col1)
        
        # Use median for robustness
        if B_col1_estimates:
            B_col1 = np.median(B_col1_estimates, axis=0)
        else:
            B_col1 = np.zeros(n_genes)
        
        # Construct B matrix
        B = np.column_stack([B_col0, B_col1])
        
        # Store inferred parameters
        self.A = A
        self.B = B
        self.lambda_auto = lambda_est
        
        print("  -> Parameter inference complete")
        
        return A, B, lambda_est
    
    def discretize_matrices(self, A, B, threshold_A=0.12, threshold_B=0.05):
        """
        Discretize continuous matrices to {-1, 0, +1}
        
        Args:
            A: Continuous gene regulatory matrix
            B: Continuous morphogen coupling matrix
            threshold_A: Threshold for A matrix discretization
            threshold_B: Threshold for B matrix discretization
            
        Returns:
            A_discrete, B_discrete: Discretized matrices
        """
        print("\nDiscretizing matrices...")
        
        A_disc = np.zeros_like(A, dtype=int)
        B_disc = np.zeros_like(B, dtype=int)
        
        # Discretize A (off-diagonal only, diagonal is irrelevant)
        for i in range(4):
            for j in range(4):
                if i == j:
                    A_disc[i, j] = 0  # Diagonal irrelevant
                elif abs(A[i, j]) < threshold_A:
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
        
        print("  -> Discretization complete")
        
        return A_disc, B_disc
    
    def simulate_dynamics(self, S0, times, x_vals, y_vals, 
                         M1_present=True, M2_present=True):
        """
        Simulate gene expression dynamics using Forward Euler method
        
        Solves: dS/dt = AS + BM - λS
        
        Args:
            S0: Initial condition [n_x, n_y, 4]
            times: Time points to simulate
            x_vals, y_vals: Spatial grid
            M1_present, M2_present: Which morphogens are active
            
        Returns:
            S_result: Simulated expressions [n_t, n_x, n_y, 4]
        """
        n_t = len(times)
        n_x, n_y = len(x_vals), len(y_vals)
        
        S_result = np.zeros((n_t, n_x, n_y, 4))
        S_result[0] = S0
        
        # Time step
        dt = times[1] - times[0] if len(times) > 1 else 0.01
        
        # Forward Euler integration
        for ti in range(1, n_t):
            for xi, x in enumerate(x_vals):
                for yi, y in enumerate(y_vals):
                    S_curr = S_result[ti - 1, xi, yi, :]
                    
                    # Morphogen values
                    M1 = x if M1_present else 0
                    M2 = y if M2_present else 0
                    M = np.array([M1, M2])
                    
                    # Compute derivative: dS/dt = AS + BM - λS
                    dS = self.A @ S_curr + self.B @ M - self.lambda_auto * S_curr
                    
                    # Update
                    S_result[ti, xi, yi, :] = S_curr + dt * dS
        
        return S_result
    
    def predict_M1_removal(self, df_test):
        """
        Predict gene expressions for M1 removal experiment (Experiment B)
        
        Steps:
        1. Compute initial steady state with M1=0, M2=y
        2. Simulate dynamics forward in time
        3. Format results as DataFrame
        
        Args:
            df_test: Test data structure (contains time, x, y coordinates)
            
        Returns:
            df_predicted: DataFrame with predictions
        """
        print("\nPredicting M1 removal experiment...")
        
        times = sorted(df_test['time'].unique())
        x_vals = sorted(df_test['x'].unique())
        y_vals = sorted(df_test['y'].unique())
        
        n_x, n_y = len(x_vals), len(y_vals)
        
        # Compute initial steady state: (λI - A)S = B*[0, M2]
        print("  Computing initial steady state...")
        S0 = np.zeros((n_x, n_y, 4))
        
        for xi, x in enumerate(x_vals):
            for yi, y in enumerate(y_vals):
                M = np.array([0, y])  # M1 removed, M2 = y
                
                # Solve linear system: (λI - A)S = BM
                try:
                    S_ss = np.linalg.solve(
                        self.lambda_auto * np.eye(4) - self.A,
                        self.B @ M
                    )
                    S0[xi, yi, :] = S_ss
                except np.linalg.LinAlgError:
                    # Fallback for singular matrices
                    S0[xi, yi, :] = np.zeros(4)
        
        # Simulate dynamics
        print("  Simulating dynamics...")
        S_pred = self.simulate_dynamics(
            S0, np.array(times), x_vals, y_vals,
            M1_present=False,  # M1 removed
            M2_present=True    # M2 active
        )
        
        # Convert to DataFrame
        print("  Formatting results...")
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
        
        df_predicted = pd.DataFrame(results)
        print(f"  -> Generated {len(df_predicted)} predictions")
        
        return df_predicted
    
    def validate_on_training(self, df_train):
        """
        Validate model on training data by simulating and comparing
        
        Returns:
            rmse: Root mean squared error
        """
        print("\nValidating on training data...")
        
        times = sorted(df_train['time'].unique())
        x_vals = sorted(df_train['x'].unique())
        y_vals = sorted(df_train['y'].unique())
        
        # Get initial condition from data
        df_t0 = df_train[df_train['time'] == times[0]]
        
        n_x, n_y = len(x_vals), len(y_vals)
        S0 = np.zeros((n_x, n_y, 4))
        
        for xi, x in enumerate(x_vals):
            for yi, y in enumerate(y_vals):
                mask = (df_t0['x'] == x) & (df_t0['y'] == y)
                if mask.any():
                    row = df_t0[mask].iloc[0]
                    S0[xi, yi, :] = [row['S1'], row['S2'], row['S3'], row['S4']]
        
        # Simulate with M1=x, M2=0 (training conditions)
        S_pred = self.simulate_dynamics(
            S0, np.array(times), x_vals, y_vals,
            M1_present=True,   # M1 active in training
            M2_present=False   # M2 removed in training
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
        print(f"  Training RMSE: {rmse:.6f}")
        
        return rmse
    
    def print_results(self):
        """Print inferred parameters in readable format"""
        print("\n" + "="*80)
        print("INFERRED PARAMETERS")
        print("="*80)
        
        print(f"\nAuto-regulation parameter: λ = {self.lambda_auto:.4f}")
        
        print("\nGene Regulatory Matrix A (continuous):")
        print("     G1      G2      G3      G4")
        for i in range(4):
            row_str = "  ".join([f"{self.A[i,j]:7.3f}" for j in range(4)])
            print(f"G{i+1}  {row_str}")
        
        print("\nMorphogen Coupling Matrix B (continuous):")
        print("     M1      M2")
        for i in range(4):
            print(f"G{i+1}  {self.B[i,0]:7.3f}  {self.B[i,1]:7.3f}")


def main():
    """Main execution pipeline"""
    
    print("="*80)
    print("GENE REGULATORY NETWORK INFERENCE")
    print("AZeotropy '26 Competition - IIT Bombay")
    print("="*80)
    
    # Initialize inference system
    grn = GRNInference()
    
    # Step 1: Load training data (M2 removal experiment)
    print("\n[STEP 1] Loading Training Data")
    print("-" * 80)
    df_train = grn.load_data('GRN_experiment_M2_removal.csv')
    print(f"Time range: [{df_train['time'].min():.2f}, {df_train['time'].max():.2f}]")
    print(f"Spatial grid: {len(df_train['x'].unique())} × {len(df_train['y'].unique())}")
    
    # Step 2: Infer parameters
    print("\n[STEP 2] Parameter Inference")
    print("-" * 80)
    A, B, lambda_auto = grn.infer_parameters(df_train, alpha=0.005)
    grn.print_results()
    
    # Step 3: Validate
    print("\n[STEP 3] Model Validation")
    print("-" * 80)
    rmse = grn.validate_on_training(df_train)
    
    # Step 4: Discretize
    print("\n[STEP 4] Matrix Discretization")
    print("-" * 80)
    A_disc, B_disc = grn.discretize_matrices(A, B, threshold_A=0.12, threshold_B=0.05)
    
    print("\nDiscrete Gene Regulatory Matrix A:")
    print("     G1  G2  G3  G4")
    for i in range(4):
        row_str = "  ".join([f"{A_disc[i,j]:2d}" for j in range(4)])
        print(f"G{i+1}  {row_str}")
    
    print("\nDiscrete Morphogen Coupling Matrix B:")
    print("     M1  M2")
    for i in range(4):
        print(f"G{i+1}  {B_disc[i,0]:2d}  {B_disc[i,1]:2d}")
    
    # Step 5: Save matrices
    print("\n[STEP 5] Saving Matrices")
    print("-" * 80)
    np.savetxt('gene_regulatory_matrix.txt', A_disc, fmt='%d')
    np.savetxt('morphogen_coupling_matrix.txt', B_disc, fmt='%d')
    print("  ✓ gene_regulatory_matrix.txt")
    print("  ✓ morphogen_coupling_matrix.txt")
    
    # Step 6: Load test data and predict
    print("\n[STEP 6] Prediction for M1 Removal")
    print("-" * 80)
    df_test = grn.load_data('GRN_experiment_M1_removal.csv')
    df_pred = grn.predict_M1_removal(df_test)
    
    # Step 7: Save predictions
    print("\n[STEP 7] Saving Predictions")
    print("-" * 80)
    df_pred.to_csv('GRN_experiment_M1_removal_predicted.csv', index=False)
    print("  ✓ GRN_experiment_M1_removal_predicted.csv")
    
    # Summary statistics
    print("\n[STEP 8] Prediction Statistics")
    print("-" * 80)
    for gene in ['S1', 'S2', 'S3', 'S4']:
        mean_val = df_pred[gene].mean()
        std_val = df_pred[gene].std()
        min_val = df_pred[gene].min()
        max_val = df_pred[gene].max()
        print(f"{gene}: μ={mean_val:7.4f}, σ={std_val:7.4f}, "
              f"range=[{min_val:7.4f}, {max_val:7.4f}]")
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE - FILES READY FOR SUBMISSION")
    print("="*80)
    
    return grn, df_pred, A_disc, B_disc


if __name__ == "__main__":
    grn, df_pred, A_disc, B_disc = main()