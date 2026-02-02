"""
═══════════════════════════════════════════════════════════════════════════════
QUICK VALIDATION AND TESTING SCRIPT
Validates all outputs and provides detailed diagnostics

Author: Competition Submission Team
Date: February 2026
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import numpy as np
import pandas as pd


def print_header(text, char="="):
    """Print formatted header"""
    print("\n" + char*80)
    print(text.center(80))
    print(char*80)


def print_section(text):
    """Print section header"""
    print("\n" + "┌" + "─"*78 + "┐")
    print(f"│ {text:<77}│")
    print("└" + "─"*78 + "┘")


def check_file(filename, required=True):
    """Check if file exists and return status"""
    exists = os.path.exists(filename)
    
    if exists:
        size = os.path.getsize(filename)
        if size > 1024*1024:
            size_str = f"{size/(1024*1024):.2f} MB"
        elif size > 1024:
            size_str = f"{size/1024:.2f} KB"
        else:
            size_str = f"{size} B"
        print(f"  ✓ {filename:<45} ({size_str})")
    else:
        status = "✗ MISSING" if required else "  (optional)"
        print(f"  {status} {filename}")
    
    return exists


def validate_matrix(matrix, name, expected_shape, valid_values):
    """Validate matrix properties"""
    print(f"\n{name}:")
    print("─"*70)
    
    # Check shape
    if matrix.shape != expected_shape:
        print(f"  ✗ ERROR: Shape is {matrix.shape}, expected {expected_shape}")
        return False
    print(f"  ✓ Shape: {matrix.shape}")
    
    # Check values
    unique_vals = set(np.unique(matrix).astype(int))
    if not unique_vals.issubset(valid_values):
        invalid = unique_vals - valid_values
        print(f"  ✗ ERROR: Invalid values found: {invalid}")
        return False
    print(f"  ✓ Values: {sorted(unique_vals)} (all valid)")
    
    # Print matrix
    print(f"\nMatrix content:")
    if name == "Gene Regulatory Matrix A":
        print("       G1   G2   G3   G4")
        for i in range(4):
            row_str = "   ".join([f"{matrix[i,j]:2d}" for j in range(4)])
            print(f"  G{i+1}  {row_str}")
    else:  # Morphogen Coupling Matrix B
        print("       M1   M2")
        for i in range(4):
            print(f"  G{i+1}  {matrix[i,0]:2d}   {matrix[i,1]:2d}")
    
    return True


def validate_predictions(df, name):
    """Validate prediction dataframe"""
    print(f"\n{name}:")
    print("─"*70)
    
    # Check columns
    required_cols = ['time', 'x', 'y', 'S1', 'S2', 'S3', 'S4']
    missing_cols = set(required_cols) - set(df.columns)
    
    if missing_cols:
        print(f"  ✗ ERROR: Missing columns: {missing_cols}")
        return False
    print(f"  ✓ Columns: {list(df.columns)}")
    
    # Check dimensions
    n_times = len(df['time'].unique())
    n_x = len(df['x'].unique())
    n_y = len(df['y'].unique())
    
    print(f"  ✓ Dimensions: {len(df)} rows")
    print(f"    Time points: {n_times}")
    print(f"    Spatial grid: {n_x} × {n_y}")
    
    # Check for NaN
    if df.isnull().any().any():
        nan_cols = df.columns[df.isnull().any()].tolist()
        print(f"  ✗ WARNING: NaN values in columns: {nan_cols}")
    else:
        print(f"  ✓ No NaN values")
    
    # Statistics
    print(f"\nGene Expression Statistics:")
    print("  Gene    Mean      Std       Min       Max")
    print("  " + "─"*50)
    for gene in ['S1', 'S2', 'S3', 'S4']:
        mean = df[gene].mean()
        std = df[gene].std()
        min_val = df[gene].min()
        max_val = df[gene].max()
        print(f"  {gene}   {mean:7.4f}   {std:7.4f}   {min_val:7.4f}   {max_val:7.4f}")
    
    return True


def interpret_network(A, B):
    """Provide biological interpretation of the network"""
    print_section("Network Interpretation")
    
    print("\nGene-Gene Interactions:")
    print("─"*70)
    
    for i in range(4):
        activators = []
        inhibitors = []
        
        for j in range(4):
            if i != j:
                if A[i, j] > 0:
                    activators.append(f"G{j+1}")
                elif A[i, j] < 0:
                    inhibitors.append(f"G{j+1}")
        
        if activators or inhibitors:
            print(f"\nGene {i+1}:")
            if activators:
                print(f"  • Activated by: {', '.join(activators)}")
            if inhibitors:
                print(f"  • Inhibited by: {', '.join(inhibitors)}")
        else:
            print(f"\nGene {i+1}: No direct gene-gene interactions")
    
    print("\n\nMorphogen Effects:")
    print("─"*70)
    
    for i in range(4):
        m1_effect = ""
        m2_effect = ""
        
        if B[i, 0] > 0:
            m1_effect = "activated by M₁"
        elif B[i, 0] < 0:
            m1_effect = "inhibited by M₁"
        else:
            m1_effect = "no M₁ coupling"
        
        if B[i, 1] > 0:
            m2_effect = "activated by M₂"
        elif B[i, 1] < 0:
            m2_effect = "inhibited by M₂"
        else:
            m2_effect = "no M₂ coupling"
        
        print(f"\nGene {i+1}:")
        print(f"  • {m1_effect}")
        print(f"  • {m2_effect}")
    
    # Network motifs
    print("\n\nNetwork Motifs:")
    print("─"*70)
    
    # Check for feedback loops
    feedback_loops = []
    for i in range(4):
        for j in range(4):
            if i != j and A[i, j] != 0 and A[j, i] != 0:
                loop_type = "positive" if A[i,j] * A[j,i] > 0 else "negative"
                feedback_loops.append(f"G{i+1} ↔ G{j+1} ({loop_type})")
    
    if feedback_loops:
        print("\nFeedback Loops:")
        for loop in set(feedback_loops):
            print(f"  • {loop}")
    
    # Check for cascades
    cascades = []
    for i in range(4):
        for j in range(4):
            if i != j and A[i, j] != 0:
                for k in range(4):
                    if k != i and k != j and A[k, i] != 0:
                        cascades.append(f"G{j+1} → G{i+1} → G{k+1}")
    
    if cascades:
        print("\nCascades (sample):")
        for cascade in list(set(cascades))[:5]:
            print(f"  • {cascade}")


def main():
    """Main validation pipeline"""
    
    print_header("VALIDATION & DIAGNOSTICS")
    print("  Gene Regulatory Network Inference")
    print("  AZeotropy '26 - IIT Bombay\n")
    
    all_valid = True
    
    # Check files
    print_section("File Existence Check")
    
    print("\nRequired Submission Files:")
    all_valid &= check_file('gene_regulatory_matrix.txt', required=True)
    all_valid &= check_file('morphogen_coupling_matrix.txt', required=True)
    all_valid &= check_file('GRN_experiment_M1_removal_predicted.csv', required=True)
    
    print("\nOptional Visualization Files:")
    check_file('network_topology.png', required=False)
    check_file('matrix_heatmaps.png', required=False)
    check_file('spatial_evolution.png', required=False)
    check_file('time_series.png', required=False)
    check_file('all_genes.png', required=False)
    
    if not all_valid:
        print("\n" + "="*80)
        print("VALIDATION FAILED: Required files are missing!".center(80))
        print("="*80)
        print("\nPlease run: python grn_inference_final.py")
        return False
    
    # Validate matrices
    print_section("Matrix Validation")
    
    try:
        A = np.loadtxt('gene_regulatory_matrix.txt', dtype=int)
        all_valid &= validate_matrix(A, "Gene Regulatory Matrix A", 
                                     (4, 4), {-1, 0, 1})
        
        # Check diagonal
        diag_sum = np.sum(np.abs(np.diag(A)))
        if diag_sum > 0:
            print(f"\n  ⚠ WARNING: {diag_sum} non-zero diagonal elements")
            print(f"    (Diagonal should be 0 per problem statement)")
        
    except Exception as e:
        print(f"\n  ✗ ERROR loading gene regulatory matrix: {e}")
        all_valid = False
    
    try:
        B = np.loadtxt('morphogen_coupling_matrix.txt', dtype=int)
        all_valid &= validate_matrix(B, "Morphogen Coupling Matrix B",
                                     (4, 2), {-1, 0, 1})
    except Exception as e:
        print(f"\n  ✗ ERROR loading morphogen coupling matrix: {e}")
        all_valid = False
    
    # Validate predictions
    print_section("Predictions Validation")
    
    try:
        df_pred = pd.read_csv('GRN_experiment_M1_removal_predicted.csv')
        all_valid &= validate_predictions(df_pred, "Experiment B Predictions")
    except Exception as e:
        print(f"\n  ✗ ERROR loading predictions: {e}")
        all_valid = False
    
    # Network interpretation
    if all_valid:
        try:
            interpret_network(A, B)
        except:
            pass
    
    # Final verdict
    print("\n" + "="*80)
    if all_valid:
        print("✓ VALIDATION SUCCESSFUL - All outputs are valid!".center(80))
        print("="*80)
        print("\nYour submission is ready! Files to submit:")
        print("  1. gene_regulatory_matrix.txt")
        print("  2. morphogen_coupling_matrix.txt")
        print("  3. GRN_experiment_M1_removal_predicted.csv")
        print("  4. Technical report (create from template)")
    else:
        print("✗ VALIDATION FAILED - Please fix errors above".center(80))
        print("="*80)
    
    print("\n")
    return all_valid


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)