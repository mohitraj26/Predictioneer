"""
═══════════════════════════════════════════════════════════════════════════════
RUN ALL - COMPLETE SOLUTION PIPELINE
Executes the entire GRN inference and visualization pipeline

Author: Competition Submission Team
Date: February 2026
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import os
import subprocess
import time


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80 + "\n")


def print_section(text):
    """Print formatted section"""
    print("\n" + "┌" + "─"*78 + "┐")
    print(f"│ {text:<77}│")
    print("└" + "─"*78 + "┘\n")


def check_file_exists(filename):
    """Check if file exists"""
    exists = os.path.exists(filename)
    status = "✓" if exists else "✗"
    print(f"  {status} {filename}")
    return exists


def check_dependencies():
    """Check if required Python packages are installed"""
    print_section("Checking Dependencies")
    
    required_packages = [
        'numpy',
        'pandas',
        'scipy',
        'sklearn',
        'matplotlib',
        'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (MISSING)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n  ERROR: Missing packages: {', '.join(missing_packages)}")
        print(f"  Install with: pip install {' '.join(missing_packages)}")
        return False
    
    print("\n  All dependencies satisfied!")
    return True


def check_input_files():
    """Check if input data files exist"""
    print_section("Checking Input Files")
    
    required_files = [
        'GRN_experiment_M2_removal.csv',
        'GRN_experiment_M1_removal.csv'
    ]
    
    all_present = True
    for filename in required_files:
        if not check_file_exists(filename):
            all_present = False
    
    if not all_present:
        print("\n  ERROR: Missing required input files!")
        print("  Please ensure both CSV files are in the current directory")
        return False
    
    print("\n  All input files present!")
    return True


def run_inference():
    """Run main GRN inference script"""
    print_section("Running GRN Inference")
    
    try:
        # Import and run
        print("  Executing grn_inference_final.py...")
        print("  " + "─"*70)
        
        import grn_inference_final
        results = grn_inference_final.main()
        
        if results is None:
            print("\n  ✗ Inference failed!")
            return False
        
        print("\n  ✓ Inference completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n  ✗ ERROR during inference: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_visualizations():
    """Run visualization script"""
    print_section("Creating Visualizations")
    
    try:
        print("  Executing create_visualizations_final.py...")
        print("  " + "─"*70)
        
        import create_visualizations_final
        create_visualizations_final.main()
        
        print("\n  ✓ Visualizations created successfully!")
        return True
        
    except Exception as e:
        print(f"\n  ✗ ERROR during visualization: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_outputs():
    """Validate all output files"""
    print_section("Validating Outputs")
    
    required_outputs = {
        'Matrices': [
            'gene_regulatory_matrix.txt',
            'morphogen_coupling_matrix.txt'
        ],
        'Predictions': [
            'GRN_experiment_M1_removal_predicted.csv'
        ],
        'Visualizations': [
            'network_topology.png',
            'matrix_heatmaps.png',
            'spatial_evolution.png',
            'time_series.png',
            'all_genes.png'
        ]
    }
    
    all_present = True
    
    for category, files in required_outputs.items():
        print(f"\n  {category}:")
        for filename in files:
            if not check_file_exists(filename):
                all_present = False
    
    if all_present:
        print("\n  ✓ All output files generated successfully!")
        return True
    else:
        print("\n  ✗ Some output files are missing!")
        return False


def print_summary():
    """Print final summary"""
    import numpy as np
    import pandas as pd
    
    print_header("SUBMISSION SUMMARY")
    
    # Load results
    try:
        A = np.loadtxt('gene_regulatory_matrix.txt', dtype=int)
        B = np.loadtxt('morphogen_coupling_matrix.txt', dtype=int)
        df_pred = pd.read_csv('GRN_experiment_M1_removal_predicted.csv')
        
        print("Gene Regulatory Matrix A:")
        print("─"*50)
        print("       G1   G2   G3   G4")
        for i in range(4):
            row_str = "   ".join([f"{A[i,j]:2d}" for j in range(4)])
            print(f"  G{i+1}  {row_str}")
        
        print("\nMorphogen Coupling Matrix B:")
        print("─"*50)
        print("       M1   M2")
        for i in range(4):
            print(f"  G{i+1}  {B[i,0]:2d}   {B[i,1]:2d}")
        
        print("\nPrediction Statistics:")
        print("─"*50)
        print(f"  Total predictions: {len(df_pred)}")
        print(f"  Time range: [{df_pred['time'].min():.2f}, {df_pred['time'].max():.2f}]")
        print(f"  Spatial coverage: {len(df_pred['x'].unique())} × {len(df_pred['y'].unique())}")
        
        print("\n" + "="*80)
        print("FILES READY FOR SUBMISSION:".center(80))
        print("="*80)
        print("  1. gene_regulatory_matrix.txt")
        print("  2. morphogen_coupling_matrix.txt")
        print("  3. GRN_experiment_M1_removal_predicted.csv")
        print("  4. Report (create using report template)")
        print("="*80)
        
    except Exception as e:
        print(f"Could not load results: {e}")


def main():
    """Main execution pipeline"""
    
    print_header("GRN INFERENCE - COMPLETE PIPELINE")
    print("  AZeotropy '26 Competition - IIT Bombay")
    print("  Automated Execution Script")
    
    start_time = time.time()
    
    # Step 1: Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Step 2: Check input files
    if not check_input_files():
        sys.exit(1)
    
    # Step 3: Run inference
    if not run_inference():
        print("\n" + "="*80)
        print("FAILED: Inference did not complete successfully".center(80))
        print("="*80)
        sys.exit(1)
    
    # Step 4: Create visualizations
    if not run_visualizations():
        print("\n  WARNING: Visualization failed, but core results are available")
    
    # Step 5: Validate outputs
    validate_outputs()
    
    # Step 6: Print summary
    print_summary()
    
    # Final message
    elapsed_time = time.time() - start_time
    print("\n" + "="*80)
    print(f"✓ PIPELINE COMPLETED SUCCESSFULLY in {elapsed_time:.1f} seconds".center(80))
    print("="*80 + "\n")


if __name__ == "__main__":
    main()