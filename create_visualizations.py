"""
Visualization Script for GRN Analysis
Creates comprehensive visualizations of network structure and predictions

Author: Competition Submission
Date: February 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def create_network_diagram(A_disc, save_path='network_diagram.png'):
    """
    Create a network diagram showing gene regulatory interactions
    
    Args:
        A_disc: Discrete gene regulatory matrix
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.axis('off')
    ax.set_title('Gene Regulatory Network Topology', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Gene positions (circular layout)
    gene_pos = {
        0: (0, 1.2),     # G1 top
        1: (1.2, 0),     # G2 right
        2: (0, -1.2),    # G3 bottom
        3: (-1.2, 0)     # G4 left
    }
    
    # Colors for genes
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    # Draw gene nodes
    for i, (x, y) in gene_pos.items():
        circle = plt.Circle((x, y), 0.3, color=colors[i], 
                           ec='black', linewidth=2.5, zorder=10)
        ax.add_patch(circle)
        ax.text(x, y, f'G{i+1}', ha='center', va='center', 
               fontweight='bold', fontsize=16, color='white', zorder=11)
    
    # Draw interactions
    for i in range(4):
        for j in range(4):
            if i != j and A_disc[i, j] != 0:
                x1, y1 = gene_pos[j]
                x2, y2 = gene_pos[i]
                
                # Calculate arrow positions (from edge to edge)
                dx, dy = x2 - x1, y2 - y1
                length = np.sqrt(dx**2 + dy**2)
                dx, dy = dx/length, dy/length
                
                start_x = x1 + 0.3 * dx
                start_y = y1 + 0.3 * dy
                end_x = x2 - 0.3 * dx
                end_y = y2 - 0.3 * dy
                
                # Arrow properties based on interaction type
                if A_disc[i, j] > 0:
                    # Activation (green, solid)
                    color = '#2ECC71'
                    linestyle = '-'
                    arrowstyle = '->'
                else:
                    # Inhibition (red, dashed)
                    color = '#E74C3C'
                    linestyle = '--'
                    arrowstyle = '-|>'
                
                # Add curved arrow for better visibility
                arrow = FancyArrowPatch(
                    (start_x, start_y), (end_x, end_y),
                    arrowstyle=arrowstyle,
                    color=color,
                    linewidth=2.5,
                    linestyle=linestyle,
                    mutation_scale=25,
                    zorder=5,
                    connectionstyle="arc3,rad=0.1"
                )
                ax.add_patch(arrow)
    
    # Add legend
    activation_patch = mpatches.Patch(color='#2ECC71', label='Activation (+1)')
    inhibition_patch = mpatches.Patch(color='#E74C3C', label='Inhibition (-1)')
    ax.legend(handles=[activation_patch, inhibition_patch], 
             loc='upper right', fontsize=12, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()


def create_comprehensive_analysis(df_train, df_pred, A_disc, B_disc, 
                                  save_path='comprehensive_analysis.png'):
    """
    Create comprehensive analysis figure with multiple panels
    
    Args:
        df_train: Training data
        df_pred: Predicted data
        A_disc, B_disc: Discrete matrices
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(20, 12))
    
    # ===== Panel 1: Gene Regulatory Matrix =====
    ax1 = plt.subplot(3, 5, 1)
    im1 = ax1.imshow(A_disc, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    ax1.set_xticks(range(4))
    ax1.set_yticks(range(4))
    ax1.set_xticklabels(['G1', 'G2', 'G3', 'G4'], fontsize=10)
    ax1.set_yticklabels(['G1', 'G2', 'G3', 'G4'], fontsize=10)
    ax1.set_xlabel('Regulator (j)', fontweight='bold', fontsize=10)
    ax1.set_ylabel('Target (i)', fontweight='bold', fontsize=10)
    ax1.set_title('Gene Regulatory Matrix A', fontweight='bold', fontsize=11)
    
    # Add values
    for i in range(4):
        for j in range(4):
            if i != j:
                color = 'white' if abs(A_disc[i,j]) == 1 else 'gray'
                ax1.text(j, i, int(A_disc[i,j]), ha="center", va="center",
                        color=color, fontweight='bold', fontsize=12)
    plt.colorbar(im1, ax=ax1, label='Interaction')
    
    # ===== Panel 2: Morphogen Coupling =====
    ax2 = plt.subplot(3, 5, 2)
    im2 = ax2.imshow(B_disc, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax2.set_xticks(range(2))
    ax2.set_yticks(range(4))
    ax2.set_xticklabels(['M1', 'M2'], fontsize=10)
    ax2.set_yticklabels(['G1', 'G2', 'G3', 'G4'], fontsize=10)
    ax2.set_xlabel('Morphogen', fontweight='bold', fontsize=10)
    ax2.set_ylabel('Gene', fontweight='bold', fontsize=10)
    ax2.set_title('Morphogen Coupling B', fontweight='bold', fontsize=11)
    
    # Add values
    for i in range(4):
        for j in range(2):
            color = 'white' if abs(B_disc[i,j]) == 1 else 'gray'
            ax2.text(j, i, int(B_disc[i,j]), ha="center", va="center",
                    color=color, fontweight='bold', fontsize=12)
    plt.colorbar(im2, ax=ax2, label='Coupling')
    
    # ===== Panel 3: Interaction Summary =====
    ax3 = plt.subplot(3, 5, 3)
    ax3.axis('off')
    ax3.set_title('Network Summary', fontweight='bold', fontsize=11)
    
    summary_text = "Gene Interactions:\n"
    summary_text += "─" * 25 + "\n"
    
    for i in range(4):
        interactions = []
        for j in range(4):
            if i != j and A_disc[i, j] != 0:
                symbol = "→" if A_disc[i, j] > 0 else "⊣"
                interactions.append(f"G{j+1}{symbol}")
        if interactions:
            summary_text += f"G{i+1}: {' '.join(interactions)}\n"
    
    summary_text += "\nMorphogen Effects:\n"
    summary_text += "─" * 25 + "\n"
    for i in range(4):
        effects = []
        for j in range(2):
            if B_disc[i, j] != 0:
                symbol = "↑" if B_disc[i, j] > 0 else "↓"
                effects.append(f"M{j+1}{symbol}")
        if effects:
            summary_text += f"G{i+1}: {' '.join(effects)}\n"
    
    ax3.text(0.1, 0.9, summary_text, transform=ax3.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # ===== Panels 4-7: Training Data Spatial Patterns =====
    times_train = [0, 0.8, 2.0, 4.0]
    
    for idx, t in enumerate(times_train):
        ax = plt.subplot(3, 5, idx + 6)
        df_t = df_train[np.isclose(df_train['time'], t, atol=0.01)]
        
        if len(df_t) > 0:
            x_vals = sorted(df_t['x'].unique())
            y_vals = sorted(df_t['y'].unique())
            
            # Create grid for Gene 1
            S1_grid = np.zeros((len(y_vals), len(x_vals)))
            for i, y in enumerate(y_vals):
                for j, x in enumerate(x_vals):
                    val = df_t[(df_t['x'] == x) & (df_t['y'] == y)]['S1'].values
                    if len(val) > 0:
                        S1_grid[len(y_vals)-1-i, j] = val[0]
            
            im = ax.imshow(S1_grid, extent=[0, 1, 0, 1], 
                          cmap='viridis', origin='lower', aspect='auto')
            ax.set_xlabel('x (M1)', fontsize=9)
            ax.set_ylabel('y (M2)', fontsize=9)
            ax.set_title(f'Training: G1\nt={t:.1f} (M2=0)', 
                        fontsize=10, fontweight='bold')
            plt.colorbar(im, ax=ax)
    
    # ===== Panels 8-11: Predicted Spatial Patterns =====
    times_pred = [0, 0.8, 2.0, 4.0]
    
    for idx, t in enumerate(times_pred):
        ax = plt.subplot(3, 5, idx + 11)
        df_t = df_pred[np.isclose(df_pred['time'], t, atol=0.01)]
        
        if len(df_t) > 0:
            x_vals = sorted(df_t['x'].unique())
            y_vals = sorted(df_t['y'].unique())
            
            # Create grid for Gene 1
            S1_grid = np.zeros((len(y_vals), len(x_vals)))
            for i, y in enumerate(y_vals):
                for j, x in enumerate(x_vals):
                    val = df_t[(df_t['x'] == x) & (df_t['y'] == y)]['S1'].values
                    if len(val) > 0:
                        S1_grid[len(y_vals)-1-i, j] = val[0]
            
            im = ax.imshow(S1_grid, extent=[0, 1, 0, 1], 
                          cmap='viridis', origin='lower', aspect='auto')
            ax.set_xlabel('x (M1)', fontsize=9)
            ax.set_ylabel('y (M2)', fontsize=9)
            ax.set_title(f'Predicted: G1\nt={t:.1f} (M1=0)', 
                        fontsize=10, fontweight='bold')
            plt.colorbar(im, ax=ax)
    
    plt.suptitle('Gene Regulatory Network: Comprehensive Analysis', 
                fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()


def create_time_series_comparison(df_train, df_pred, 
                                  save_path='time_series_comparison.png'):
    """
    Create time series comparison at center point
    
    Args:
        df_train: Training data
        df_pred: Predicted data
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Gene Expression Time Evolution (x=0.5, y=0.5)', 
                fontsize=14, fontweight='bold')
    
    # Extract center point data
    df_train_center = df_train[
        (df_train['x'] == 0.5) & (df_train['y'] == 0.5)
    ].sort_values('time')
    
    df_pred_center = df_pred[
        (df_pred['x'] == 0.5) & (df_pred['y'] == 0.5)
    ].sort_values('time')
    
    genes = ['S1', 'S2', 'S3', 'S4']
    gene_names = ['Gene 1', 'Gene 2', 'Gene 3', 'Gene 4']
    colors_train = '#2E86AB'
    colors_pred = '#A23B72'
    
    for idx, (gene, name) in enumerate(zip(genes, gene_names)):
        ax = axes[idx // 2, idx % 2]
        
        # Plot training
        ax.plot(df_train_center['time'], df_train_center[gene],
               'o-', linewidth=2.5, markersize=5, 
               label='Training (M2 removed)', color=colors_train, alpha=0.8)
        
        # Plot prediction
        ax.plot(df_pred_center['time'], df_pred_center[gene],
               's-', linewidth=2.5, markersize=5, 
               label='Predicted (M1 removed)', color=colors_pred, alpha=0.8)
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.set_xlabel('Time', fontweight='bold', fontsize=11)
        ax.set_ylabel('Expression Deviation', fontweight='bold', fontsize=11)
        ax.set_title(f'{name}', fontweight='bold', fontsize=12)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()


def create_all_genes_spatial(df_train, df_pred, 
                             save_path='all_genes_spatial.png'):
    """
    Create spatial patterns for all genes at t=0
    
    Args:
        df_train: Training data
        df_pred: Predicted data
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle('Spatial Expression Patterns at t=0 (Steady State)', 
                fontsize=14, fontweight='bold')
    
    genes = ['S1', 'S2', 'S3', 'S4']
    gene_names = ['Gene 1', 'Gene 2', 'Gene 3', 'Gene 4']
    
    for gene_idx, (gene, name) in enumerate(zip(genes, gene_names)):
        # Training data (top row)
        ax = axes[0, gene_idx]
        df_t0 = df_train[df_train['time'] == 0]
        
        x_vals = sorted(df_t0['x'].unique())
        y_vals = sorted(df_t0['y'].unique())
        
        grid = np.zeros((len(y_vals), len(x_vals)))
        for i, y in enumerate(y_vals):
            for j, x in enumerate(x_vals):
                val = df_t0[(df_t0['x'] == x) & (df_t0['y'] == y)][gene].values
                if len(val) > 0:
                    grid[len(y_vals)-1-i, j] = val[0]
        
        im = ax.imshow(grid, extent=[0, 1, 0, 1], cmap='RdYlBu_r', 
                      origin='lower', aspect='auto')
        ax.set_title(f'Training: {name}\n(M2 removed)', fontweight='bold')
        ax.set_xlabel('x (M1)')
        ax.set_ylabel('y (M2)')
        plt.colorbar(im, ax=ax)
        
        # Predicted data (bottom row)
        ax = axes[1, gene_idx]
        df_t0 = df_pred[df_pred['time'] == 0]
        
        grid = np.zeros((len(y_vals), len(x_vals)))
        for i, y in enumerate(y_vals):
            for j, x in enumerate(x_vals):
                val = df_t0[(df_t0['x'] == x) & (df_t0['y'] == y)][gene].values
                if len(val) > 0:
                    grid[len(y_vals)-1-i, j] = val[0]
        
        im = ax.imshow(grid, extent=[0, 1, 0, 1], cmap='RdYlBu_r', 
                      origin='lower', aspect='auto')
        ax.set_title(f'Predicted: {name}\n(M1 removed)', fontweight='bold')
        ax.set_xlabel('x (M1)')
        ax.set_ylabel('y (M2)')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()


def main():
    """Main visualization pipeline"""
    
    print("="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    df_train = pd.read_csv('GRN_experiment_M2_removal.csv')
    df_pred = pd.read_csv('GRN_experiment_M1_removal_predicted.csv')
    A_disc = np.loadtxt('gene_regulatory_matrix.txt', dtype=int)
    B_disc = np.loadtxt('morphogen_coupling_matrix.txt', dtype=int)
    print("  ✓ Data loaded")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    create_network_diagram(A_disc, 'network_diagram.png')
    
    create_comprehensive_analysis(df_train, df_pred, A_disc, B_disc,
                                  'comprehensive_analysis.png')
    
    create_time_series_comparison(df_train, df_pred,
                                  'time_series_comparison.png')
    
    create_all_genes_spatial(df_train, df_pred,
                            'all_genes_spatial.png')
    
    print("\n" + "="*80)
    print("✓ ALL VISUALIZATIONS CREATED")
    print("="*80)


if __name__ == "__main__":
    main()