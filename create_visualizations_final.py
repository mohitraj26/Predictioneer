"""
═══════════════════════════════════════════════════════════════════════════════
ENHANCED VISUALIZATION SCRIPT
Creates publication-quality figures for GRN analysis

Author: Competition Submission Team
Date: February 2026
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib import cm
import seaborn as sns

# Configure matplotlib for high-quality output
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.5


def create_network_topology(A_disc, save_path='network_topology.png'):
    """
    Create professional network topology diagram
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.axis('off')
    
    # Title
    ax.text(0, 2.2, 'Gene Regulatory Network Topology', 
           ha='center', fontsize=18, fontweight='bold')
    
    # Gene positions (circular layout)
    angles = np.array([90, 0, 270, 180]) * np.pi / 180
    radius = 1.5
    gene_pos = {
        i: (radius * np.cos(angles[i]), radius * np.sin(angles[i]))
        for i in range(4)
    }
    
    # Colors
    gene_colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
    
    # Draw genes
    for i, (x, y) in gene_pos.items():
        circle = Circle((x, y), 0.35, color=gene_colors[i], 
                       ec='black', linewidth=3, zorder=10)
        ax.add_patch(circle)
        ax.text(x, y, f'Gene {i+1}', ha='center', va='center',
               fontsize=14, fontweight='bold', color='white', zorder=11)
    
    # Draw interactions
    for i in range(4):
        for j in range(4):
            if i != j and A_disc[i, j] != 0:
                x1, y1 = gene_pos[j]
                x2, y2 = gene_pos[i]
                
                # Calculate positions
                dx, dy = x2 - x1, y2 - y1
                length = np.sqrt(dx**2 + dy**2)
                dx, dy = dx/length, dy/length
                
                start_x = x1 + 0.35 * dx
                start_y = y1 + 0.35 * dy
                end_x = x2 - 0.35 * dx
                end_y = y2 - 0.35 * dy
                
                # Style based on interaction
                if A_disc[i, j] > 0:
                    color = '#27AE60'  # Green for activation
                    linestyle = '-'
                    arrowstyle = '->'
                    label = '+'
                else:
                    color = '#C0392B'  # Red for inhibition
                    linestyle = '--'
                    arrowstyle = '-|>'
                    label = '-'
                
                # Draw arrow
                arrow = FancyArrowPatch(
                    (start_x, start_y), (end_x, end_y),
                    arrowstyle=arrowstyle,
                    color=color,
                    linewidth=3,
                    linestyle=linestyle,
                    mutation_scale=30,
                    zorder=5,
                    connectionstyle="arc3,rad=0.15"
                )
                ax.add_patch(arrow)
                
                # Add label
                mid_x = (start_x + end_x) / 2
                mid_y = (start_y + end_y) / 2
                ax.text(mid_x, mid_y, label, fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle='circle', facecolor='white', 
                                edgecolor=color, linewidth=2),
                       ha='center', va='center', zorder=6)
    
    # Legend
    activation_line = mpatches.Patch(color='#27AE60', label='Activation (+1)')
    inhibition_line = mpatches.Patch(color='#C0392B', label='Inhibition (-1)')
    ax.legend(handles=[activation_line, inhibition_line],
             loc='upper right', fontsize=12, framealpha=0.95,
             edgecolor='black', fancybox=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ {save_path}")
    plt.close()


def create_matrix_heatmaps(A_disc, B_disc, save_path='matrix_heatmaps.png'):
    """
    Create heatmap visualizations of matrices
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gene Regulatory Matrix A
    im1 = ax1.imshow(A_disc, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    ax1.set_xticks(range(4))
    ax1.set_yticks(range(4))
    ax1.set_xticklabels(['Gene 1', 'Gene 2', 'Gene 3', 'Gene 4'], fontsize=11)
    ax1.set_yticklabels(['Gene 1', 'Gene 2', 'Gene 3', 'Gene 4'], fontsize=11)
    ax1.set_xlabel('Regulator (j)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Target (i)', fontsize=12, fontweight='bold')
    ax1.set_title('Gene Regulatory Matrix A\n(Aᵢⱼ = effect of j on i)', 
                 fontsize=13, fontweight='bold', pad=15)
    
    # Add values
    for i in range(4):
        for j in range(4):
            if i != j:
                value = int(A_disc[i, j])
                color = 'white' if abs(value) == 1 else 'gray'
                text = '+1' if value == 1 else ('-1' if value == -1 else '0')
                ax1.text(j, i, text, ha='center', va='center',
                        color=color, fontsize=14, fontweight='bold')
            else:
                ax1.text(j, i, 'X', ha='center', va='center',
                        color='gray', fontsize=12, fontweight='bold')
    
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Interaction Type', fontsize=11, fontweight='bold')
    
    # Morphogen Coupling Matrix B
    im2 = ax2.imshow(B_disc, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax2.set_xticks(range(2))
    ax2.set_yticks(range(4))
    ax2.set_xticklabels(['M₁(x)', 'M₂(y)'], fontsize=11)
    ax2.set_yticklabels(['Gene 1', 'Gene 2', 'Gene 3', 'Gene 4'], fontsize=11)
    ax2.set_xlabel('Morphogen', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Gene', fontsize=12, fontweight='bold')
    ax2.set_title('Morphogen Coupling Matrix B', 
                 fontsize=13, fontweight='bold', pad=15)
    
    # Add values
    for i in range(4):
        for j in range(2):
            value = int(B_disc[i, j])
            color = 'white' if abs(value) == 1 else 'gray'
            text = '+1' if value == 1 else ('-1' if value == -1 else '0')
            ax2.text(j, i, text, ha='center', va='center',
                    color=color, fontsize=14, fontweight='bold')
    
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Coupling Type', fontsize=11, fontweight='bold')
    
    plt.suptitle('Inferred Network Matrices', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ {save_path}")
    plt.close()


def create_spatial_evolution(df_train, df_pred, save_path='spatial_evolution.png'):
    """
    Create spatial evolution comparison
    """
    fig = plt.figure(figsize=(20, 10))
    
    times_to_plot = [0, 1.0, 2.0, 4.0]
    
    for idx, t in enumerate(times_to_plot):
        # Training data (M2 removed)
        ax = plt.subplot(2, 4, idx + 1)
        df_t = df_train[np.isclose(df_train['time'], t, atol=0.05)]
        
        if len(df_t) > 0:
            x_vals = sorted(df_t['x'].unique())
            y_vals = sorted(df_t['y'].unique())
            
            grid = np.zeros((len(y_vals), len(x_vals)))
            for i, y in enumerate(y_vals):
                for j, x in enumerate(x_vals):
                    val = df_t[(df_t['x'] == x) & (df_t['y'] == y)]['S1'].values
                    if len(val) > 0:
                        grid[len(y_vals)-1-i, j] = val[0]
            
            im = ax.imshow(grid, extent=[0, 1, 0, 1], cmap='RdYlBu_r',
                          origin='lower', aspect='auto')
            ax.set_xlabel('x (M₁)', fontsize=10, fontweight='bold')
            ax.set_ylabel('y (M₂)', fontsize=10, fontweight='bold')
            ax.set_title(f'Training: Gene 1\nt = {t:.1f} (M₂ removed)',
                        fontsize=11, fontweight='bold')
            plt.colorbar(im, ax=ax, label='Expression')
        
        # Predicted data (M1 removed)
        ax = plt.subplot(2, 4, idx + 5)
        df_t = df_pred[np.isclose(df_pred['time'], t, atol=0.05)]
        
        if len(df_t) > 0:
            x_vals = sorted(df_t['x'].unique())
            y_vals = sorted(df_t['y'].unique())
            
            grid = np.zeros((len(y_vals), len(x_vals)))
            for i, y in enumerate(y_vals):
                for j, x in enumerate(x_vals):
                    val = df_t[(df_t['x'] == x) & (df_t['y'] == y)]['S1'].values
                    if len(val) > 0:
                        grid[len(y_vals)-1-i, j] = val[0]
            
            im = ax.imshow(grid, extent=[0, 1, 0, 1], cmap='RdYlBu_r',
                          origin='lower', aspect='auto')
            ax.set_xlabel('x (M₁)', fontsize=10, fontweight='bold')
            ax.set_ylabel('y (M₂)', fontsize=10, fontweight='bold')
            ax.set_title(f'Predicted: Gene 1\nt = {t:.1f} (M₁ removed)',
                        fontsize=11, fontweight='bold')
            plt.colorbar(im, ax=ax, label='Expression')
    
    plt.suptitle('Spatial-Temporal Evolution: Training vs Predicted',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ {save_path}")
    plt.close()


def create_time_series_analysis(df_train, df_pred, save_path='time_series.png'):
    """
    Create time series comparison at key spatial points
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Center point
    df_train_center = df_train[(df_train['x'] == 0.5) & (df_train['y'] == 0.5)].sort_values('time')
    df_pred_center = df_pred[(df_pred['x'] == 0.5) & (df_pred['y'] == 0.5)].sort_values('time')
    
    genes = ['S1', 'S2', 'S3', 'S4']
    gene_names = ['Gene 1', 'Gene 2', 'Gene 3', 'Gene 4']
    
    for idx, (gene, name) in enumerate(zip(genes, gene_names)):
        ax = axes[idx // 2, idx % 2]
        
        # Training
        ax.plot(df_train_center['time'], df_train_center[gene],
               'o-', linewidth=2.5, markersize=6, label='Training (M₂ removed)',
               color='#3498DB', alpha=0.8)
        
        # Predicted
        ax.plot(df_pred_center['time'], df_pred_center[gene],
               's-', linewidth=2.5, markersize=6, label='Predicted (M₁ removed)',
               color='#E74C3C', alpha=0.8)
        
        ax.axhline(y=0, color='black', linestyle=':', alpha=0.5, linewidth=1.5)
        ax.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('Expression Deviation', fontsize=12, fontweight='bold')
        ax.set_title(f'{name} Expression at (x=0.5, y=0.5)',
                    fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.suptitle('Gene Expression Time Evolution',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ {save_path}")
    plt.close()


def create_all_genes_comparison(df_train, df_pred, save_path='all_genes.png'):
    """
    Create comparison of all genes at initial time
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    genes = ['S1', 'S2', 'S3', 'S4']
    gene_names = ['Gene 1', 'Gene 2', 'Gene 3', 'Gene 4']
    
    for gene_idx, (gene, name) in enumerate(zip(genes, gene_names)):
        # Training (top row)
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
        ax.set_title(f'Training: {name}\n(M₂ removed, t=0)',
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('x (M₁)', fontsize=10)
        ax.set_ylabel('y (M₂)', fontsize=10)
        plt.colorbar(im, ax=ax)
        
        # Predicted (bottom row)
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
        ax.set_title(f'Predicted: {name}\n(M₁ removed, t=0)',
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('x (M₁)', fontsize=10)
        ax.set_ylabel('y (M₂)', fontsize=10)
        plt.colorbar(im, ax=ax)
    
    plt.suptitle('All Genes: Spatial Patterns at Initial Time',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ {save_path}")
    plt.close()


def main():
    """Main visualization pipeline"""
    
    print("\n" + "="*80)
    print(" "*25 + "CREATING VISUALIZATIONS")
    print("="*80 + "\n")
    
    try:
        # Load data
        print("[1] Loading data files...")
        df_train = pd.read_csv('GRN_experiment_M2_removal.csv')
        df_pred = pd.read_csv('GRN_experiment_M1_removal_predicted.csv')
        A_disc = np.loadtxt('gene_regulatory_matrix.txt', dtype=int)
        B_disc = np.loadtxt('morphogen_coupling_matrix.txt', dtype=int)
        print("  ✓ All data loaded\n")
        
        # Create visualizations
        print("[2] Generating visualizations...")
        
        create_network_topology(A_disc, 'network_topology.png')
        create_matrix_heatmaps(A_disc, B_disc, 'matrix_heatmaps.png')
        create_spatial_evolution(df_train, df_pred, 'spatial_evolution.png')
        create_time_series_analysis(df_train, df_pred, 'time_series.png')
        create_all_genes_comparison(df_train, df_pred, 'all_genes.png')
        
        print("\n" + "="*80)
        print(" "*20 + "✓ ALL VISUALIZATIONS CREATED")
        print("="*80 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n✗ ERROR: Required file not found - {e}")
        print("Please run grn_inference_final.py first to generate the data files\n")
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")


if __name__ == "__main__":
    main()