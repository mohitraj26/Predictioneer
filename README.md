# Gene Regulatory Network Inference – Complete Solution  
**AZeotropy ’26 Competition | IIT Bombay**

---

## 📌 Overview
This repository contains a complete, end-to-end solution for **Gene Regulatory Network (GRN) inference from spatial gene expression dynamics**, developed for the **AZeotropy ’26** competition conducted by **IIT Bombay**.

The solution covers:
- Parameter inference using regularized regression
- Steady-state constraint analysis
- Dynamic simulation and prediction
- Visualization and interpretation of inferred networks

---

## 📂 Contents
1. Python Scripts  
2. Required Data Files  
3. Output Files for Submission  
4. How to Run  
5. Results Summary  
6. Methodology  
7. Troubleshooting  
8. File Structure  
9. Mathematical Formulation  

---

## 1️⃣ Python Scripts

### `grn_inference.py`
Main inference script containing:
- `GRNInference` class (complete implementation)
- Parameter estimation using **Ridge Regression**
- Steady-state constraint analysis
- Forward Euler simulation
- Model validation and prediction
- Extensive inline documentation

### `create_visualizations.py`
Visualization and analysis script containing:
- GRN network topology diagram
- Comprehensive diagnostic plots
- Time-series comparison plots
- Spatial gene expression patterns
- All-genes spatial visualization

---

## 2️⃣ Required Data Files  
*(Place all files in the same directory)*

| File Name | Description |
|----------|-------------|
| `GRN_experiment_M2_removal.csv` | Training data (Experiment A) |
| `GRN_experiment_M1_removal.csv` | Test structure (Experiment B) |

---

## 3️⃣ Output Files for Submission

### **PART I – Network Inference (40%)**
- `gene_regulatory_matrix.txt` → 4×4 matrix **A** ∈ {−1, 0, +1}  
- `morphogen_coupling_matrix.txt` → 4×2 matrix **B** ∈ {−1, 0, +1}

### **PART II – Prediction (30%)**
- `GRN_experiment_M1_removal_predicted.csv` → Predicted gene expression

### **PART III – Report (30%)**
- `GRN_Inference_Report.docx` → Technical report

### **Supporting Visualizations**
- `network_diagram.png`
- `comprehensive_analysis.png`
- `time_series_comparison.png`
- `all_genes_spatial.png`

---

## 4️⃣ How to Run

### Step 1: Install Dependencies
```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn
