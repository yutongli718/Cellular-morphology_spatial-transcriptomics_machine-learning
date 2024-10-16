import os
import pandas as pd
from tqdm import tqdm
from os.path import join
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    processed_data_dir = "/project/zhihuanglab/yutong/HEST_project/results_zhi/processed_WSIs"
    df_meta = pd.read_csv(join('/project/zhihuanglab/yutong/HEST_project/results_zhi', 'meta_homo_sapiens_Xenium_ST.csv'))
    case_id = "TENX106"
    
    gene_name = "ACTA2"  # Replace with the gene of interest

    with open(join(processed_data_dir, case_id, "ACTA2_filtered_geneExp.pkl"), "rb") as f:
        matched_cells_data = pickle.load(f)

    metadata = matched_cells_data["metadata"]
    all_data = matched_cells_data["data"]
    gene_expression = matched_cells_data["gene_expression"]
   
    # Ensure that gene_expression and all_data share the same barcodes
    common_barcodes = gene_expression.index.intersection(all_data.keys())
    # Filter gene_expression and all_data to only include common barcodes
    gene_expression = gene_expression.loc[common_barcodes]
    all_data = {barcode: all_data[barcode] for barcode in common_barcodes}

    assert len(all_data) == len(gene_expression)  # Ensure data and gene expression share the same barcodes

    feature_area = []
    feature_major_axis = []
    gene_exprs = []

    for barcode in tqdm(all_data.keys()):
        data = all_data[barcode]

        # Check if the necessary morphology features are present in data
        if "Morphology | area" not in data or "Morphology | major_axis_length" not in data:
            print(f"Barcode {barcode} is missing morphology features. Skipping.")
            continue

        # Get the morphology features
        area = data["Morphology | area"]
        major_axis_length = data["Morphology | major_axis_length"]

        # Get gene expression value
        gene_expr = gene_expression.loc[barcode]
        if gene_name in gene_expr.index:
            gene_value = gene_expr[gene_name]
        else:
            gene_value = 0  # or handle as appropriate for your data

        # Append to lists
        feature_area.append(area)
        feature_major_axis.append(major_axis_length)
        gene_exprs.append(gene_value)

    # Convert lists to numpy arrays
    feature_area = np.array(feature_area, dtype=np.float64)
    feature_major_axis = np.array(feature_major_axis, dtype=np.float64)
    gene_exprs = np.array(gene_exprs, dtype=np.float64)

    # Check if feature_area is empty
    if feature_area.size == 0:
        print("No data available for plotting. Exiting.")
        exit()

    # Normalize the feature_area if desired
    feature_area_normalized = feature_area / np.max(feature_area) * 5

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.kdeplot(x=feature_area_normalized, y=gene_exprs, cmap="Reds", fill=True, ax=ax)
    correlation = np.corrcoef(feature_area_normalized, gene_exprs)[0, 1]
    ax.set(title=f"Density Map of Area and {gene_name} Expression\nPearson Correlation: {correlation:.3f}",
           xlabel="Normalized Cell Area", ylabel=f"{gene_name} Expression")
    ax.grid(True)

    # Save the figure to a file
    fig.savefig(join(processed_data_dir, case_id, f"area_{gene_name}_density_map.png"))
