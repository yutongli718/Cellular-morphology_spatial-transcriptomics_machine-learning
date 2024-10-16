import os
import pandas as pd
from os.path import join, isfile
import numpy as np
import pickle

def filter_cells_by_gene_expression(pickle_file_path, gene_identifiers):
    try:
        with open(pickle_file_path, 'rb') as f:
            data_dict = pickle.load(f)
    except Exception as e:
        print(f"Failed to load pickle file at {pickle_file_path}: {e}")
        return None, None, None, None

    gene_expression = data_dict.get('gene_expression')
    if gene_expression is None:
        print(f"'gene_expression' key not found in {pickle_file_path}.")
        return None, None, None, None

    if not isinstance(gene_expression, pd.DataFrame):
        gene_expression = pd.DataFrame(gene_expression)

    target_gene = next((gene for gene in gene_identifiers if gene in gene_expression.columns), None)
    if not target_gene:
        print(f"Neither {gene_identifiers[0]} nor {gene_identifiers[1]} found in {pickle_file_path}.")
        return None, None, None, None

    filtered_gene_expression = gene_expression[gene_expression[target_gene] > 0]
    if filtered_gene_expression.empty:
        print(f"No cells passed the expression filter for {target_gene} in {pickle_file_path}.")
        return None, None, None, None

    metadata = data_dict.get('metadata')
    data_dict_data = data_dict.get('data')

    return filtered_gene_expression, metadata, data_dict_data, target_gene

def process_case(processed_data_dir, case_id, gene_identifiers):
    pickle_file_path = join(processed_data_dir, case_id, "matched_cells_data.pkl")
    if not isfile(pickle_file_path):
        print(f"File for {case_id} not found at {pickle_file_path}, skipping.")
        return

    filtered_gene_expression, metadata, data, target_gene = filter_cells_by_gene_expression(
        pickle_file_path,
        gene_identifiers
    )

    if filtered_gene_expression is not None:
        output_file_name = "ACTA2_filtered_geneExp.pkl"
        output_file_path = join(processed_data_dir, case_id, output_file_name)

        filtered_data_dict = {
            'gene_expression': filtered_gene_expression,
            'metadata': metadata,
            'data': data
        }

        try:
            with open(output_file_path, 'wb') as f:
                pickle.dump(filtered_data_dict, f)
            print(f"Filtered gene expression data saved for {case_id} to {output_file_path}")
        except Exception as e:
            print(f"Error saving filtered data for {case_id}: {e}")

def main():
    processed_data_dir = "/project/zhihuanglab/yutong/HEST_project/results_zhi/processed_WSIs"
    metadata_csv_path = "/project/zhihuanglab/yutong/HEST_project/results_zhi/meta_homo_sapiens_Xenium_ST.csv"
    gene_identifiers = ["Acta2", "ACTA2"]

    if not os.path.isfile(metadata_csv_path):
        print(f"Metadata CSV file not found at {metadata_csv_path}. Exiting.")
        return

    df_meta = pd.read_csv(metadata_csv_path)
    if "filename" not in df_meta.columns:
        print("The metadata CSV does not contain a 'filename' column. Exiting.")
        return

    for idx, row in df_meta.iterrows():
        filename = row["filename"]
        case_id = os.path.splitext(filename)[0]
        process_case(processed_data_dir, case_id, gene_identifiers)

if __name__ == "__main__":
    main()

