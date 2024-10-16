import h5py
import scanpy as sc
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from os.path import join
import os
import pickle
import openslide
import argparse
from tqdm import tqdm
import seaborn as sns

class MM_ST:
    def __init__(self, dataset_dir, work_dir, case_id, meta_info):
        print(f"Processing Case ID: {case_id}")
        
        # Directory paths
        self.dataset_dir = dataset_dir
        self.work_dir = work_dir
        self.case_id = case_id
        
        # Paths to nuclei statistics and related files
        self.nuc_stat_path = join(self.work_dir, 'results_zhi', 'processed_WSIs', f'{case_id}', 
                                  'stardist_results', 'nuclei_stat_f16.npy')
        self.columns_path = join(self.work_dir, 'results_zhi', 'processed_WSIs', f'{case_id}', 
                                 'stardist_results', 'nuclei_stat_columns.csv')
        self.rows_path = join(self.work_dir, 'results_zhi', 'processed_WSIs', f'{case_id}', 
                              'stardist_results', 'nuclei_stat_index.csv')
        self.centroids_path = join(self.work_dir, 'results_zhi', 'processed_WSIs', f'{case_id}', 
                                   'stardist_results', 'centroids.npy')
        self.contours_path = join(self.work_dir, 'results_zhi', 'processed_WSIs', f'{case_id}', 
                                  'stardist_results', 'contours.npy')
        
        # Output directory
        self.output_folder = join(self.work_dir, 'results_zhi', 'processed_WSIs', f'{case_id}')
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Metadata
        self.meta_info = meta_info
        self.st_technology = meta_info["st_technology"]
        self.spot_diameter = meta_info["spot_diameter"]
        self.inter_spot_dist = meta_info["inter_spot_dist"]  # Distance in microns between spots
        
    def _read_data(self):
        """
        Reads and processes spatial transcriptomics data, matches cells with nucleus statistics,
        and saves the matched data.
        """
        pickle_file_to_save = join(self.output_folder, f'matched_cells_data.pkl')
        # if os.path.exists(pickle_file_to_save):
        #     print(f"File already exists. Skipping...")
        #     return
        # Read image data
        self.slide = openslide.OpenSlide(join(self.dataset_dir, "wsis", f'{self.case_id}.tif'))
        self.mpp = float(self.slide.properties['openslide.mpp-x'])
        reference_mpp_1x = 10  # microns per pixel at 1x
        self.magnification = reference_mpp_1x / self.mpp
        
        # Read nuclei.io processed data
        if not os.path.exists(self.nuc_stat_path):
            print(f"nuclei stat file not found. skipping...")
            return
        self.nuc_stat = np.load(self.nuc_stat_path)
        self.columns = [f"{eval(v)[0]} | {eval(v)[1]}" for v in pd.read_csv(self.columns_path, index_col=0).values.astype(str).reshape(-1)]
        self.nuc_stat = pd.DataFrame(self.nuc_stat, columns=self.columns)
        self.centroids = np.load(self.centroids_path)
        self.contours = np.load(self.contours_path, allow_pickle=True)
        print("Number of nuclei:", len(self.nuc_stat))
        self.nuc_stat["centroid_x"] = self.centroids[:,0]
        self.nuc_stat["centroid_y"] = self.centroids[:,1]
        # Read spatial data
        self.st_data = sc.read_h5ad(join(self.dataset_dir, "st", f'{self.case_id}.h5ad'))
        
        # Add normalization steps here
        sc.pp.normalize_total(self.st_data, target_sum=1e4)  # Normalize to counts per 10,000 (CPM)
        sc.pp.log1p(self.st_data)
        
        # Step 5: Convert Gene Expression Data to Dense DataFrame
        if not isinstance(self.st_data.X, np.ndarray):
            st_dense_matrix = pd.DataFrame(self.st_data.X.toarray(), index=self.st_data.obs.index, 
                                           columns=self.st_data.var.index)
        else:
            st_dense_matrix = pd.DataFrame(self.st_data.X, index=self.st_data.obs.index, 
                                           columns=self.st_data.var.index)
        print("Converted gene expression data to dense DataFrame.")
        
        # Step 6: Prepare Metadata for Saving
        this_meta = self.meta_info.to_dict()
        this_meta["mpp"] = self.mpp
        this_meta["magnification"] = self.magnification
        
        data_dict = {
            "metadata": this_meta,
            "data": {},
            "gene_expression": st_dense_matrix
        }
        
        # Step 7: Assign Meaningful Column Names to Spatial Coordinates
        if "pxl_col_in_fullres" in self.st_data.obs.columns and "pxl_row_in_fullres" in self.st_data.obs.columns:
            spatial_df = self.st_data.obs[['pxl_col_in_fullres', 'pxl_row_in_fullres']].copy()
            spatial_df.rename(columns={'pxl_col_in_fullres': 'x_coord', 'pxl_row_in_fullres': 'y_coord'}, inplace=True)
            self.st_data.obsm['spatial'] = spatial_df
            print("Assigned 'x_coord' and 'y_coord' to obsm['spatial'].")
            self.st_data.obs['x_coord'] = spatial_df['x_coord']
            self.st_data.obs['y_coord'] = spatial_df['y_coord']
            print("Added 'x_coord' and 'y_coord' to obs.")
        else:
            # Use the coordinates from self.st_data.obsm['spatial']
            if 'spatial' in self.st_data.obsm:
                spatial_data = self.st_data.obsm['spatial']
                if isinstance(spatial_data, np.ndarray):
                    spatial_df = pd.DataFrame(
                        spatial_data,
                        columns=['pxl_col_in_fullres', 'pxl_row_in_fullres'],
                        index=self.st_data.obs.index  # Ensure the index matches
                    )
                    spatial_df.rename(columns={'pxl_col_in_fullres': 'x_coord', 'pxl_row_in_fullres': 'y_coord'}, inplace=True)
                    self.st_data.obsm['spatial'] = spatial_df
                    print("Assigned 'pxl_col_in_fullres' and 'pxl_row_in_fullres' to obsm['spatial'] from NumPy array.")
                    self.st_data.obs['x_coord'] = spatial_df['x_coord']
                    self.st_data.obs['y_coord'] = spatial_df['y_coord']
                    print("Added 'x_coord' and 'y_coord' to obs.")
                else:
                    print("The 'spatial' data in obsm is a DataFrame.")
            else:
                print("No spatial data found in obsm['spatial'].")

        # Step 8: Extract Spatial Coordinates
        cell_coords = self.st_data.obsm['spatial'][['x_coord', 'y_coord']].values
        print(f"Extracted spatial coordinates with shape {cell_coords.shape}.")
        
        # Step 9: Extract Nucleus Centroid Coordinates
        nucleus_coords = self.nuc_stat[['centroid_x', 'centroid_y']].values
        print(f"Extracted nucleus centroids with shape {nucleus_coords.shape}.")
        
        # Step 10: Build KDTree for Efficient Spatial Matching
        try:
            nucleus_tree = cKDTree(nucleus_coords)
            print("Built KDTree for nucleus centroids.")
        except Exception as e:
            print(f"Error building KDTree: {e}")
            return
        
        # Step 11: Query Nearest Nucleus for Each Cell
        distance_threshold = 20  # Adjust based on your data's spatial scale
        distances, indices = nucleus_tree.query(cell_coords, distance_upper_bound=distance_threshold)
        print(f"Performed nearest neighbor search with distance threshold {distance_threshold}.")
        
        # Step 12: Assign Matched Nucleus Statistics to Cells
        # Initialize columns in adata.obs for nucleus statistics
        for col in self.nuc_stat.columns:
            self.st_data.obs[col] = np.nan  # Initialize with NaN
        
        # Iterate over each cell to assign nucleus stats
        matched_cells = 0
        matched_barcodes = []  # Initialize matched_barcodes
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            if dist != np.inf:
                barcode = self.st_data.obs.index[i]
                matched_barcodes.append(barcode)
                for col in self.nuc_stat.columns:
                    self.st_data.obs.at[barcode, col] = self.nuc_stat.iloc[idx][col]
                matched_cells += 1
        total_nuclei = len(self.st_data.obs)
        print(f"Number of cells matched with nuclei: {matched_cells} out of {len(self.st_data.obs)}.")
        print(f"Number of unmatched cells: {len(self.st_data.obs) - matched_cells}.")

        # Step 13: Save Matched Data to Pickle File
        # Collect matched nuclei statistics along with spatial coordinates
        matched_nucstat = self.st_data.obs.loc[matched_barcodes]
        
        # Filter gene_expression to matched cells
        filtered_gene_expression = data_dict["gene_expression"].loc[matched_barcodes]
        print(f"Filtered gene expression data to matched cells for {self.case_id}.")
        
        # Update data_dict
        data_dict["data"] = matched_nucstat.to_dict(orient='index')
        data_dict["gene_expression"] = filtered_gene_expression  # Update to include only matched cells
            
        
        # Save to data_dict
        data_dict["data"] = matched_nucstat.to_dict(orient='index')
        #breakpoint()
        # Save the pickle file
        try:
            with open(pickle_file_to_save, 'wb') as f:
                pickle.dump(data_dict, f)
            print(f"Matched data saved to {pickle_file_to_save}.")
        except Exception as e:
            print(f"Error saving matched data: {e}")
            return
        
        # Optional: Sort matched cells by number of matched nuclei (if applicable)
        # This depends on your specific requirements and data structure
        # For demonstration, we'll assume each cell matches to one nucleus
        # Hence, sorting isn't necessary here
        
        # Step 14: Visual Verification (Optional)
        self._visualize_matches(matched_cells, total_nuclei)
        
    def _visualize_matches(self, matched_cells, total_nuclei):
        """
        Visualizes the spatial distribution of matched cells and nuclei.
        """
        # Extract spatial coordinates
        x = self.st_data.obsm['spatial']['x_coord']
        y = self.st_data.obsm['spatial']['y_coord']
        
        # Extract matched nuclei centroids
        matched_nuclei = self.st_data.obs.dropna(subset=["centroid_x", "centroid_y"])
        nuc_x = matched_nuclei['centroid_x']
        nuc_y = matched_nuclei['centroid_y']
        
        # Plotting
        plt.figure(figsize=(10, 8))
        plt.scatter(x, y, s=10, alpha=0.5, label='Cells')
        plt.scatter(nuc_x, nuc_y, s=20, alpha=0.7, c='red', label='Nuclei')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(f"Spatial Distribution of Cells and Nuclei\nMatched cells: {matched_cells} out of {total_nuclei}")
        plt.legend()
        
        filename = f"{case_id}_spatial_distribution.png"
        filepath = os.path.join("/project/zhihuanglab/yutong/HEST_project/results_zhi/paired_data_visualization", filename)
        # Save the figure
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        print(f"Plot saved")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='/project/zhihuanglab/common/datasets/HEST-1K/hest_data/', type=str)
    parser.add_argument('--work_dir', default="/project/zhihuanglab/yutong/HEST_project/", type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Load metadata
    df_meta = pd.read_csv(join(args.work_dir, 'results_zhi', 'meta_homo_sapiens_Xenium_ST.csv'))
    # Iterate over each case in metadata
    for i in tqdm(df_meta.index, desc="Processing Cases"):
        case_id = df_meta.loc[i, "filename"].split(".tif")[0]
        meta_info = df_meta.loc[i]
        
        # Initialize MM_ST object
        mm_st = MM_ST(dataset_dir=args.dataset_dir, work_dir=args.work_dir, 
                     case_id=case_id, meta_info=meta_info)
        # Read and process data
        mm_st._read_data()

    breakpoint()

