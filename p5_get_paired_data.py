import h5py
import scanpy as sc
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from os.path import join
import os
import zipfile
import json
import argparse
import seaborn as sns
import openslide
from tqdm import tqdm
import pickle

class MM_ST:
    def __init__(self, dataset_dir, work_dir, case_id, meta_info):

        print(case_id)
        # Original files
        self.dataset_dir = dataset_dir
        self.work_dir = work_dir
        self.case_id = case_id
        # nuclei.io processed files
        self.nuc_stat_path = join(self.work_dir, 'results_zhi', 'processed_WSIs', f'{case_id}', 'stardist_results', 'nuclei_stat_f16.npy')
        self.columns_path = join(self.work_dir, 'results_zhi', 'processed_WSIs', f'{case_id}', 'stardist_results', 'nuclei_stat_columns.csv')
        self.rows_path = join(self.work_dir, 'results_zhi', 'processed_WSIs', f'{case_id}', 'stardist_results', 'nuclei_stat_index.csv')
        self.centroids_path = join(self.work_dir, 'results_zhi', 'processed_WSIs', f'{case_id}', 'stardist_results', 'centroids.npy')
        self.contours_path = join(self.work_dir, 'results_zhi', 'processed_WSIs', f'{case_id}', 'stardist_results', 'contours.npy')
        
        # Output files
        self.output_folder = join(self.work_dir, 'results_zhi', 'processed_WSIs', f'{case_id}')
        os.makedirs(self.output_folder, exist_ok=True)

        self.meta_info = meta_info
        st_technology = meta_info["st_technology"]
        spot_diameter = meta_info["spot_diameter"]
        inter_spot_dist = meta_info["inter_spot_dist"] # the distance in um between two spots on the same row.
        spot_estimate_dist = meta_info["spot_estimate_dist"]
        """
        # if spot_estimate_dist equals 6:
        +---+---+---+---+---+---+---+---+
        |   |   |   |   |   |   |   |   |  <- Row 1
        +---+---+---+---+---+---+---+---+
        |   | S |   |   |   |   |   | S |  <- Row 2 (spot_estimate_dist = 6)
        +---+---+---+---+---+---+---+---+
        |   |   |   |   |   |   |   |   |  <- Row 3
        +---+---+---+---+---+---+---+---+
        |   |   |   |   |   |   |   |   |  <- Row 4
        +---+---+---+---+---+---+---+---+
        Spots in Row 2 span from Column 2 to Column 8, which is a column distance of 6.
        """


    def _read_data(self):
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
        sc.pp.log1p(self.st_data)  # Logarithmize the data
        # Scale the data
        #sc.pp.scale(self.st_data)
        
        self.patch_data = h5py.File(join(self.dataset_dir, "patches", f'{self.case_id}.h5'), 'r')
        patch_data_barcode = self.patch_data['barcode'][:].reshape(-1).astype(str)
        patch_data_coords = self.patch_data['coords'][:]
        patch_data_df = pd.DataFrame(np.c_[patch_data_barcode, patch_data_coords], columns=['barcode', 'x', 'y'])
        print("Number of patches:", len(self.patch_data['barcode']))
        print("Number of rows from ST data:", self.st_data.obs.shape[0])
        
        # Load corresponding image patch
        spot_diameter_in_micron = self.meta_info["spot_diameter"]
        spot_diameter_in_pixel = spot_diameter_in_micron / self.mpp
        
        # self.st_data.uns["spatial"]["ST"]["images"]["downscaled_fullres"].shape
        if not isinstance(self.st_data.X, np.ndarray):
            st_dense_matrix = pd.DataFrame(self.st_data.X.toarray(), index=self.st_data.obs.index, columns=self.st_data.var.index)
        else:
            st_dense_matrix = pd.DataFrame(self.st_data.X, index=self.st_data.obs.index, columns=self.st_data.var.index)

        # Open an H5 file to save the data

        this_meta = self.meta_info.to_dict()
        this_meta["mpp"] = self.mpp
        this_meta["magnification"] = self.magnification
        
        data_dict = {"metadata": this_meta,
                     "data": {},
                     "gene_expression": st_dense_matrix}
        
        print(this_meta)
        breakpoint()
        if "pxl_col_in_fullres" in self.st_data.obs.columns:
            use_st_data_coords = True
        else:
            print("########## pxl_col_in_fullres not found. Using pxl_col instead.")
            if len(self.st_data.obs) != len(patch_data_coords):
                print("Lengths do not match.")
                return
            use_st_data_coords = False

        matched_cells_data = []
        for idx, barcode in enumerate(self.st_data.obs.index):
            if use_st_data_coords:
                center_x, center_y = self.st_data.obs.loc[barcode, ["pxl_col_in_fullres", 'pxl_row_in_fullres']].astype(float).round().astype(int)
            else:
                if barcode not in patch_data_barcode:
                    print(f"{barcode} not found in patch data.")
                    raise Exception()
                center_x, center_y = patch_data_coords[idx]

            x1, y1 = int(np.round(center_x - spot_diameter_in_pixel / 2)), int(np.round(center_y - spot_diameter_in_pixel / 2))
            x2, y2 = int(np.round(center_x + spot_diameter_in_pixel / 2)), int(np.round(center_y + spot_diameter_in_pixel / 2))
            
            # get patch image
            patch_img = self.slide.read_region((x1, y1), 0, (x2 - x1, y2 - y1))
            patch_img_np = np.array(patch_img)[..., :3]
            
            # find matched cells in nuclei statistics
            subset_index = (self.centroids[:,0] > x1) & (self.centroids[:,0] < x2) & (self.centroids[:,1] > y1) & (self.centroids[:,1] < y2)
            n_matched_cells = subset_index.sum()
            # print(n_matched_cells)
            matched_nucstat = self.nuc_stat.loc[subset_index]
            matched_contours = self.contours[subset_index]
            # Adjust contours by subtracting x1 and y1
            adjusted_contours = [contour - [x1, y1] for contour in matched_contours]
            
            gene_expression = st_dense_matrix.loc[barcode]
            # Save the number of matched cells along with the barcode and patch image
            matched_cells_data.append((barcode, n_matched_cells, patch_img_np, adjusted_contours))
            
            data_dict["data"][barcode] = {"n_matched_cells": n_matched_cells, "matched_nucstat": matched_nucstat,
                                                    "matched_contours_global": matched_contours, "matched_contours_local": adjusted_contours,
                                                    "patch_img_np": patch_img_np, "patch_x1": x1, "patch_y1": y1, "patch_x2": x2, "patch_y2": y2}
        with open(pickle_file_to_save, 'wb') as f:
            pickle.dump(data_dict, f)
        # breakpoint()
        # Sort by number of matched cells in descending order
        matched_cells_data_sorted = sorted(matched_cells_data, key=lambda x: x[1], reverse=True)

        # Select the top 10, bottom 10, and middle 10 barcodes
        top_10 = matched_cells_data_sorted[-10:]  # Last 10 elements (highest values)
        bottom_10 = matched_cells_data_sorted[:10]  # First 10 elements (lowest values)
        middle_10 = matched_cells_data_sorted[len(matched_cells_data_sorted)//2 - 5: len(matched_cells_data_sorted)//2 + 5]  # Middle 10 elements
        quantile_75_index = int(0.75 * len(matched_cells_data_sorted))
        quantile_75 = matched_cells_data_sorted[quantile_75_index - 5: quantile_75_index + 5]  # 10 elements around the 75th percentile
        quantile_99_index = int(0.99 * len(matched_cells_data_sorted))
        quantile_99 = matched_cells_data_sorted[quantile_99_index - 5: quantile_99_index + 5]  # 10 elements around the 90th percentile

        # Combine these lists
        selected_data = bottom_10 + middle_10 + quantile_75 + quantile_99 + top_10

        # Plotting the selected patch images
        fig, axes = plt.subplots(5, 10, figsize=(30, 12))
        axes = axes.flatten()

        for ax, (barcode, n_matched_cells, patch_img_np, contour) in zip(axes, selected_data):
            ax.imshow(patch_img_np)
            # Plot the matched contours on the image
            # print(contour)
            for c in contour:
                ax.plot(c[:, 0], c[:, 1], color='yellow', linewidth=1)
            ax.set_title(f'{barcode}\nMatched Cells: {n_matched_cells}')
            ax.axis('off')

        # Save the figure to a file
        output_image_path = join(args.work_dir, 'results_zhi', 'processed_WSIs', f'{case_id}', f'ST_spots_with_n_of_cells.png')
        plt.tight_layout()
        plt.savefig(output_image_path, dpi=300)
        plt.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='/project/zhihuanglab/common/datasets/HEST-1K/hest_data/', type=str)
    parser.add_argument('--work_dir', default="/project/zhihuanglab/yutong/HEST_project/", type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    df_meta = pd.read_csv(join(args.work_dir, 'results_zhi', 'meta_homo_sapiens_Xenium_ST.csv'))
    for i in tqdm(df_meta.index):
        case_id = df_meta.loc[i, "filename"].split(".tif")[0]
        (df_meta["st_technology"] + "  " + df_meta["spot_diameter"].astype(str) + "  " + df_meta["inter_spot_dist"].astype(str)).value_counts()
        meta_info = df_meta.loc[i]
        self = MM_ST(args.dataset_dir, args.work_dir, case_id, meta_info)
        
        self._read_data()

    #breakpoint()