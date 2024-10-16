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
# module load openslide/4.0.0.3
# conda install -c bioconda openslide-python
# pip install openslide-python

# export LD_LIBRARY_PATH=/home/zhihuang/anaconda3/envs/HEST/lib:$LD_LIBRARY_PATH

if __name__ == "__main__":
    wsi_dir = "/project/zhihuanglab/common/datasets/HEST-1K/hest_data/wsis"
    metadata_dir = "/project/zhihuanglab/common/datasets/HEST-1K/hest_data/metadata"
    save_dir = "/project/zhihuanglab/yutong/HEST_project/results_zhi"

    #################################################################################
    # Step 1. Gather information
    #################################################################################
    if not os.path.exists(join(save_dir, "meta_full.csv")):
        df = pd.DataFrame()
        pbar = tqdm(total=len(os.listdir(wsi_dir)))
        for i, filename in enumerate(os.listdir(wsi_dir)):
            pbar.update(1)
            if filename.endswith(".tif"):
                svs = openslide.OpenSlide(os.path.join(wsi_dir, filename))
                df.loc[i, "filename"] = filename
                df.loc[i, "magnification_x"] = svs.properties["openslide.mpp-x"]
                df.loc[i, "magnification_y"] = svs.properties["openslide.mpp-y"]
                # read metadata
                with open(join(metadata_dir, filename.replace(".tif", ".json")), "r") as f:
                    metadata = json.load(f)
                df.loc[i, "pixel_size_um_estimated"] = metadata["pixel_size_um_estimated"]
                df.loc[i, "magnification_in_metadata"] = float(metadata["magnification"].replace("x", ""))
                df.loc[i, "dataset_title"] = metadata["dataset_title"]
                df.loc[i, "st_technology"] = metadata["st_technology"]
                df.loc[i, "organ"] = metadata["organ"]
                df.loc[i, "tissue"] = metadata["tissue"]
                df.loc[i, "disease_comment"] = metadata["disease_comment"]
                df.loc[i, "species"] = metadata["species"]
                if "spot_estimate_dist" in metadata.keys():
                    df.loc[i, "spot_estimate_dist"] = metadata["spot_estimate_dist"]
                else:
                    df.loc[i, "spot_estimate_dist"] = None
                df.loc[i, "spot_diameter"] = metadata["spot_diameter"]
                df.loc[i, "inter_spot_dist"] = metadata["inter_spot_dist"]
                
        
        df["magnification_x"] = df["magnification_x"].astype(float)
        df["magnification_y"] = df["magnification_y"].astype(float)
        df["pixel_size_um_estimated"] = df["pixel_size_um_estimated"].astype(float)
        df["magnification_in_metadata"] = df["magnification_in_metadata"].astype(float)

        assert np.all(df["magnification_x"] == df["magnification_y"])
        

        df.to_csv(join(save_dir, "meta_full.csv"), index=False)
    else:
        df = pd.read_csv(join(save_dir, "meta_full.csv"))

    #################################################################################
    # Step 2. visualize the magnification.
    #################################################################################
        
    # Assuming 'df' is your DataFrame and it already includes 'st_technology' and 'magnification_x'
    technologies = df['st_technology'].unique()  # This would ideally return something like ['Spatial Transcriptomics', 'Visium', 'Xenium', 'Visium HD']

    # Set up a 2x2 grid of subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
    axes = axes.flatten()  # Flatten the axes array for easier iteration

    # Plot histograms for each technology
    for ax, tech in zip(axes, technologies):
        # Filter the DataFrame for each technology and plot the 'magnification_x' distribution
        tech_data = df[df['st_technology'] == tech]['magnification_x']
        ax.hist(tech_data, bins=20, alpha=0.75, label=f'{tech} Magnification X')
        ax.set_title(f'{tech} Magnification X Distribution')
        ax.set_xlabel('Magnification X')
        ax.set_ylabel('Frequency')
        ax.legend()
    # Adjust layout to prevent overlap
    fig.tight_layout()

    fig.savefig(join(save_dir, "magnification_check.png"))
    

    #################################################################################
    # Step3. only interested in human & st+visium
    #################################################################################

    df_subset = df.loc[[v in ["Xenium"] for v in df["st_technology"]]]
    #df_subset = df_subset.loc[(df_subset["species"] == "Homo sapiens")]
    df_subset["organ"].value_counts()
    df_subset["disease_comment"].value_counts()
    
    df_subset.to_csv(join(save_dir, "meta_homo_sapiens_Xenium_ST.csv"), index=False)