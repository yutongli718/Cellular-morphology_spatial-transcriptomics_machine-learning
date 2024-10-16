import json
import numpy as np
import pandas as pd
from shapely.geometry import shape, MultiPolygon
import os
import argparse
from tqdm import tqdm
from os.path import join

def extract_centroids_and_contours(dataset_dir, work_dir, case_id):
    # Construct the file path using dataset_dir and case_id
    file_path = os.path.join(dataset_dir, f'cellvit_seg/{case_id}_cellvit_seg/media/ssd2/hest/preprocessing/{case_id}/{case_id}/cell_detection/cells.geojson')
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    # Load GeoJSON file
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    centroids = []
    contours = []

    for feature in data:
        geometry = shape(feature['geometry'])
        
        if isinstance(geometry, MultiPolygon):
            for polygon in geometry.geoms:  # Correct iteration over MultiPolygon
                centroid = polygon.centroid
                centroids.append([centroid.x, centroid.y])
                contours.append(np.array(polygon.exterior.coords))
        else:
            centroid = geometry.centroid
            centroids.append([centroid.x, centroid.y])
            contours.append(np.array(geometry.exterior.coords))

    centroids = np.array(centroids)
    contours = np.array(contours, dtype=object)  # dtype=object to allow arrays of different lengths

    # Define output file paths using work_dir and case_id
    output_dir = os.path.join(work_dir, f'results_zhi/processed_WSIs/{case_id}/stardist_results')
    
    # Ensure the output directory exists, create if not
    os.makedirs(output_dir, exist_ok=True)

    # Save to .npy files
    centroids_output_path = os.path.join(output_dir, 'centroids.npy')
    contours_output_path = os.path.join(output_dir, 'contours.npy')

    np.save(centroids_output_path, centroids)
    np.save(contours_output_path, contours)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='/project/zhihuanglab/common/datasets/HEST-1K/hest_data/', type=str)
    parser.add_argument('--work_dir', default="/project/zhihuanglab/yutong/HEST_project/", type=str)
    parser.add_argument('--split', default=0, type=int)
    # parser.add_argument('--case_id', default="INT4", type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    df_meta = pd.read_csv(os.path.join(args.work_dir, 'results_zhi', 'meta_homo_sapiens_Xenium_ST.csv'))
    for i in tqdm(df_meta.index):
        # if i % 10 != args.split: continue
        case_id = df_meta.loc[i, "filename"].split(".tif")[0]
        extract_centroids_and_contours(args.dataset_dir, args.work_dir, case_id)






