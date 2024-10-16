from os.path import join
from tqdm import tqdm
import os
import shutil
import pandas as pd
import numpy as np
import argparse
import openslide
from shapely.geometry import Polygon
from shapely.geometry import LineString
from PIL import Image, ImageDraw, ImageFont
import shutil

def major_axis_length(polygon):
    mbr = polygon.minimum_rotated_rectangle
    # Get the four points of the MBR
    mbr_points = list(mbr.exterior.coords)
    # Calculate distances between consecutive points (there are 4 sides)
    length_1 = LineString([mbr_points[0], mbr_points[1]]).length
    length_2 = LineString([mbr_points[1], mbr_points[2]]).length
    # The major axis length is the maximum of these lengths
    return max(length_1, length_2)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='/project/zhihuanglab/common/datasets/HEST-1K/hest_data/', type=str)
    parser.add_argument('--work_dir', default="/project/zhihuanglab/yutong/HEST_project/", type=str)
    # parser.add_argument('--case_id', default="INT4", type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cell_vit_contour_processed_dir = "/project/zhihuanglab/yutong/HEST_project/results_zhi/processed_WSIs"

    path2dfstat = join(args.work_dir, "results_zhi", "nuclei_stat_comparison_cellViT_vs_stardist.csv")
    if False:#os.path.exists(path2dfstat):
        df_stat = pd.read_csv(path2dfstat, index_col=0)
    else:
        df_stat = pd.DataFrame()
        for case_id in tqdm(os.listdir(cell_vit_contour_processed_dir)):
            slide = openslide.OpenSlide(join(args.dataset_dir, "wsis", f"{case_id}.tif"))
            cell_vit_results = np.load(join(cell_vit_contour_processed_dir, case_id, "stardist_results", "contours.npy"), allow_pickle=True)
            stardist_results = np.load(join(cell_vit_contour_processed_dir, case_id, "stardist_results_by_stardist", "contours.npy"), allow_pickle=True)

            reference_mpp_1x = 10  # microns per pixel at 1x
            df_stat.loc[case_id, "WSI_mpp"] = float(slide.properties["openslide.mpp-x"])
            df_stat.loc[case_id, "WSI_magnification"] = reference_mpp_1x / df_stat.loc[case_id, "WSI_mpp"]

            # Scale contours in standard 40x magnification
            resize_factor = 40 / df_stat.loc[case_id, "WSI_magnification"]
            cell_vit_results = cell_vit_results * resize_factor
            stardist_results = stardist_results * resize_factor

            df_stat.loc[case_id, "n_nuclei_cellViT"] = len(cell_vit_results)
            df_stat.loc[case_id, "n_nuclei_stardist"] = len(stardist_results)

            # area of cells
            list_of_areas_cellViT = [Polygon(points).area for points in cell_vit_results]
            list_of_areas_stardist = [Polygon(stardist_results[i, ...]).area for i in range(len(stardist_results))]

            df_stat.loc[case_id, "area_mean_cellViT"] = np.mean(list_of_areas_cellViT)
            df_stat.loc[case_id, "area_mean_stardist"] = np.mean(list_of_areas_stardist)

            df_stat.loc[case_id, "area_max_cellViT"] = np.max(list_of_areas_cellViT)
            df_stat.loc[case_id, "area_max_stardist"] = np.max(list_of_areas_stardist)

            df_stat.loc[case_id, "area_min_cellViT"] = np.min(list_of_areas_cellViT)
            df_stat.loc[case_id, "area_min_stardist"] = np.min(list_of_areas_stardist)

            # Calculating quantiles
            df_stat.loc[case_id, "area_99%_quantile_cellViT"] = np.quantile(list_of_areas_cellViT, 0.99)
            df_stat.loc[case_id, "area_99%_quantile_stardist"] = np.quantile(list_of_areas_stardist, 0.99)

            df_stat.loc[case_id, "area_95%_quantile_cellViT"] = np.quantile(list_of_areas_cellViT, 0.95)
            df_stat.loc[case_id, "area_95%_quantile_stardist"] = np.quantile(list_of_areas_stardist, 0.95)

            df_stat.loc[case_id, "area_50%_quantile_cellViT"] = np.quantile(list_of_areas_cellViT, 0.50)
            df_stat.loc[case_id, "area_50%_quantile_stardist"] = np.quantile(list_of_areas_stardist, 0.50)

            df_stat.loc[case_id, "area_10%_quantile_cellViT"] = np.quantile(list_of_areas_cellViT, 0.10)
            df_stat.loc[case_id, "area_10%_quantile_stardist"] = np.quantile(list_of_areas_stardist, 0.10)

            df_stat.loc[case_id, "area_5%_quantile_cellViT"] = np.quantile(list_of_areas_cellViT, 0.05)
            df_stat.loc[case_id, "area_5%_quantile_stardist"] = np.quantile(list_of_areas_stardist, 0.05)

            df_stat.loc[case_id, "area_1%_quantile_cellViT"] = np.quantile(list_of_areas_cellViT, 0.01)
            df_stat.loc[case_id, "area_1%_quantile_stardist"] = np.quantile(list_of_areas_stardist, 0.01)
            
            
            # major axis length
            # list_of_major_axes_cellViT = [major_axis_length(Polygon(points)) for points in cell_vit_results]
            # list_of_major_axes_stardist = [major_axis_length(Polygon(stardist_results[i, ...])) for i in range(len(stardist_results))]

            # df_stat.loc[case_id, "major_axis_mean_cellViT"] = np.mean(list_of_major_axes_cellViT)
            # df_stat.loc[case_id, "major_axis_mean_stardist"] = np.mean(list_of_major_axes_stardist)

            # df_stat.loc[case_id, "major_axis_max_cellViT"] = np.max(list_of_major_axes_cellViT)
            # df_stat.loc[case_id, "major_axis_max_stardist"] = np.max(list_of_major_axes_stardist)

            # df_stat.loc[case_id, "major_axis_min_cellViT"] = np.min(list_of_major_axes_cellViT)
            # df_stat.loc[case_id, "major_axis_min_stardist"] = np.min(list_of_major_axes_stardist)

            # # Calculating quantiles for major axis length
            # df_stat.loc[case_id, "major_axis_99%_quantile_cellViT"] = np.quantile(list_of_major_axes_cellViT, 0.99)
            # df_stat.loc[case_id, "major_axis_99%_quantile_stardist"] = np.quantile(list_of_major_axes_stardist, 0.99)

            # df_stat.loc[case_id, "major_axis_95%_quantile_cellViT"] = np.quantile(list_of_major_axes_cellViT, 0.95)
            # df_stat.loc[case_id, "major_axis_95%_quantile_stardist"] = np.quantile(list_of_major_axes_stardist, 0.95)

            # df_stat.loc[case_id, "major_axis_50%_quantile_cellViT"] = np.quantile(list_of_major_axes_cellViT, 0.50)
            # df_stat.loc[case_id, "major_axis_50%_quantile_stardist"] = np.quantile(list_of_major_axes_stardist, 0.50)

            # df_stat.loc[case_id, "major_axis_10%_quantile_cellViT"] = np.quantile(list_of_major_axes_cellViT, 0.10)
            # df_stat.loc[case_id, "major_axis_10%_quantile_stardist"] = np.quantile(list_of_major_axes_stardist, 0.10)

            # df_stat.loc[case_id, "major_axis_5%_quantile_cellViT"] = np.quantile(list_of_major_axes_cellViT, 0.05)
            # df_stat.loc[case_id, "major_axis_5%_quantile_stardist"] = np.quantile(list_of_major_axes_stardist, 0.05)

            # df_stat.loc[case_id, "major_axis_1%_quantile_cellViT"] = np.quantile(list_of_major_axes_cellViT, 0.01)
            # df_stat.loc[case_id, "major_axis_1%_quantile_stardist"] = np.quantile(list_of_major_axes_stardist, 0.01)

        df_stat.to_csv(path2dfstat)

    for case_id in tqdm(df_stat.index):
        # now compare the difference.
        # check difference
        # argidx = df_stat["area_5%_quantile_cellViT"].argmin()
        # case_id = "MISC12" # df_stat.index[argidx]
        slide = openslide.OpenSlide(join(args.dataset_dir, "wsis", f"{case_id}.tif"))
        cell_vit_centroids = np.load(join(cell_vit_contour_processed_dir, case_id, "stardist_results", "centroids.npy"), allow_pickle=True)
        stardist_centroids = np.load(join(cell_vit_contour_processed_dir, case_id, "stardist_results_by_stardist", "centroids.npy"), allow_pickle=True)
        cell_vit_results = np.load(join(cell_vit_contour_processed_dir, case_id, "stardist_results", "contours.npy"), allow_pickle=True)
        stardist_results = np.load(join(cell_vit_contour_processed_dir, case_id, "stardist_results_by_stardist", "contours.npy"), allow_pickle=True)
        stardist_probability = np.load(join(cell_vit_contour_processed_dir, case_id, "stardist_results_by_stardist", "probability.npy"), allow_pickle=True)
        stardist_results = stardist_results.astype(np.int32)
        
        # stardist_centroids = stardist_centroids[stardist_probability > 0.4, ...]
        # stardist_results = stardist_results[stardist_probability > 0.4, ...]
        
        # # we need to scale the centroids to the original size
        # reference_mpp_1x = 10  # microns per pixel at 1x
        # WSI_mpp = float(slide.properties["openslide.mpp-x"])
        # WSI_magnification = reference_mpp_1x / WSI_mpp
        # reference_mag = 20
        # resize_factor = reference_mag / WSI_magnification
        # stardist_centroids = stardist_centroids / resize_factor
        # stardist_results = stardist_results / resize_factor
        
        x1, y1 = 3072, 3072
        x2, y2 = 4000, 4000
        
        x1, y1 = 0, 0
        x2, y2 = slide.level_dimensions[0][0], slide.level_dimensions[0][1]
        img = slide.read_region((x1, y1), 0, (x2 - x1, y2 - y1))
        # breakpoint()
        # use PIL to draw the contour using cell_vit_results and stardist_results (2 images, side by side for comparison)

        # Function to draw contours
        def draw_contours(image, contours, color):
            draw = ImageDraw.Draw(image)
            for i in range(len(contours)):
                contour = contours[i]
                points = [tuple(point) for point in contour]
                draw.line(points, fill=color, width=2)
            return image

        # Create a copy of the image for each set of contours
        img_cell_vit = img.copy()
        img_stardist = img.copy()

        # breakpoint()

        matched_idx_cellvit = (cell_vit_centroids[:,0]>=x1) & (cell_vit_centroids[:,0]<=x2) & (cell_vit_centroids[:,1]>=y1) & (cell_vit_centroids[:,1]<=y2)
        matched_idx_stardist = (stardist_centroids[:,0]>=x1) & (stardist_centroids[:,0]<=x2) & (stardist_centroids[:,1]>=y1) & (stardist_centroids[:,1]<=y2)
        
        # Extract the contours for the matched ROIs
        cellvit_contour_roi = cell_vit_results[matched_idx_cellvit]
        stardist_contour_roi = stardist_results[matched_idx_stardist]

        # Adjust the contours to the local coordinates
        for i in range(len(cellvit_contour_roi)):
            cellvit_contour_roi[i] = cellvit_contour_roi[i] - np.array([x1, y1])

        for i in range(len(stardist_contour_roi)):
            stardist_contour_roi[i] = stardist_contour_roi[i] - np.array([x1, y1])

        # Draw contours on the respective images
        img_cell_vit = draw_contours(img_cell_vit, cellvit_contour_roi, color='red')
        img_stardist = draw_contours(img_stardist, stardist_contour_roi, color='blue')

        # Combine the two images side by side
        combined_img = Image.new('RGB', (img_cell_vit.width * 2, img_cell_vit.height))
        combined_img.paste(img_cell_vit, (0, 0))
        combined_img.paste(img_stardist, (img_cell_vit.width, 0))

        # resize to 4000 pixel
        size = combined_img.size
        combined_img = combined_img.resize((12000, int(size[1] * 12000 / size[0])))

        # Add titles
        draw = ImageDraw.Draw(combined_img)
        font = ImageFont.load_default()

        title_text = f"CellViT ({len(cellvit_contour_roi)} nuclei)     Stardist ({len(stardist_contour_roi)} nuclei)"
        title_bbox = draw.textbbox((0, 0), title_text, font=font)

        # Calculate the width and height of the text from the bounding box
        title_w = title_bbox[2] - title_bbox[0]
        title_h = title_bbox[3] - title_bbox[1]

        title_position = (combined_img.width // 2 - title_w // 2, 10)
        draw.text(title_position, title_text, fill="white", font=font)

        combined_img.save(join(args.work_dir, "results_zhi", f"comparison_cellViT_vs_stardist_case={case_id}.png"))
        # breakpoint()