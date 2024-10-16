import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from os.path import join
from PIL import Image, ImageDraw
import openslide

def load_img_and_feature(result_dir, wsi_dir, folder, nuclei_id):
    nuclei_stat = np.load(join(result_dir, folder, "stardist_results", "nuclei_stat_f16.npy"))
    nuclei_stat_columns = pd.read_csv(join(result_dir, folder, "stardist_results", "nuclei_stat_columns.csv"), index_col=0).values.flatten()
    nuclei_stat_columns = np.array([col.replace("('", "").replace("')", "").replace("', '", "__") for col in nuclei_stat_columns])

    nuclei_stat = pd.DataFrame(nuclei_stat, columns=nuclei_stat_columns)
    contours = np.load(join(result_dir, folder, "stardist_results", "contours.npy"), allow_pickle=True)
    centroids = np.load(join(result_dir, folder, "stardist_results", "centroids.npy"), allow_pickle=True)
    wsi = openslide.OpenSlide(join(wsi_dir, f"{folder}.tif"))
    mpp = float(wsi.properties["openslide.mpp-x"])
    reference_mpp_1x = 10  # microns per pixel at 1x
    magnification = reference_mpp_1x / mpp
    # read nuclei img raw
    width = 100
    height = 100
    print(f"Folder: {folder}")
    print(f"Mag: {magnification:.4f}")
    this_contour = contours[nuclei_id]
    centroid_x, centroid_y = np.round(centroids[nuclei_id,]).astype(int)
    x1 = centroid_x - width//2
    y1 = centroid_y - height//2
    img = wsi.read_region((x1, y1), 0, size=(width, height))
    
    # Draw the contour on the image using PIL's ImageDraw
    draw = ImageDraw.Draw(img)
    adjusted_contour = [(point[0] - x1, point[1] - y1) for point in this_contour]
    draw.polygon(adjusted_contour, outline="green", width=1)
    
    feature_area = int(nuclei_stat.loc[nuclei_id, "Morphology__area"])
    feature_major_axis_length = nuclei_stat.loc[nuclei_id, "Morphology__major_axis_length"]
    img.save(join(result_dir, "..", "quick_verification", f"{folder}__mag={magnification:.2f}x__id={nuclei_id}__area={feature_area}__majoraxis={feature_major_axis_length}.png"))

if __name__ == "__main__":
    result_dir = "/project/zhihuanglab/yutong/HEST_project/results_zhi/processed_WSIs"
    wsi_dir = "/project/zhihuanglab/common/datasets/HEST-1K/hest_data/wsis"
    for folder in os.listdir(result_dir):
        if not os.path.exists(join(result_dir, folder, "stardist_results", "nuclei images")):
            continue
        print(folder)

    
    # load_img_and_feature(result_dir, wsi_dir, folder="MEND88", nuclei_id=20019)
    # load_img_and_feature(result_dir, wsi_dir, folder="MEND88", nuclei_id=20008)
    # load_img_and_feature(result_dir, wsi_dir, folder="MEND88", nuclei_id=20030)
    # load_img_and_feature(result_dir, wsi_dir, folder="INT2", nuclei_id=10047)
    # load_img_and_feature(result_dir, wsi_dir, folder="INT15", nuclei_id=75)
    # load_img_and_feature(result_dir, wsi_dir, folder="INT15", nuclei_id=86)
    # load_img_and_feature(result_dir, wsi_dir, folder="INT15", nuclei_id=98)
    #load_img_and_feature(result_dir, wsi_dir, folder="TENX123", nuclei_id=40063)
    load_img_and_feature(result_dir, wsi_dir, folder="TENX106", nuclei_id=40064)
    