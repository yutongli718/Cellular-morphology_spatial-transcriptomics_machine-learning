from os.path import join
from tqdm import tqdm
import os
import shutil
import pandas as pd
import argparse
# module load openslide/4.0.0.3
# conda install -c bioconda openslide-python
# pip install openslide-python

# export LD_LIBRARY_PATH=/home/zhihuang/anaconda3/envs/HEST/lib:$LD_LIBRARY_PATH
def split_list(input_list, n):
    # Calculate the size of each sublist
    k, m = divmod(len(input_list), n)
    return [input_list[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='/project/zhihuanglab/common/datasets/HEST-1K/hest_data/', type=str)
    parser.add_argument('--work_dir', default="/project/zhihuanglab/yutong/HEST_project/", type=str)
    # parser.add_argument('--case_id', default="INT4", type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cell_vit_contour_processed_dir = "/project/zhihuanglab/yutong/HEST_project/results_zhi/processed_WSIs"

    df_meta = pd.read_csv(os.path.join(args.work_dir, 'results_zhi', 'meta_homo_sapiens_visium_ST.csv'))

    # List to hold all the bash commands
    all_bash_commands = []
    number_of_processed_cases = 0
    for i in tqdm(df_meta.index):
        case_id = df_meta.loc[i, "filename"].split(".tif")[0]
        if not os.path.exists(join(cell_vit_contour_processed_dir, case_id, "stardist_results_by_stardist", "contours.npy")):
            # Generate bash command
            bash_command = f"/home/zhihuang/anaconda3/envs/HEST/bin/python /project/zhihuanglab/yutong/HEST_project/scripts_zhi/feature_pre-calculation/main.py --slidepath '/project/zhihuanglab/common/datasets/HEST-1K/hest_data/wsis/{case_id}.tif' --stardist_dir '/project/zhihuanglab/yutong/HEST_project/results_zhi/processed_WSIs/{case_id}/stardist_results_by_stardist' --stage segmentation"
            # Add the command to the list
            all_bash_commands.append(bash_command)
            print(case_id, " not ready.")
        else:
            number_of_processed_cases += 1

    print(f"Number of processed cases: {number_of_processed_cases}/{len(df_meta)}")
    # exit()
    
    # Number of bash
    n_process = 1
    sublists = split_list(all_bash_commands, n=n_process)
    
    # Path to save the bash script
    bash_savedir = "/project/zhihuanglab/yutong/HEST_project/scripts_zhi/bash_p3_segmentation"
    # Clear all files inside the folder
    if os.path.exists(bash_savedir):
        for root, dirs, files in os.walk(bash_savedir):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                shutil.rmtree(os.path.join(root, dir))
    os.makedirs(bash_savedir, exist_ok=True)
    os.makedirs(join(bash_savedir, "logs"), exist_ok=True)

    for i, sublist in enumerate(sublists):
        # Create bsub command
        bash_content = f"""#!/bin/bash
#BSUB -J seg_{i:02d}
#BSUB -o {bash_savedir}/logs/seg_{i:02d}.out
#BSUB -e {bash_savedir}/logs/seg_{i:02d}.err
#BSUB -q gpu1
#BSUB -R "rusage[mem=16000]"

#load openslide
export PYTHONUNBUFFERED=1 # print to output immediately
"""
        # mem=16384 is 16GB of memory
        bash_content += "\n".join(sublist)
        # Write the bash script to a file
        bash_file = join(bash_savedir, f"seg_{i:02d}.sh")
        with open(bash_file, 'w') as f:
            f.write(bash_content)

    # Write a master bash script to call all bash files
    master_bash_file = join(bash_savedir, "_run_all.sh")
    with open(master_bash_file, 'w') as f:
        f.write("#!/bin/bash\n\n")
        for i in range(n_process):
            bash_file = join(bash_savedir, f"seg_{i:02d}.sh")
            f.write(f"bsub < {bash_file}\n")

    # Make the master bash script executable
    os.chmod(master_bash_file, 0o755)