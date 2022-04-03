import cv2
import argparse
import glob
import numpy as np
import os, sys

sys.path.append("../")
from utils.pc_util import draw_point_cloud

parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument('--data_dir', default='../SRXYZresults', type=str)
parser.add_argument('--exp_name', default='Vis2', type=str)

args = parser.parse_args()

if __name__ == "__main__":
    file_dir = os.listdir('/home/lizhuangzi/Desktop/SPU/SRXYZresults/')
    number = 0
    pcd_list = []
    for idx in range(len(file_dir)):
        file = file_dir[idx]
        fullpath = os.path.join(args.data_dir,file)

        filename = file.split('.')[0]
        # idxname = filename.split('_')[0]

        image_save_dir = os.path.join("../vis_result2", args.exp_name)

        pcd = np.loadtxt(fullpath)
        # img = draw_point_cloud(pcd, zrot=120 / 180.0 * np.pi, xrot=30 / 180.0 * np.pi, yrot=50 / 180.0 * np.pi,
        #                        diameter=5,normalize=True,canvasSize=1000,space=480)
        img = draw_point_cloud(pcd, zrot=90 / 180.0 * np.pi, xrot=90 / 180.0 * np.pi, yrot=0 / 180.0 * np.pi,
                               diameter=5, normalize=True, canvasSize=1000, space=480)
        # img = draw_point_cloud(pcd, zrot=90 / 180.0 * np.pi, xrot=90 / 180.0 * np.pi, yrot=0 / 180.0 * np.pi,
        #                        diameter=6, normalize=True, canvasSize=980)
        img = (img * 255).astype(np.uint8)
        image_save_path = os.path.join(image_save_dir, filename + ".jpg")
        cv2.imwrite(image_save_path, img)



