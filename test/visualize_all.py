import cv2
import argparse
import glob
import numpy as np
import os, sys
sys.path.append("../")
from utils.pc_util import draw_point_cloud

parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument('--data_dir',default='../SRXYZresults', type=str)
parser.add_argument('--exp_name',default='Vis2',type=str)

args = parser.parse_args()

if __name__=="__main__":
    file_dir=glob.glob(os.path.join(args.data_dir,"*.xyz"))##visualize all xyz file
    number = 0
    pcd_list=[]
    for idxname in range(49):
        #
        # filename = file.split('/')[-1]
        # idxname = filename.split('_')[0]
        
        srname = str(idxname)+'_sr.xyz'
        lrname = str(idxname)+'_lr.xyz'
        hrname = str(idxname)+'_hr.xyz'
        srfilepath = os.path.join(args.data_dir,srname)
        lrfilepath = os.path.join(args.data_dir,lrname)
        hrfilepath = os.path.join(args.data_dir, hrname)
        number+=1
        pcd_list.append(lrfilepath)
        pcd_list.append(srfilepath)
        pcd_list.append(hrfilepath)

    image_save_dir=os.path.join("../vis_result2",args.exp_name)
    if os.path.exists(image_save_dir)==False:
        os.makedirs(image_save_dir)

    for file in pcd_list:
        file_name=file.split("/")[-1].split('.')[0]
        pcd=np.loadtxt(file)
        # img = draw_point_cloud(pcd, zrot=120 / 180.0 * np.pi, xrot=30 / 180.0 * np.pi, yrot=50 / 180.0 * np.pi,
        #                        diameter=5,normalize=True,canvasSize=1000,space=480)
        img = draw_point_cloud(pcd, zrot=90 / 180.0 * np.pi, xrot=90 / 180.0 * np.pi, yrot=0 / 180.0 * np.pi,
                               diameter=5,normalize=True,canvasSize=1000,space=480)
        # img = draw_point_cloud(pcd, zrot=90 / 180.0 * np.pi, xrot=90 / 180.0 * np.pi, yrot=0 / 180.0 * np.pi,
        #                        diameter=6, normalize=True, canvasSize=980)
        img=(img*255).astype(np.uint8)
        image_save_path=os.path.join(image_save_dir,file_name+".jpg")
        cv2.imwrite(image_save_path,img)
