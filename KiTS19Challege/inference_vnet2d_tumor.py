from __future__ import print_function, division
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from Vnet2d.vnet_modelbase import BLVnet2dModule
from dataprocess.utils import file_name_path
import numpy as np
import cv2


def inference():
    height = 512
    width = 512
    Vnet2d = BLVnet2dModule(512, 512, channels=1, costname=("dice coefficient",), inference=True,
                            model_path="log\\tumor\diceVnet2d\model\BLVNet2ddiceloss.pd")
    tumor3d_path = "D:\Data\kits19\kits19test\kidney_modify"
    kits_path = "D:\Data\kits19\kits19test\src"
    tumor2d_path = "D:\Data\kits19\kits19test\\tumor2d"
    # step2 get all train image
    path_list = file_name_path(kits_path)
    # step3 get signal train image and mask
    for subsetindex in range(len(path_list)):
        kits_subset_path = kits_path + "/" + str(path_list[subsetindex]) + "/"
        kidney_mask_path = tumor3d_path + "/" + str(path_list[subsetindex]) + "/"
        tumor_mask_path = tumor2d_path + "/" + str(path_list[subsetindex]) + "/"
        index = 0
        imagelist = []
        masklist = []
        for _ in os.listdir(kits_subset_path):
            image = cv2.imread(kits_subset_path + "/" + str(index) + ".bmp", cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(kidney_mask_path + "/" + str(index) + ".bmp", cv2.IMREAD_GRAYSCALE)
            imagelist.append(image)
            masklist.append(mask)
            index += 1

        imagearray = np.array(imagelist)
        imagearray = np.reshape(imagearray, (index, height, width))
        maskarray = np.array(masklist)
        maskarray = np.reshape(maskarray, (index, height, width))
        imagemask = np.zeros((index, height, width), np.uint8)

        for z in range(index):
            if np.max(maskarray[z]) != 0:
                imagemask[z, :, :] = Vnet2d.prediction(imagearray[z])

        mask = imagemask.copy()
        mask[imagemask > 0] = 255
        result = np.clip(mask, 0, 255).astype('uint8')
        if not os.path.exists(tumor_mask_path):
            os.makedirs(tumor_mask_path)
        for j in range(index):
            cv2.imwrite(tumor_mask_path + str(j) + ".bmp", result[j])


inference()
