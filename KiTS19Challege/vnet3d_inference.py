from __future__ import print_function, division
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from Vnet.model_vnet3d import Vnet3dModule
from Vnet.util import removesmallConnectedCompont
from dataprocess.utils import calcu_dice
import numpy as np
import cv2
import SimpleITK as sitk


def inference():
    depth_z = 48
    Vnet3d = Vnet3dModule(512, 512, depth_z, channels=1, costname=("dice coefficient",), inference=True,
                          model_path="log\segmeation\model\Vnet3d.pd-20000")
    test_path = "E:\junqiangchen\data\kits19\kits19process\\"
    image_path = "Image"
    mask_path = "Mask"
    dice_values = []
    for num in range(200, 210, 1):
        index = 0
        batch_xs = []
        batch_ys = []
        test_image_path = test_path + image_path + "/" + str(num)
        test_mask_path = test_path + mask_path + "/" + str(num)
        for _ in os.listdir(test_image_path):
            image = cv2.imread(test_image_path + "/" + str(index) + ".bmp", cv2.IMREAD_GRAYSCALE)
            label = cv2.imread(test_mask_path + "/" + str(index) + ".bmp", cv2.IMREAD_GRAYSCALE)
            batch_xs.append(image)
            batch_ys.append(label)
            index += 1
        xs_array = np.array(batch_xs)
        ys_array = np.array(batch_ys)
        xs_array = np.reshape(xs_array, (index, 512, 512))
        ys_array = np.reshape(ys_array, (index, 512, 512))
        ys_pd_array = np.empty((index, 512, 512), np.uint8)

        last_depth = 0
        for depth in range(0, index // depth_z, 1):
            patch_xs = xs_array[depth * depth_z:(depth + 1) * depth_z, :, :]
            pathc_pd = Vnet3d.prediction(patch_xs)
            ys_pd_array[depth * depth_z:(depth + 1) * depth_z, :, :] = pathc_pd
            last_depth = depth
        if index != depth_z * last_depth:
            patch_xs = xs_array[(index - depth_z):index, :, :]
            pathc_pd = Vnet3d.prediction(patch_xs)
            ys_pd_array[(index - depth_z):index, :, :] = pathc_pd
        ys_pd_sitk = sitk.GetImageFromArray(ys_pd_array)
        ys_pd_array = removesmallConnectedCompont(ys_pd_sitk, 0.4)
        ys_pd_array = np.clip(ys_pd_array, 0, 255).astype('uint8')
        dice_value = calcu_dice(ys_pd_array, ys_array)
        print("num,dice:", (num, dice_value))
        dice_values.append(dice_value)
        for depth in range(0, index, 1):
            cv2.imwrite(test_mask_path + "/" + str(depth) + "predict.bmp", ys_pd_array[depth])
    average = sum(dice_values) / len(dice_values)
    print("average dice:", average)


inference()
