import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from Vnet.model_vnet3d import Vnet3dModule
import numpy as np
from dataprocess.utils import file_name_path
import cv2


def inference():
    """
    Vnet network segmentation kidney fine segmatation
    :return:
    """
    depth_z = 32
    Vnet3d = Vnet3dModule(512, 512, depth_z, channels=1, costname=("dice coefficient",), inference=True,
                          model_path="log\\tumor\VNet\model\Vnet3d.pd")

    kits_path = "D:\Data\kits19\kits19test\src"
    kidney_path = "D:\Data\kits19\kits19test\kidney_modify"
    result_path = "D:\Data\kits19\kits19test\\tumor3d\\"
    """
    load itk image,change z Spacing value to 1,and save image ,liver mask ,tumor mask
    :return:None
    """
    # step2 get all train image
    path_list = file_name_path(kits_path)
    # step3 get signal train image and mask
    for subsetindex in range(len(path_list)):
        kits_subset_path = kits_path + "/" + str(path_list[subsetindex]) + "/"
        kidney_subset_path = kidney_path + "/" + str(path_list[subsetindex]) + "/"
        images = []
        kidneys = []
        index = 0
        for _ in os.listdir(kits_subset_path):
            image = cv2.imread(kits_subset_path + "/" + str(index) + ".bmp", cv2.IMREAD_GRAYSCALE)
            kidney = cv2.imread(kidney_subset_path + "/" + str(index) + ".bmp", cv2.IMREAD_GRAYSCALE)
            images.append(image)
            kidneys.append(kidney)
            index += 1
        images_array = np.array(images)
        images_array = np.reshape(images_array, (index, 512, 512))
        kidneys_array = np.array(kidneys)
        kidneys_array = np.reshape(kidneys_array, (index, 512, 512))
        startposition, endposition = np.where(kidneys_array)[0][[0, -1]]
        ys_pd_array = np.empty((index, 512, 512), np.uint8)

        last_depth = 0
        for depth in range(0, index // depth_z, 1):
            patch_xs = images_array[depth * depth_z:(depth + 1) * depth_z, :, :]
            pathc_pd = Vnet3d.prediction(patch_xs)
            ys_pd_array[depth * depth_z:(depth + 1) * depth_z, :, :] = pathc_pd
            last_depth = depth
        if index != depth_z * last_depth:
            patch_xs = images_array[(index - depth_z):index, :, :]
            pathc_pd = Vnet3d.prediction(patch_xs)
            ys_pd_array[(index - depth_z):index, :, :] = pathc_pd

        ys_pd_array = np.clip(ys_pd_array, 0, 255).astype('uint8')
        # get expand roi
        startposition = startposition - 5
        endposition = endposition + 5
        if startposition < 0:
            startposition = 0
        if endposition == index:
            endposition = index
        mask_array = np.zeros((index, 512, 512), np.uint8)
        mask_array[startposition:endposition] = ys_pd_array[startposition:endposition]
        sub_mask_path = result_path + path_list[subsetindex]
        if not os.path.exists(sub_mask_path):
            os.makedirs(sub_mask_path)
        for i in range(np.shape(ys_pd_array)[0]):
            cv2.imwrite(sub_mask_path + "/" + str(i) + ".bmp", mask_array[i])


inference()
