from __future__ import print_function, division
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from Vnet.model_vnet3d import Vnet3dModule
from dataprocess.utils import file_name_path
from dataprocess.finedata2dprepare import load_itkfilewithtrucation, resize_image_itk
from Vnet.util import removesmallConnectedCompont
import numpy as np
import cv2
import SimpleITK as sitk


def inference():
    """
    Vnet network segmentation kidney fine segmatation
    :return:
    """
    depth_z = 32
    Vnet3d = Vnet3dModule(512, 512, depth_z, channels=1, costname=("dice coefficient",), inference=True,
                          model_path="log\segmeation\VNet\model\Vnet3d.pd")
    kits_path = "D:\Data\kits19\kits19\\test"
    image_name = "imaging.nii.gz"
    result_path = "D:\Data\kits19\kits19test"

    """
    load itk image,change z Spacing value to 1,and save image ,liver mask ,tumor mask
    :return:None
    """
    # step2 get all train image
    path_list = file_name_path(kits_path)
    read = open("kidneyrang.txt", 'r')
    # step3 get signal train image and mask
    for subsetindex in range(len(path_list)):
        line = read.readline()
        line = line.split(',')
        casename = line[0]
        start = int(line[1])
        end = int(line[2][0:-1])
        kits_subset_path = kits_path + "/" + str(path_list[subsetindex]) + "/"
        file_image = kits_subset_path + image_name
        # 1 load itk image and truncate value with upper and lower and get rang kideny region
        src = load_itkfilewithtrucation(file_image, 300, -200)
        originSpacing = src.GetSpacing()
        src_array = sitk.GetArrayFromImage(src)
        sub_src_array = src_array[:, :, start:end]
        sub_src = sitk.GetImageFromArray(sub_src_array)
        sub_src.SetSpacing(originSpacing)
        print(sub_src.GetSize())
        thickspacing, widthspacing = originSpacing[0], originSpacing[1]
        # 2 change z spacing >1.0 to 1.0
        if thickspacing > 1.0:
            _, sub_src = resize_image_itk(sub_src, newSpacing=(1.0, widthspacing, widthspacing),
                                          originSpcaing=(thickspacing, widthspacing, widthspacing),
                                          resamplemethod=sitk.sitkLinear)
        xs_array = sitk.GetArrayFromImage(sub_src)
        xs_array = np.swapaxes(xs_array, 0, 2)
        index = np.shape(xs_array)[0]
        ys_pd_array = np.zeros(np.shape(xs_array), np.uint8)

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
        ys_pd_array = removesmallConnectedCompont(ys_pd_sitk, 0.2)
        ys_pd_array = np.clip(ys_pd_array, 0, 255).astype('uint8')
        sub_src_path = result_path + "/src/" + casename
        sub_pred_path = result_path + "/kidney_modify/" + casename
        if not os.path.exists(sub_src_path):
            os.makedirs(sub_src_path)
        if not os.path.exists(sub_pred_path):
            os.makedirs(sub_pred_path)
        for i in range(np.shape(xs_array)[0]):
            cv2.imwrite(sub_src_path + "/" + str(i) + ".bmp", xs_array[i])
            cv2.imwrite(sub_pred_path + "/" + str(i) + ".bmp", ys_pd_array[i])


inference()
