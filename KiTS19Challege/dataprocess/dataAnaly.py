from __future__ import print_function, division
import SimpleITK as sitk
import json
import numpy as np
from dataprocess.utils import file_name_path

kits_path = "D:\Data\KiTS\kits19_download\data"
image_name = "imaging.nii.gz"
mask_name = "segmentation.nii.gz"

kits_json = "kits.json"
case_id = 'case_id'
width_spacing = 'captured_pixel_width'
slice_spacing = 'captured_slice_thickness'


def getTrunctedThresholdValue():
    """
    remove outside of liver region value,and expand the tumor range when normalization 0 to 1.
    calculate the overlap between liver mask and src image with range of lower and upper value.
    :return:None
    """
    # step1 set threshold value
    upper = 300
    lower = -200
    num_points = 0.0
    num_inliers = 0.0
    # step2 get all train image
    path_list = file_name_path(kits_path)
    # step3 get signal train image and mask
    for subsetindex in range(len(path_list)):
        kits_subset_path = kits_path + "/" + str(path_list[subsetindex]) + "/"
        file_image = kits_subset_path + image_name
        src = sitk.ReadImage(file_image, sitk.sitkInt16)
        srcimg = sitk.GetArrayFromImage(src)
        mask_path = kits_subset_path + mask_name
        seg = sitk.ReadImage(mask_path, sitk.sitkUInt8)
        segimg = sitk.GetArrayFromImage(seg)
        seg_maskimage = segimg.copy()
        seg_maskimage[segimg > 0] = 255

        inliers = 0
        num_point = 0
        for z in range(seg_maskimage.shape[0]):
            for y in range(seg_maskimage.shape[1]):
                for x in range(seg_maskimage.shape[2]):
                    if seg_maskimage[z][y][x] != 0:
                        num_point += 1
                        if (srcimg[z][y][x] < upper) and (srcimg[z][y][x] > lower):
                            inliers += 1
        # if not seg mask,not calculate
        if num_point != 0:
            print('{:.4}%'.format(inliers / num_point * 100))
            num_points += num_point
            num_inliers += inliers
    print("all percent:", num_inliers / num_points)


def getImageSpacing():
    """
    get image spacing from json file
    :return:
    """
    with open(kits_path + "/" + kits_json, 'r') as f:
        ImageSpacings = json.load(f)
        return ImageSpacings


def getImageSizeandSpacing():
    """
    get image and spacing
    :return:
    """
    kits_Spacings = getImageSpacing()
    path_list = file_name_path(kits_path)
    for subsetindex in range(len(path_list)):
        kits_subset_path = kits_path + "/" + str(path_list[subsetindex]) + "/"
        file_image = kits_subset_path + image_name
        src = sitk.ReadImage(file_image, sitk.sitkInt16)
        imageSize = src.GetSize()
        for index in range(len(kits_Spacings)):
            if str(path_list[subsetindex]) == kits_Spacings[index][case_id]:
                widthspacing = kits_Spacings[index][width_spacing]
                thickspacing = kits_Spacings[index][slice_spacing]
                print("image size,widthspcing,thickspacing:", (imageSize, widthspacing, thickspacing))


def getMaskLabelMaxValue():
    """
    get max mask value
    kits mask have three value:0,1,2(0 is backgroud ,1 is kidney,2 is tumor)
    :return:
    """
    path_list = file_name_path(kits_path)
    for subsetindex in range(len(path_list)):
        kits_subset_path = kits_path + "/" + str(path_list[subsetindex]) + "/"
        mask_path = kits_subset_path + mask_name
        seg = sitk.ReadImage(mask_path, sitk.sitkUInt8)
        segimg = sitk.GetArrayFromImage(seg)
        seg_maskimage = segimg.copy()
        max_value = np.max(seg_maskimage)
        print("max_mask_value:", max_value)


#getMaskLabelMaxValue()
#getImageSizeandSpacing()
#getTrunctedThresholdValue()