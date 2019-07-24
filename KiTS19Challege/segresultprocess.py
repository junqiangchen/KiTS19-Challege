from __future__ import print_function, division
import os
from dataprocess.utils import file_name_path
from dataprocess.finedata2dprepare import load_itkfilewithtrucation, resize_image_itk
from Vnet.util import removesmallConnectedCompont, morphologicaloperation
import numpy as np
import cv2
import SimpleITK as sitk


# step 1
def removekidneysmallobj():
    """
    去除Vnet肾脏结果的小物体
    :return:
    """
    kits_path = "D:\Data\kits19\kits19test\kidney_modify"
    result_path = "D:\Data\kits19\kits19test\\kidney_modify\\"
    """
    load itk image,change z Spacing value to 1,and save image ,liver mask ,tumor mask
    :return:None
    """
    # step2 get all train image
    path_list = file_name_path(kits_path)
    # step3 get signal train image and mask
    for subsetindex in range(len(path_list)):
        kits_subset_path = kits_path + "/" + str(path_list[subsetindex]) + "/"
        images = []
        index = 0
        for _ in os.listdir(kits_subset_path):
            image = cv2.imread(kits_subset_path + "/" + str(index) + ".bmp", cv2.IMREAD_GRAYSCALE)
            images.append(image)
            index += 1
        ys_pd_array = np.array(images)
        ys_pd_array = np.reshape(ys_pd_array, (index, 512, 512))
        ys_pd_sitk = sitk.GetImageFromArray(ys_pd_array)
        ys_pd_array = removesmallConnectedCompont(ys_pd_sitk, 0.2)
        ys_pd_array = np.clip(ys_pd_array, 0, 255).astype('uint8')
        sub_src_path = result_path + path_list[subsetindex]
        if not os.path.exists(sub_src_path):
            os.makedirs(sub_src_path)
        for i in range(np.shape(ys_pd_array)[0]):
            cv2.imwrite(sub_src_path + "/" + str(i) + ".bmp", ys_pd_array[i])


# step 2
def remove2d3dtumorsmallobj():
    """
    去除2d和3dVnet的肿瘤的小物体
    :return:
    """
    tumor3d_path = "D:\Data\kits19\kits19test\\tumor3d"
    tumor2d_path = "D:\Data\kits19\kits19test\\tumor2d"
    """
    load itk image,change z Spacing value to 1,and save image ,liver mask ,tumor mask
    :return:None
    """
    # step2 get all train image
    path_list = file_name_path(tumor3d_path)
    # step3 get signal train image and mask
    for subsetindex in range(len(path_list)):
        kits_subset_path = tumor3d_path + "/" + str(path_list[subsetindex]) + "/"
        tumor2d_subset_path = tumor2d_path + "/" + str(path_list[subsetindex]) + "/"
        images = []
        tumor2dmasks = []
        index = 0
        for _ in os.listdir(kits_subset_path):
            image = cv2.imread(kits_subset_path + "/" + str(index) + ".bmp", cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(tumor2d_subset_path + "/" + str(index) + ".bmp", cv2.IMREAD_GRAYSCALE)
            images.append(image)
            tumor2dmasks.append(mask)
            index += 1
        tumor3d_array = np.array(images)
        tumor3d_array = np.reshape(tumor3d_array, (index, 512, 512))
        tumor2d_array = np.array(tumor2dmasks)
        tumor2d_array = np.reshape(tumor2d_array, (index, 512, 512))
        tumor2d_array_sitk = sitk.GetImageFromArray(tumor2d_array)
        tumor2d_array = removesmallConnectedCompont(tumor2d_array_sitk, 0.2)
        tumor2d_array_sitk = sitk.GetImageFromArray(tumor3d_array)
        tumor3d_array = removesmallConnectedCompont(tumor2d_array_sitk, 0.2)
        tumor2d_array = np.clip(tumor2d_array, 0, 255).astype('uint8')
        tumor3d_array = np.clip(tumor3d_array, 0, 255).astype('uint8')
        sub_tumor3d_path = tumor3d_path + "/" + path_list[subsetindex]
        sub_tumor2d_path = tumor2d_path + "/" + path_list[subsetindex]
        if not os.path.exists(sub_tumor3d_path):
            os.makedirs(sub_tumor3d_path)
        if not os.path.exists(sub_tumor2d_path):
            os.makedirs(sub_tumor2d_path)
        for i in range(np.shape(tumor3d_array)[0]):
            cv2.imwrite(sub_tumor3d_path + "/" + str(i) + ".bmp", tumor3d_array[i])
            cv2.imwrite(sub_tumor2d_path + "/" + str(i) + ".bmp", tumor2d_array[i])


# step 3
def tumor2d3doverlap():
    """
    求2d和3dVnet的肿瘤区域的交集结果
    :return:
    """
    tumor3d_path = "D:\Data\kits19\kits19test\\tumor3d"
    tumor2d_path = "D:\Data\kits19\kits19test\\tumor2d"
    result_path = "D:\Data\kits19\kits19test\\tumor2_3d\\"
    """
    load itk image,change z Spacing value to 1,and save image ,liver mask ,tumor mask
    :return:None
    """
    # step2 get all train image
    path_list = file_name_path(tumor3d_path)
    # step3 get signal train image and mask
    for subsetindex in range(len(path_list)):
        kits_subset_path = tumor3d_path + "/" + str(path_list[subsetindex]) + "/"
        tumor2d_subset_path = tumor2d_path + "/" + str(path_list[subsetindex]) + "/"
        images = []
        tumor2dmasks = []
        index = 0
        for _ in os.listdir(kits_subset_path):
            image = cv2.imread(kits_subset_path + "/" + str(index) + ".bmp", cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(tumor2d_subset_path + "/" + str(index) + ".bmp", cv2.IMREAD_GRAYSCALE)
            images.append(image)
            tumor2dmasks.append(mask)
            index += 1
        tumor3d_array = np.array(images)
        tumor3d_array = np.reshape(tumor3d_array, (index, 512, 512))
        tumor2d_array = np.array(tumor2dmasks)
        tumor2d_array = np.reshape(tumor2d_array, (index, 512, 512))
        tumor_array = tumor3d_array & tumor2d_array
        tumor_array_sitk = sitk.GetImageFromArray(tumor_array)
        tumor_array = removesmallConnectedCompont(tumor_array_sitk, 0.2)
        tumor_array = np.clip(tumor_array, 0, 255).astype('uint8')
        sub_tumor_path = result_path + path_list[subsetindex]
        if not os.path.exists(sub_tumor_path):
            os.makedirs(sub_tumor_path)
        for i in range(np.shape(tumor_array)[0]):
            cv2.imwrite(sub_tumor_path + "/" + str(i) + ".bmp", tumor_array[i])


# step 4
def tumor2d3dmerge():
    """
    求2d和3d交集的区域分别与2d和3d相连接的区域都保留下来
    :return:
    """
    tumor3d_path = "D:\Data\kits19\kits19test\\tumor3d"
    tumor2d_path = "D:\Data\kits19\kits19test\\tumor2d"
    tumor2d3d_path = "D:\Data\kits19\kits19test\\tumor2_3d"
    result_path = "D:\Data\kits19\kits19test\\tumor_modify2\\"
    """
    load itk image,change z Spacing value to 1,and save image ,liver mask ,tumor mask
    :return:None
    """
    # step2 get all train image
    path_list = file_name_path(tumor3d_path)
    # step3 get signal train image and mask
    for subsetindex in range(len(path_list)):
        kits_subset_path = tumor3d_path + "/" + str(path_list[subsetindex]) + "/"
        tumor2d_subset_path = tumor2d_path + "/" + str(path_list[subsetindex]) + "/"
        tumor2d3d_subset_path = tumor2d3d_path + "/" + str(path_list[subsetindex]) + "/"
        images = []
        tumor2dmasks = []
        tumo2d3dmasks = []
        index = 0
        for _ in os.listdir(kits_subset_path):
            image = cv2.imread(kits_subset_path + "/" + str(index) + ".bmp", cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(tumor2d_subset_path + "/" + str(index) + ".bmp", cv2.IMREAD_GRAYSCALE)
            tumor = cv2.imread(tumor2d3d_subset_path + "/" + str(index) + ".bmp", cv2.IMREAD_GRAYSCALE)
            images.append(image)
            tumor2dmasks.append(mask)
            tumo2d3dmasks.append(tumor)
            index += 1
        tumor3d_array = np.array(images)
        tumor3d_array = np.reshape(tumor3d_array, (index, 512, 512))
        tumor2d_array = np.array(tumor2dmasks)
        tumor2d_array = np.reshape(tumor2d_array, (index, 512, 512))
        tumor2d3d_array = np.array(tumo2d3dmasks)
        tumor2d3d_array = np.reshape(tumor2d3d_array, (index, 512, 512))
        tumor_array = np.zeros((index, 512, 512), np.int)
        for z in range(index):
            tumor2d3d = tumor2d3d_array[z]
            if np.max(tumor2d3d) != 0:
                tumor2d3dlabels, tumor2d3dout = cv2.connectedComponents(tumor2d3d)
                tumor3d = tumor3d_array[z]
                tumor3dlabels, tumor3dout = cv2.connectedComponents(tumor3d)
                tumor2d = tumor2d_array[z]
                tumor2dlabels, tumor2dout = cv2.connectedComponents(tumor2d)

                for i in range(1, tumor2d3dlabels):
                    tumor2d3doutmask = np.zeros(tumor2d3dout.shape, np.int)
                    tumor2d3doutmask[tumor2d3dout == i] = 255
                    for j in range(1, tumor3dlabels):
                        tumor3doutmask = np.zeros(tumor3dout.shape, np.int)
                        tumor3doutmask[tumor3dout == j] = 255
                        if cv2.countNonZero(tumor2d3doutmask & tumor3doutmask):
                            tumor_array[z] = tumor_array[z] + tumor3doutmask
                    for k in range(1, tumor2dlabels):
                        tumor2doutmask = np.zeros(tumor2dout.shape, np.int)
                        tumor2doutmask[tumor2dout == k] = 255
                        if cv2.countNonZero(tumor2d3doutmask & tumor2doutmask):
                            tumor_array[z] = tumor_array[z] + tumor2doutmask

        tumor2d_array_sitk = sitk.GetImageFromArray(tumor_array)
        tumor2d_array = removesmallConnectedCompont(tumor2d_array_sitk, 0.2)
        tumor_array = np.clip(tumor2d_array, 0, 255).astype('uint8')
        sub_tumor_path = result_path + path_list[subsetindex]
        if not os.path.exists(sub_tumor_path):
            os.makedirs(sub_tumor_path)
        for i in range(np.shape(tumor_array)[0]):
            cv2.imwrite(sub_tumor_path + "/" + str(i) + ".bmp", tumor_array[i])


# step 5
def tumor2d3dinkidney():
    """
    保留肾脏区域范围内的肿瘤2d和3d结果
    :return:
    """
    tumor3d_path = "D:\Data\kits19\kits19test\\tumor3d"
    tumor2d_path = "D:\Data\kits19\kits19test\\tumor2d"
    kidney_path = "D:\Data\kits19\kits19test\kidney_modify"
    result_path = "D:\Data\kits19\kits19test\\tumor_modify1\\"
    """
    load itk image,change z Spacing value to 1,and save image ,liver mask ,tumor mask
    :return:None
    """
    # step2 get all train image
    path_list = file_name_path(tumor3d_path)
    # step3 get signal train image and mask
    for subsetindex in range(len(path_list)):
        kidney_subset_path = kidney_path + "/" + str(path_list[subsetindex]) + "/"
        kits_subset_path = tumor3d_path + "/" + str(path_list[subsetindex]) + "/"
        tumor2d_subset_path = tumor2d_path + "/" + str(path_list[subsetindex]) + "/"
        kidneys = []
        images = []
        tumor2dmasks = []
        index = 0
        for _ in os.listdir(kits_subset_path):
            kidney = cv2.imread(kidney_subset_path + "/" + str(index) + ".bmp", cv2.IMREAD_GRAYSCALE)
            image = cv2.imread(kits_subset_path + "/" + str(index) + ".bmp", cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(tumor2d_subset_path + "/" + str(index) + ".bmp", cv2.IMREAD_GRAYSCALE)
            kidneys.append(kidney)
            images.append(image)
            tumor2dmasks.append(mask)
            index += 1
        kidneys_array = np.array(kidneys)
        kidneys_array = np.reshape(kidneys_array, (index, 512, 512))
        kidneys_array_sitk = sitk.GetImageFromArray(kidneys_array)
        kidneys_array = morphologicaloperation(kidneys_array_sitk, 5, "dilate")

        tumor3d_array = np.array(images)
        tumor3d_array = np.reshape(tumor3d_array, (index, 512, 512))

        tumor2d_array = np.array(tumor2dmasks)
        tumor2d_array = np.reshape(tumor2d_array, (index, 512, 512))

        tumor3d_array = tumor3d_array & kidneys_array
        tumor2d_array = tumor2d_array & kidneys_array

        tumor_array = tumor3d_array | tumor2d_array
        tumor_array_sitk = sitk.GetImageFromArray(tumor_array)
        tumor_array = removesmallConnectedCompont(tumor_array_sitk, 0.1)
        tumor_array = np.clip(tumor_array, 0, 255).astype('uint8')
        sub_tumor_path = result_path + path_list[subsetindex]
        if not os.path.exists(sub_tumor_path):
            os.makedirs(sub_tumor_path)
        for i in range(np.shape(tumor_array)[0]):
            cv2.imwrite(sub_tumor_path + "/" + str(i) + ".bmp", tumor_array[i])


# step 6
def tumormodifyallmerge():
    """
    将肾脏区域内的2d和3d肿瘤区域并集结果与2d和3d肿瘤交集结果相加
    :return:
    """
    tumor2_path = "D:\Data\kits19\kits19test\\tumor_modify2"
    tumor1_path = "D:\Data\kits19\kits19test\\tumor_modify1"
    result_path = "D:\Data\kits19\kits19test\\tumor_modify"
    """
    load itk image,change z Spacing value to 1,and save image ,liver mask ,tumor mask
    :return:None
    """
    # step2 get all train image
    path_list = file_name_path(tumor2_path)
    # step3 get signal train image and mask
    for subsetindex in range(len(path_list)):
        kits_subset2_path = tumor2_path + "/" + str(path_list[subsetindex]) + "/"
        kits_subset1_path = tumor1_path + "/" + str(path_list[subsetindex]) + "/"
        images = []
        masks = []
        index = 0
        for _ in os.listdir(kits_subset1_path):
            image = cv2.imread(kits_subset1_path + "/" + str(index) + ".bmp", cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(kits_subset2_path + "/" + str(index) + ".bmp", cv2.IMREAD_GRAYSCALE)
            images.append(image)
            masks.append(mask)
            index += 1
        tumor3d_array = np.array(images)
        tumor3d_array = np.reshape(tumor3d_array, (index, 512, 512))
        tumor2d_array = np.array(masks)
        tumor2d_array = np.reshape(tumor2d_array, (index, 512, 512))
        tumor_array = tumor2d_array + tumor3d_array
        tumor_array = np.clip(tumor_array, 0, 255).astype('uint8')
        sub_tumor3d_path = result_path + "/" + path_list[subsetindex]
        if not os.path.exists(sub_tumor3d_path):
            os.makedirs(sub_tumor3d_path)
        for i in range(np.shape(tumor_array)[0]):
            cv2.imwrite(sub_tumor3d_path + "/" + str(i) + ".bmp", tumor_array[i])


# step 7
def outputresult():
    """
    将最后肾脏结果和肿瘤结果输出成比赛结果
    :return:
    """
    kits_path = "D:\Data\kits19\kits19\\test_data"
    kidney_path = "D:\Data\kits19\kits19test\kidney_modify"
    tumor_path = "D:\Data\kits19\kits19test\\tumor_modify"
    image_name = "imaging.nii.gz"
    result_path = "D:\Data\kits19\kits19"

    height, width = 512, 512
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

        kidney_mask_path = kidney_path + "/" + str(path_list[subsetindex]) + "/"
        tumor_mask_path = tumor_path + "/" + str(path_list[subsetindex]) + "/"
        index = 0
        kidneylist = []
        tumorlist = []
        for _ in os.listdir(kidney_mask_path):
            image = cv2.imread(kidney_mask_path + "/" + str(index) + ".bmp", cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(tumor_mask_path + "/" + str(index) + ".bmp", cv2.IMREAD_GRAYSCALE)
            kidneylist.append(image)
            tumorlist.append(mask)
            index += 1

        kidneyarray = np.array(kidneylist)
        kidneyarray = np.reshape(kidneyarray, (index, height, width))
        tumorarray = np.array(tumorlist)
        tumorarray = np.reshape(tumorarray, (index, height, width))
        outmask = np.zeros((index, height, width), np.uint8)
        outmask[kidneyarray == 255] = 1
        outmask[tumorarray == 255] = 2
        # 1 load itk image and truncate value with upper and lower and get rang kideny region
        src = load_itkfilewithtrucation(file_image, 300, -200)
        originSize = src.GetSize()
        originSpacing = src.GetSpacing()
        thickspacing, widthspacing = originSpacing[0], originSpacing[1]
        outmask = np.swapaxes(outmask, 0, 2)
        mask_sitk = sitk.GetImageFromArray(outmask)
        mask_sitk.SetSpacing((1.0, widthspacing, widthspacing))
        mask_sitk.SetOrigin(src.GetOrigin())
        mask_sitk.SetDirection(src.GetDirection())
        # 2 change z spacing >1.0 to originspacing
        if thickspacing > 1.0:
            _, mask_sitk = resize_image_itk(mask_sitk, newSpacing=(thickspacing, widthspacing, widthspacing),
                                            originSpcaing=(1.0, widthspacing, widthspacing),
                                            resamplemethod=sitk.sitkLinear)
        else:
            mask_sitk.SetSpacing(originSpacing)
        src_mask_array = np.zeros((originSize[0], height, width), np.uint8)
        mask_array = sitk.GetArrayFromImage(mask_sitk)
        mask_array = np.swapaxes(mask_array, 0, 2)
        # make sure the subregion have same size
        if (end - start) != np.shape(mask_array)[0]:
            start = start
            end = start + np.shape(mask_array)[0]
            if end > originSize[0]:
                end = originSize[0]
                start = end - np.shape(mask_array)[0]
        src_mask_array[start:end] = mask_array
        src_mask_array = np.swapaxes(src_mask_array, 0, 2)
        mask_itk = sitk.GetImageFromArray(src_mask_array)
        mask_itk.SetSpacing(originSpacing)
        mask_itk.SetOrigin(src.GetOrigin())
        mask_itk.SetDirection(src.GetDirection())
        mask_name = result_path + "/" + "prediction" + casename[4:10] + ".nii.gz"
        sitk.WriteImage(mask_itk, mask_name)


outputresult()
