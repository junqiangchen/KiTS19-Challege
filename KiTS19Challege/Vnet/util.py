from __future__ import print_function, division
import SimpleITK as sitk
import numpy as np
import cv2
import os
from glob import glob


def load_itk(filename):
    """
    load mhd files and normalization 0-255
    :param filename:
    :return:
    """
    rescalFilt = sitk.RescaleIntensityImageFilter()
    rescalFilt.SetOutputMaximum(255)
    rescalFilt.SetOutputMinimum(0)
    # Reads the image using SimpleITK
    itkimage = rescalFilt.Execute(sitk.Cast(sitk.ReadImage(filename), sitk.sitkFloat32))
    return itkimage


def removesmallConnectedCompont(sitk_maskimg, rate=0.5):
    cc = sitk.ConnectedComponent(sitk_maskimg)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.Execute(cc, sitk_maskimg)
    maxlabel = 0
    maxsize = 0
    for l in stats.GetLabels():
        size = stats.GetPhysicalSize(l)
        if maxsize < size:
            maxlabel = l
            maxsize = size

    not_remove = []
    for l in stats.GetLabels():
        size = stats.GetPhysicalSize(l)
        if size > maxsize * rate:
            not_remove.append(l)
    labelmaskimage = sitk.GetArrayFromImage(cc)
    outmask = labelmaskimage.copy()
    outmask[labelmaskimage != maxlabel] = 0
    for i in range(len(not_remove)):
        outmask[labelmaskimage == not_remove[i]] = 255
    return outmask


def getLargestConnectedCompont(sitk_maskimg):
    cc = sitk.ConnectedComponent(sitk_maskimg)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.Execute(cc, sitk_maskimg)
    maxlabel = 0
    maxsize = 0
    for l in stats.GetLabels():
        size = stats.GetPhysicalSize(l)
        if maxsize < size:
            maxlabel = l
            maxsize = size

    labelmaskimage = sitk.GetArrayFromImage(cc)
    outmask = labelmaskimage.copy()
    outmask[labelmaskimage == maxlabel] = 255
    outmask[labelmaskimage != maxlabel] = 0
    return outmask


def morphologicaloperation(sitk_maskimg, kernelsize, name='open'):
    if name == 'open':
        morphoimage = sitk.BinaryMorphologicalOpening(sitk_maskimg, [kernelsize, kernelsize, kernelsize])
        labelmaskimage = sitk.GetArrayFromImage(morphoimage)
        outmask = labelmaskimage.copy()
        outmask[labelmaskimage == 1.0] = 255
        return outmask
    if name == 'close':
        morphoimage = sitk.BinaryMorphologicalClosing(sitk_maskimg, [kernelsize, kernelsize, kernelsize])
        labelmaskimage = sitk.GetArrayFromImage(morphoimage)
        outmask = labelmaskimage.copy()
        outmask[labelmaskimage == 1.0] = 255
        return outmask
    if name == 'dilate':
        morphoimage = sitk.BinaryDilate(sitk_maskimg, [kernelsize, kernelsize, kernelsize])
        labelmaskimage = sitk.GetArrayFromImage(morphoimage)
        outmask = labelmaskimage.copy()
        outmask[labelmaskimage == 1.0] = 255
        return outmask
    if name == 'erode':
        morphoimage = sitk.BinaryErode(sitk_maskimg, [kernelsize, kernelsize, kernelsize])
        labelmaskimage = sitk.GetArrayFromImage(morphoimage)
        outmask = labelmaskimage.copy()
        outmask[labelmaskimage == 1.0] = 255
        return outmask


def gettestiamge():
    src = load_itk("D:\Data\LIST\LITS-Challenge-Test-Data\\test-volume-" + str(51) + ".nii")
    srcimg = sitk.GetArrayFromImage(src)
    for i in range(np.shape(srcimg)[0]):
        image = srcimg[i]
        image = np.clip(image, 0, 255).astype('uint8')
        cv2.imwrite("D:\Data\LIST\LITS-Challenge-Test-Data\\" + str(51) + "\\" + str(i) + ".bmp", image)


def getmaxsizeimage():
    srcpath = "D:\Data\LIST\LITS-Challenge-Test-Data\\test-volume-" + str(38) + ".nii"
    maskpath = "D:\Data\LIST\\test\PredictMask\\38"

    src = load_itk(srcpath)
    srcimg = sitk.GetArrayFromImage(src)

    maskimage = np.empty(shape=np.shape(srcimg), dtype=np.uint8)
    index = 0
    for _ in os.listdir(maskpath):
        masktmp = cv2.imread(maskpath + "/" + str(index) + ".bmp", cv2.IMREAD_GRAYSCALE)
        maskimage[index, :, :] = masktmp
        index += 1

    sitk_maskimg = sitk.GetImageFromArray(maskimage)
    origin = np.array(src.GetOrigin())
    # Read the spacing along each dimension
    spacing = np.array(src.GetSpacing())
    sitk_maskimg.SetSpacing(spacing)
    sitk_maskimg.SetOrigin(origin)
    maskimage = getLargestConnectedCompont(sitk_maskimg)
    for i in range(np.shape(maskimage)[0]):
        image = maskimage[i]
        image = np.clip(image, 0, 255).astype('uint8')
        cv2.imwrite("D:\Data\LIST\\test\PredictMask\\38_1\\" + str(i) + ".bmp", image)


def save_npy2csv(path, name, labelnum=1):
    """
    this is for classify,save label+filepath into csv
    """
    out = open(name, 'w')
    file_list = glob(path + "*.npy")
    out.writelines("index" + "," + "filename" + "\n")
    for index in range(len(file_list)):
        out.writelines(str(labelnum) + "," + file_list[index] + "\n")
    

# gettestiamge()
# getmaxsizeimage()
# save_npy2csv("G:\Data\LIDC\LUNA16\classsification\\1_aug\\", "nodel_positive.csv", 1)
