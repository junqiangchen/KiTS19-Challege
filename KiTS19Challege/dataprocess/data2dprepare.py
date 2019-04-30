from __future__ import print_function, division
import os
import SimpleITK as sitk
import cv2
import numpy as np
from dataprocess.utils import file_name_path
from dataprocess.dataAnaly import getImageSpacing


def getRangImageDepth(image):
    """
    :param image:
    :return:rang of image depth
    """
    # startposition, endposition = np.where(image)[0][[0, -1]]
    fistflag = True
    startposition = 0
    endposition = 0
    for z in range(image.shape[0]):
        notzeroflag = np.max(image[z])
        if notzeroflag and fistflag:
            startposition = z
            fistflag = False
        if notzeroflag:
            endposition = z
    return startposition, endposition


def resize_image_itk(itkimage, newSpacing, originSpcaing, resamplemethod=sitk.sitkNearestNeighbor):
    """
    image resize withe sitk resampleImageFilter
    :param itkimage:
    :param newSpacing:such as [1,1,1]
    :param resamplemethod:
    :return:
    """
    newSpacing = np.array(newSpacing, float)
    # originSpcaing = itkimage.GetSpacing()
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()
    factor = newSpacing / originSpcaing
    newSize = originSize / factor
    newSize = newSize.astype(np.int)
    resampler.SetReferenceImage(itkimage)
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetSize(newSize.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)
    if resamplemethod == sitk.sitkNearestNeighbor:
        itkimgResampled = sitk.Threshold(itkimgResampled, 0, 1.0, 255)
    imgResampled = sitk.GetArrayFromImage(itkimgResampled)
    return imgResampled, itkimgResampled


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


def load_itkfilewithtrucation(filename, upper=200, lower=-200):
    """
    load mhd files,set truncted value range and normalization 0-255
    :param filename:
    :param upper:
    :param lower:
    :return:
    """
    # 1,tructed outside of liver value
    srcitkimage = sitk.Cast(sitk.ReadImage(filename), sitk.sitkFloat32)
    srcitkimagearray = sitk.GetArrayFromImage(srcitkimage)
    srcitkimagearray[srcitkimagearray > upper] = upper
    srcitkimagearray[srcitkimagearray < lower] = lower
    # 2,get tructed outside of liver value image
    sitktructedimage = sitk.GetImageFromArray(srcitkimagearray)
    origin = np.array(srcitkimage.GetOrigin())
    spacing = np.array(srcitkimage.GetSpacing())
    sitktructedimage.SetSpacing(spacing)
    sitktructedimage.SetOrigin(origin)
    # 3 normalization value to 0-255
    rescalFilt = sitk.RescaleIntensityImageFilter()
    rescalFilt.SetOutputMaximum(255)
    rescalFilt.SetOutputMinimum(0)
    itkimage = rescalFilt.Execute(sitk.Cast(sitktructedimage, sitk.sitkFloat32))
    return itkimage


def gen_subregion(srcimg, seg_image, trainimagefile, trainMaskfile, expandslice=5):
    # 5 get the roi range of mask,and expand number slices before and after,and get expand range roi image
    # for kinery
    startpostion, endpostion = getRangImageDepth(seg_image)
    if startpostion == endpostion:
        return
    imagez = np.shape(seg_image)[0]
    startpostion = startpostion - expandslice
    endpostion = endpostion + expandslice
    if startpostion < 0:
        startpostion = 0
    if endpostion > imagez:
        endpostion = imagez

    src_kineryimg = srcimg[startpostion:endpostion, :, :]
    seg_kineryimage = seg_image[startpostion:endpostion, :, :]
    # 6 write src, liver mask and tumor mask image
    for z in range(seg_kineryimage.shape[0]):
        src_kineryimg = np.clip(src_kineryimg, 0, 255).astype('uint8')
        cv2.imwrite(trainimagefile + "\\" + str(z) + ".bmp", src_kineryimg[z])
        cv2.imwrite(trainMaskfile + "\\" + str(z) + ".bmp", seg_kineryimage[z])


def proKitsdata():
    expandslice = 5
    kits_path = "E:\junqiangchen\data\kits19\kits19\data"
    image_name = "imaging.nii.gz"
    mask_name = "segmentation.nii.gz"

    case_id = 'case_id'
    width_spacing = 'captured_pixel_width'
    slice_spacing = 'captured_slice_thickness'

    proImage = "E:\junqiangchen\data\kits19\kits19process\Image\\"
    proMask = "E:\junqiangchen\data\kits19\kits19process\Mask\\"
    protumorImage = "E:\junqiangchen\data\kits19\kits19tumorprocess\Image\\"
    protumorMask = "E:\junqiangchen\data\kits19\kits19tumorprocess\Mask\\"
    """
    load itk image,change z Spacing value to 1,and save image ,liver mask ,tumor mask
    :return:None
    """
    seriesindex = 0
    # step2 get all train image
    path_list = file_name_path(kits_path)
    kits_Spacings = getImageSpacing()
    # step3 get signal train image and mask
    for subsetindex in range(0, 210, 1):
        kits_subset_path = kits_path + "/" + str(path_list[subsetindex]) + "/"
        #kits_subset_path = kits_path + "/" + "case_00160" + "/"
        file_image = kits_subset_path + image_name
        # 1 load itk image and truncate value with upper and lower
        src = load_itkfilewithtrucation(file_image, 300, -200)
        mask_path = kits_subset_path + mask_name
        seg = sitk.ReadImage(mask_path, sitk.sitkUInt8)
        for space_index in range(len(kits_Spacings)):
            if str(path_list[subsetindex]) == kits_Spacings[space_index][case_id]:
                widthspacing = kits_Spacings[space_index][width_spacing]
                thickspacing = kits_Spacings[space_index][slice_spacing]
                break
        # 2 change z spacing >1.0 to 1.0
        if thickspacing > 1.0:
            _, seg = resize_image_itk(seg, newSpacing=(1.0, widthspacing, widthspacing),
                                      originSpcaing=(thickspacing, widthspacing, widthspacing),
                                      resamplemethod=sitk.sitkNearestNeighbor)
            _, src = resize_image_itk(src, newSpacing=(1.0, widthspacing, widthspacing),
                                      originSpcaing=(thickspacing, widthspacing, widthspacing),
                                      resamplemethod=sitk.sitkLinear)
        # 3 get resample array(image and segmask)
        segimg = sitk.GetArrayFromImage(seg)
        print(segimg.shape)
        segimg = np.swapaxes(segimg, 0, 2)
        print(segimg.shape)
        srcimg = sitk.GetArrayFromImage(src)
        srcimg = np.swapaxes(srcimg, 0, 2)

        trainimagefile = proImage + str(seriesindex)
        trainMaskfile = proMask + str(seriesindex)
        if not os.path.exists(trainimagefile):
            os.makedirs(trainimagefile)
        if not os.path.exists(trainMaskfile):
            os.makedirs(trainMaskfile)

        tumorimagefile = protumorImage + str(seriesindex)
        tumorMaskfile = protumorMask + str(seriesindex)
        if not os.path.exists(tumorimagefile):
            os.makedirs(tumorimagefile)
        if not os.path.exists(tumorMaskfile):
            os.makedirs(tumorMaskfile)
        # 4 get mask
        seg_kineryimage = segimg.copy()
        seg_kineryimage[segimg > 0] = 255
        gen_subregion(srcimg, seg_kineryimage, trainimagefile, trainMaskfile, expandslice)
        seg_kinerytumorimage = segimg.copy()
        seg_kinerytumorimage[segimg == 2] = 255
        gen_subregion(srcimg, seg_kinerytumorimage, tumorimagefile, tumorMaskfile, expandslice)
        seriesindex += 1


proKitsdata()
