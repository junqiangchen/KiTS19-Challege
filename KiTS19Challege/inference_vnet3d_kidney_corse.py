import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from Vnet.model_vnet3d import Vnet3dModule
from dataprocess.utils import file_name_path
from dataprocess.finedata2dprepare import getRangImageDepth
from dataprocess.corsedata2dprepare import load_itkfilewithtrucation, resize_image_itkwithsize
import SimpleITK as sitk
import numpy as np


def inference():
    """
        Vnet network segmentation kidney corse segmatation,get range of kidney
        Course segmentation,resize image to fixed size,segmentation the mask and get mask range
        :return:
        """
    depth_z = 64
    height = 256
    Vnet3d = Vnet3dModule(height, height, depth_z, channels=1, costname=("dice coefficient",), inference=True,
                          model_path="log\segmeation\CoarseVNet\model\Vnet3d.pd")
    fixed_size = [depth_z, height, height]
    kits_path = "D:\Data\kits19\kits19\\test"
    image_name = "imaging.nii.gz"

    """
    load itk image,change z Spacing value to 1,and save image ,liver mask ,tumor mask
    :return:None
    """
    # step2 get all train image
    path_list = file_name_path(kits_path)
    file_name = "kidneyrang.txt"
    out = open(file_name, 'w')
    # step3 get signal train image and mask
    for subsetindex in range(len(path_list)):
        kits_subset_path = kits_path + "/" + str(path_list[subsetindex]) + "/"
        file_image = kits_subset_path + image_name
        # 1 load itk image and truncate value with upper and lower
        src = load_itkfilewithtrucation(file_image, 300, -200)
        originSize = src.GetSize()
        originSpacing = src.GetSpacing()
        thickspacing, widthspacing = originSpacing[0], originSpacing[1]
        # 2 change image size to fixed size(512,512,64)
        _, src = resize_image_itkwithsize(src, newSize=fixed_size,
                                          originSize=originSize,
                                          originSpcaing=[thickspacing, widthspacing, widthspacing],
                                          resamplemethod=sitk.sitkLinear)
        # 3 get resample array(image and segmask)
        srcimg = sitk.GetArrayFromImage(src)
        srcimg = np.swapaxes(srcimg, 0, 2)
        ys_pd_array = Vnet3d.prediction(srcimg)
        ys_pd_array = np.clip(ys_pd_array, 0, 255).astype('uint8')

        ys_pd_array = np.swapaxes(ys_pd_array, 0, 2)
        ys_pd_itk = sitk.GetImageFromArray(ys_pd_array)
        ys_pd_itk.SetSpacing(src.GetSpacing())
        ys_pd_itk.SetOrigin(src.GetOrigin())
        ys_pd_itk.SetDirection(src.GetDirection())

        _, ys_pd_itk = resize_image_itkwithsize(ys_pd_itk, newSize=originSize,
                                                originSize=fixed_size,
                                                originSpcaing=[src.GetSpacing()[0], src.GetSpacing()[1],
                                                               src.GetSpacing()[2]],
                                                resamplemethod=sitk.sitkNearestNeighbor)

        pd_array = sitk.GetArrayFromImage(ys_pd_itk)
        print(np.shape(pd_array))

        # 4 get range of corse kidney
        expandslice = 5
        startpostion, endpostion = getRangImageDepth(pd_array)
        if startpostion == endpostion:
            print("corse error")
        imagez = np.shape(pd_array)[2]
        startpostion = startpostion - expandslice
        endpostion = endpostion + expandslice
        if startpostion < 0:
            startpostion = 0
        if endpostion > imagez:
            endpostion = imagez
        print("casenaem:", path_list[subsetindex])
        print("startposition:", startpostion)
        print("endpostion:", endpostion)
        out.writelines(path_list[subsetindex] + "," + str(startpostion) + "," + str(endpostion) + "\n")


inference()
