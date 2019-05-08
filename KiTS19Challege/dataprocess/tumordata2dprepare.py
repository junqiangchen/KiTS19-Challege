from __future__ import print_function, division
import os
import cv2
import numpy as np


def prepare2ddata(srcpath, maskpath, trainImage, trainMask, number, height, width):
    for i in range(0, number, 1):
        index = 0
        listsrc = []
        listmask = []
        for _ in os.listdir(srcpath + str(i)):
            image = cv2.imread(srcpath + str(i) + "/" + str(index) + ".bmp", cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (width, height))
            label = cv2.imread(maskpath + str(i) + "/" + str(index) + ".bmp", cv2.IMREAD_GRAYSCALE)
            label = cv2.resize(label, (width, height))
            listsrc.append(image)
            listmask.append(label)
            index += 1

        imagearray = np.array(listsrc)
        imagearray = np.reshape(imagearray, (index, height, width))
        maskarray = np.array(listmask)
        maskarray = np.reshape(maskarray, (index, height, width))
        srcimg = np.clip(imagearray, 0, 255).astype('uint8')
        for j in range(index):
            if np.max(maskarray[j]) == 255:
                cv2.imwrite(trainImage + "\\" + str(i) + "_" + str(j) + ".bmp", srcimg[j])
                cv2.imwrite(trainMask + "\\" + str(i) + "_" + str(j) + ".bmp", maskarray[j])


def preparetumortrain2ddata():
    height = 512
    width = 512
    srcpath = "E:\junqiangchen\data\kits19\kits19tumorprocess\Image\\"
    maskpath = "E:\junqiangchen\data\kits19\kits19tumorprocess\Mask\\"
    trainImage = "E:\junqiangchen\data\kits19\kits19tumorsegmentation\Image"
    trainMask = "E:\junqiangchen\data\kits19\kits19tumorsegmentation\Mask"
    prepare2ddata(srcpath=srcpath, maskpath=maskpath, trainImage=trainImage, trainMask=trainMask, number=210,
                  height=height, width=width)


preparetumortrain2ddata()
