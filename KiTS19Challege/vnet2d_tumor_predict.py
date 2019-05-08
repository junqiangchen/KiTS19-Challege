from __future__ import print_function, division
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from Vnet2d.vnet_modelbase import BLVnet2dModule
from dataprocess.utils import calcu_dice
import pandas as pd
import numpy as np
import cv2


def predict():
    csvdata = pd.read_csv('dataprocess\\data/testtumor.csv')
    maskdata = csvdata.iloc[:, 1].values
    imagedata = csvdata.iloc[:, 0].values

    dice_values = []
    Vnet2d = BLVnet2dModule(512, 512, channels=1, costname=("dice coefficient",), inference=True,
                            model_path="log\\tumorseg\diceVnet2d\model\BLVNet2ddiceloss.pd")
    for index in range(imagedata.shape[0]):
        mask_gt = cv2.imread(maskdata[index], cv2.IMREAD_GRAYSCALE)
        image_gt = cv2.imread(imagedata[index], cv2.IMREAD_GRAYSCALE)
        mask_pd = Vnet2d.prediction(image_gt)
        dice_value = calcu_dice(mask_pd, mask_gt)
        print("index,dice:", (index, dice_value))
        dice_values.append(dice_value)
    average = sum(dice_values) / len(dice_values)
    print("average dice:", average)


predict()