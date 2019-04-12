import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from Vnet.model_vnet3d import Vnet3dModule
import numpy as np
import pandas as pd


def train():
    '''
    Preprocessing for dataset
    '''
    # Read  data set (Train data from CSV file)
    csvdata = pd.read_csv('dataprocess\\data/train.csv')
    maskdata = csvdata.iloc[:, 1].values
    imagedata = csvdata.iloc[:, 0].values
    # shuffle imagedata and maskdata together
    perm = np.arange(len(imagedata))
    np.random.shuffle(perm)
    imagedata = imagedata[perm]
    maskdata = maskdata[perm]

    Vnet3d = Vnet3dModule(128, 128, 32, channels=1, costname=("dice coefficient",))
    Vnet3d.train(imagedata, maskdata, "Vnet3d.pd", "log\\segmeation\\", 0.001, 0.5, 10, 1)


train()
