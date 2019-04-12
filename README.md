# KiTS19——2019 Kidney Tumor Segmentation Challenge
> This is an example of the CT images Kidney Tumor Segmentation
![](KiTS19_header.png)

## How to Use

**1、Preprocess**

* analyze the ct image,and get the slice thickness and window width and position:run the dataAnaly.py
* keep Kidney region range image:run the data2dprepare.py
* generate patch(128,128,32) kidney image and mask:run the data3dprepare.py
* save patch image and mask into csv file:run the utils.py,like file trainSegmentation.csv
* split trainSegmentation.csv into training set and test set:run subset.py

**2、Kidney Segmentation**
* the VNet model

![](3dVNet.png) 

* train and predict in the script of vnet3d_train.py and vnet3d_predict.py


## Result


## Contact
* https://github.com/junqiangchen
* email: 1207173174@qq.com，yixuanwang@hust.edu.cn
* Contact: junqiangChen，yixuanWang（王艺璇）
* WeChat Number: 1207173174
* WeChat Public number: 最新医学影像技术
