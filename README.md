# 2018_Huawei_Software_Challenge
2018华为软件精英挑战赛初赛代码
赛题介绍：http://codecraft.devcloud.huaweicloud.com/home/detail
## 使用的预测模型：
本题需要预测在给定段时间内各虚拟机种类的申请数量，属于回归问题。我们利用线性回归，岭回归，LASSO回归，局部加强线性回归作为预测模型进行尝试。
## 特征选择：
* 特征1：本题需要预测在给定段时间n内各虚拟机种类的申请数量，因此我们以n为窗口大小，m(0<m<n)为步长，计算训练数据中n天内各虚拟机种类的申请数量。
* 特征2：段时间n内每个星期1～7的个数。
* 对特征1,2做最大最小归一化。
## 放置算法：
## 文件描述：
* ecs.py：对输入文件的处理。
* const.py:基本类的定义，以及对Matrix类简单功能的实现。
* regmodel.py：对线性回归，岭回归，LASSO回归，局部加强线性回归等算法的实现。
* predictor.py：特征选择，预测，及放置。
