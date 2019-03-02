from const import Matrix
import math
import random

class LinerModel:
    def __init__(self, m_x, m_y, **query):
        self.b = (m_x.transpose() * m_x).LUP_inverse() * m_x.transpose() * m_y

    def predict(self, m_x):
        return int(round((m_x * self.b).value[0][0]))



class RidgeModel:
    def __init__(self,m_x,m_y,lam = 0.2, **query):
        self.demon = m_x.transpose() * m_x + eye(len(m_x[0]))*lam
        self.b = self.demon.LUP_inverse()*m_x.transpose()*m_y
    def predict(self,m_x):
        return int(round((m_x * self.b).value[0][0]))



class LwlrModel:
    def __init__(self, m_x, m_y, **query):
        self.k = query['k']
        self.m_x = m_x
        self.m_y = m_y

    def predict(self, m_x):#每次测试都要计算参数
        weights = Matrix.eye(self.m_x.height)  #h为样本数量，self.m_x为训练值
        for j in range(weights.height):
            diff_mat = m_x - Matrix([self.m_x.value[j]])#测试值与每个训练值的差异
            weights.value[j][j] = math.exp((diff_mat * diff_mat.transpose() * (-0.5 / self.k ** 2)).value[0][0])#weights是对角阵，weights[j][j]为测试值与第j个样本的高斯核
        xTx = self.m_x.transpose() * weights * self.m_x
        ws = xTx.LUP_inverse() * self.m_x.transpose() * weights * self.m_y#求w值，即(XTDX)-1*XTDY
        return int(round((m_x * ws).value[0][0]))
