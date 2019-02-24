from __future__ import division
import datetime as dt
from collections import Counter
import copy
import random


class Server:
    server_type = {
        'flavor1': {'CPU': 1, 'MEM': 1},
        'flavor2': {'CPU': 1, 'MEM': 2},
        'flavor3': {'CPU': 1, 'MEM': 4},
        'flavor4': {'CPU': 2, 'MEM': 2},
        'flavor5': {'CPU': 2, 'MEM': 4},
        'flavor6': {'CPU': 2, 'MEM': 8},
        'flavor7': {'CPU': 4, 'MEM': 4},
        'flavor8': {'CPU': 4, 'MEM': 8},
        'flavor9': {'CPU': 4, 'MEM': 16},
        'flavor10': {'CPU': 8, 'MEM': 8},
        'flavor11': {'CPU': 8, 'MEM': 16},
        'flavor12': {'CPU': 8, 'MEM': 32},
        'flavor13': {'CPU': 16, 'MEM': 16},
        'flavor14': {'CPU': 16, 'MEM': 32},
        'flavor15': {'CPU': 16, 'MEM': 64},
    }

    def __init__(self, CPU_num, MEM_num):
        self.CPU_num = CPU_num
        self.MEM_num = MEM_num
        self.CPU_remain = CPU_num
        self.MEM_remain = MEM_num
        self.vitu_cnt = Counter()

    def can_place(self, ser_type):
        return self.CPU_remain > Server.server_type[ser_type]['CPU'] \
               and self.MEM_remain > Server.server_type[ser_type]['MEM']

    def place(self, ser_type):
        self.CPU_remain -= Server.server_type[ser_type]['CPU']
        self.MEM_remain -= Server.server_type[ser_type]['MEM']
        self.vitu_cnt[ser_type] += 1


class Data:
    class Target:
        def __init__(self, target_array):
            target_array = [item.strip() for item in target_array]
            strs = target_array[0].split(' ')
            self.CPU, self.MEM = (int(strs[0]), int(strs[1]))
            self.type_num = int(target_array[2])
            self.ser_list = []
            for i in range(self.type_num):
                strs = target_array[i + 3].split(' ')
                self.ser_list.append(strs[0])
            self.type = target_array[self.type_num + 4][:3]
            self.start_time = dt.datetime.strptime(target_array[self.type_num + 6], '%Y-%m-%d %H:%M:%S')
            self.end_time = dt.datetime.strptime(target_array[self.type_num + 7], '%Y-%m-%d %H:%M:%S')

    def get_ecs_datas(self, ecs_lines):
        datas = []
        for line in ecs_lines:
            strs = line.strip().split('\t')
            datas.append({
                'type': strs[1],
                'date': dt.datetime.strptime(strs[2], '%Y-%m-%d %H:%M:%S')
            })
        return datas

    def __init__(self, ecs_lines, input_lines):
        self.ecs_datas = self.get_ecs_datas(ecs_lines)
        self.targets = Data.Target(input_lines)


class Matrix:

    def __init__(self, rows):
        self.value = copy.deepcopy(rows)
        self.width, self.height = len(rows[0]), len(rows)
        for i in range(self.height):
            for j in range(self.width):
                self.value[i][j] = float(self.value[i][j])

    def __add__(self, obj):
        return Matrix([[self.value[i][j] + obj.value[i][j] for j in range(self.width)] for i in range(self.height)])

    def __sub__(self, obj):
        return Matrix([[self.value[i][j] - obj.value[i][j] for j in range(self.width)] for i in range(self.height)])

    def __mul__(self, obj):

        if not isinstance(obj, Matrix):
            return Matrix([[item * obj for item in row] for row in self.value])
        result = []
        for i in range(self.height):
            row = []
            for j in range(obj.width):
                row.append(sum(map(lambda x, y: x * y, self.value[i], [obj.value[k][j] for k in range(obj.height)])))
            result.append(row)
        return Matrix(result)
    #代数余子式
    def cofactor(self, row, column):
        return ((-1) ** (row + column)) * self.minor(row, column)

    @staticmethod
    #生成单位矩阵
    def eye(size):
        result = []
        for i in range(size):
            result.append([1 if i == j else 0 for j in range(size)])
        return Matrix(result)
    #向矩阵添加随机扰动
    def random_self(self):
        for i in range(self.height):
            for j in range(self.width):
                self.value[i][j] += (random.randint(-100, 100) / 10000000)

    def cofactor_matrix(self):
        return Matrix([[self.cofactor(i, j) for j in range(self.width)] for i in range(self.height)])

    def delete_column(self, column):
        self.width -= 1
        for row in self.value:
            row.pop(column)

    def delete(self, row, column):
        self.height -= 1
        self.value.pop(row)
        self.width -= 1
        for row in self.value:
            row.pop(column)

    def add_row(self, row):
        self.height += 1
        return self.value.append(row)

    #行列式
    def determinant(self):
        if self.height == 1 and self.width == 1:
            return self.value[0][0]
        return float(sum([self.value[0][i] * self.cofactor(0, i) for i in range(self.width)]))

    #判断是否可逆
    def invertible(self):
        return self.width == self.height and self.determinant() != 0

    def minor(self, i, j):
        m = Matrix(self.value)
        m.delete(i, j)
        return m.determinant()

    #转置
    def transpose(self):
        return Matrix([[self.value[j][i] for j in range(self.height)] for i in range(self.width)])

    #求逆
    def inverse(self):
        while self.determinant() == 0:
            self.random_self()
        #伴随矩阵乘行列式的倒数
        return self.cofactor_matrix().transpose() * (1 / self.determinant())

    def LUPDecomposition(self, A):
        n = len(A)
        pi = [i for i in range(n)]
        for k in range(n):
            k1 = 0
            p = 0
            for i in range(k, n):
                if abs(A[i][k]) > p:
                    p = abs(A[i][k])
                    k1 = i
            if p == 0:
                return None
            pi[k], pi[k1] = pi[k1], pi[k]
            for i in range(n):
                A[k][i], A[k1][i] = A[k1][i], A[k][i]

            for i in range(k + 1, n):
                A[i][k] = A[i][k] / A[k][k]
                for j in range(k + 1, n):
                    A[i][j] = A[i][j] - A[i][k] * A[k][j]

        return A, pi

    def splitLU(self, A):
        n = len(A)

        L = [[0 for row in range(n)] for col in range(n)]
        U = [[0 for row in range(n)] for col in range(n)]

        for jc in range(n):
            L[jc][jc] = 1.0
            for i in range(jc):
                L[jc][i] = A[jc][i]
        for jc in range(n):
            for i in range(jc, n):
                U[jc][i] = A[jc][i]

        return L, U

    def LUPSolve(self, L, U, pi, b):
        n = len(L)
        x = [0.0 for jc in range(n)]
        y = [0.0 for jc in range(n)]
        for i in range(n):
            summation = 0.0
            for j in range(i):
                summation += L[i][j] * y[j]
            y[i] = b[pi[i]] - summation
        for i in reversed(range(n)):
            summation = 0.0
            for j in range(i + 1, n):
                summation += U[i][j] * x[j]
            x[i] = (y[i] - summation) / U[i][i]
        return x

    def LUP_inverse(self):

        inv_A = []
        while True:
            A = copy.deepcopy(self.value)
            tmp = self.LUPDecomposition(A)
            if tmp is None:
                self.random_self()
            else:
                A, PP = tmp
                LL, UU = self.splitLU(A)
                break
        for i in range(self.height):
            b = [0] * self.height
            b[i] = 1
            inv_A_each = self.LUPSolve(LL, UU, PP, b)
            inv_A.append(inv_A_each)
        ret_A = Matrix(inv_A).transpose()
        return ret_A