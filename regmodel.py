from const import Matrix
import math
import random
# class ListLinerModel:
#     def __init__(self, m_x, m_y, **query):
#         lenght = 5
#         name = query['name']
#         self.index = int(name[6:]) - 1
#         new_X = []
#         new_Y = []
#         for i in range(len(m_x.value) - lenght):
#             x = []
#             for j in range(lenght + 1):
#                 x.append(m_x.value[i + j][self.index])
#             new_X.append(x)
#             new_Y.append([m_y.value[i + lenght][0]])

#         m_x = Matrix(new_X)
#         m_y = Matrix(new_Y)
#         self.last = new_X[-1]
#         self.b = (m_x.transpose() * m_x).LUP_inverse() * m_x.transpose() * m_y

#     def predict(self, m_x):
#         del self.last[0]
#         self.last.append(m_x.value[0][self.index])
#         m_x = Matrix([self.last])
#         return int(round((m_x * self.b).value[0][0])) if int(round((m_x * self.b).value[0][0])) > 0 else 0


class LinerModel:
    def __init__(self, m_x, m_y, **query):
        self.b = (m_x.transpose() * m_x).LUP_inverse() * m_x.transpose() * m_y

    def predict(self, m_x):
        return int(round((m_x * self.b).value[0][0]))


class CopyModel:
    def __init__(self, m_x, m_y, **query):
        self.name = query['name']
        pass

    def predict(self, m_x):
        return 0



class HoltWintersModel:

    def __init__(self, m_x, m_y, **query):

        season_size = int(query['season_size'])
        number_of_seasons = int(query['number_of_seasons'])
        name = query['name']
        self.index = int(name[6:]) - 1
        new_X = []
        for i in m_x.value[0-season_size*number_of_seasons:]:
            new_X.append(i[self.index])

        self.time_series = new_X

        self.number_of_seasons = number_of_seasons
        self.season_size = season_size
        self.seasonal_factor = []
        self.average_component = []
        self.tendence_component = []
        self.seasonal_component = []
        self.init_components()
        self.calculate_components(0.2, 0.15, 0.05)

    def season_time(self, time):

        if (time % self.season_size) is 0:
            return int((time / self.season_size))
        return int((time / self.season_size)) + 1

    def predict(self, m_x):
        ls = len(self.time_series)
        return int(round(self.estimate(ls+1, ls)))

    def season_index_time(self, time):

        seasonal_index = time % self.season_size
        if seasonal_index is 0:
            seasonal_index = self.season_size
        return seasonal_index

    def season_moving_average(self, season):
        floor = (season - 1) * self.season_size
        ceil = season * self.season_size
        moving_average = 0.0

        for y in self.time_series[floor:ceil]:
            moving_average += y
        return moving_average/self.season_size
    
    def season_factor(self, time):

        season = self.season_time(time)
        moving_average = self.season_moving_average(season)
        seasonal_index = self.season_index_time(time)
        tendence_component = self.tendence_component[0]

        tendence = (((self.season_size + 1.0) / 2.0) - seasonal_index) * tendence_component
        print(moving_average, tendence)
        seasonal_index_average = moving_average - tendence + 1
        factor = self.time_series[time - 1] / seasonal_index_average

        return factor

    def insert_seasonal_component(self, seasonal_index, season, value):

        if (len(self.seasonal_component)) < seasonal_index:
            while len(self.seasonal_component) < seasonal_index:
                self.seasonal_component.append([])
        
        if len(self.seasonal_component[seasonal_index - 1]) < (season + 1):
            while len(self.seasonal_component[seasonal_index - 1]) < (season + 1):
                self.seasonal_component[seasonal_index - 1].append(None)
        self.seasonal_component[seasonal_index - 1][season] = value

    def season_component(self, time):

        if time <= 0:
            season = 0
            seasonal_index = self.season_index_time(time + self.season_size)
        else:
            seasonal_index = self.season_index_time(time)
            season = self.season_time(time)
        return self.seasonal_component[seasonal_index - 1][season]

    def estimate(self, time, base_time):
        estimation = (self.average_component[base_time] + self.tendence_component[base_time] * (time - base_time)) * \
                     self.season_component(time - self.season_size)
        return estimation

    def init_components(self):

        first_season_average = self.season_moving_average(1)
        last_season_average = self.season_moving_average(self.number_of_seasons)
        self.tendence_component.append((last_season_average - first_season_average) /
                                       ((self.number_of_seasons - 1) * self.season_size))
        self.average_component.append(first_season_average - ((self.season_size / 2.0) * self.tendence_component[0]))
        for time in range(1, len(self.time_series) + 1):
            self.seasonal_factor.append(self.season_factor(time))

        seasonal_index_average = []
        for seasonal_index in range(1, self.season_size + 1):
            seasonal_index_sum = 0.0
            for m in range(self.number_of_seasons):
                index = seasonal_index + (m * self.season_size)
                factor = self.seasonal_factor[index - 1]
                seasonal_index_sum += factor
            seasonal_index_average.append(seasonal_index_sum * (1.0 / self.number_of_seasons))

        snt_average_sum = 0.0
        for snt_average in seasonal_index_average:
            snt_average_sum += snt_average
        adjustment_level = self.season_size / snt_average_sum

        for seasonal_index in range(1, self.season_size + 1):
            value = seasonal_index_average[seasonal_index - 1] * adjustment_level
            self.insert_seasonal_component(seasonal_index, 0, value)

    def calculate_components(self, alpha, beta, gamma):
        for time in range(1, len(self.time_series) + 1):
            average_component = alpha * (self.time_series[time - 1] / self.season_component(time - self.season_size)) \
                                + ((1 - alpha) * (self.average_component[time - 1] + self.tendence_component[time - 1]))
            self.average_component.append(average_component)
            tendence_component = (beta * (self.average_component[time] - self.average_component[time - 1])) + \
                                 ((1 - beta) * self.tendence_component[time - 1])
            self.tendence_component.append(tendence_component)
            seasonal_component = gamma * self.time_series[time - 1] / self.average_component[time] + \
                                 (1 - gamma) * self.season_component(time - self.season_size)

            index = self.season_index_time(time)
            season = self.season_time(time)
            self.insert_seasonal_component(index, season, seasonal_component)


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
