from __future__ import division
import datetime as dt
from collections import Counter
import copy
from const import Data, Matrix, Server
from regmodel import LwlrModel, LinerModel, ListLinerModel, HoltWintersModel, CopyModel


class Regression:

    def __init__(self, list_x, list_y, targets, model=LinerModel, **query):
        self.usebase = query['usebase']query
        if self.usebase:
            for i in range(len(list_x)):
                for ser in targets.ser_list:
                    list_y[i][ser] -= list_x[i][ser]

        self.future_list = []
        self.future_list_init()
        result = []
        for x in list_x:
            line = []
            for future in self.future_list:
                line.append(x[future])
            result.append(line)

        self.targets = targets
        self.matrix_x = Matrix(result)
        self.predict_y = {}
        self.model = {}

        for ser in targets.ser_list:
            self.predict_y[ser] = Matrix([[item[ser]] for item in list_y])
            query.update({'name': ser})

            self.model[ser] = model(self.matrix_x, self.predict_y[ser], **query)

    def future_list_init(self):

        self.future_list.extend(['week'+str(i) for i in range(7)])
        #self.future_list.extend(['day'+str(i) for i in range(1,32)])
        #self.future_list.extend(['month'+str(i) for i in range(1, 13)])
        #self.future_list.extend(['flavor'+str(i) for i in range(1, 16)])

    def predict(self, list_x):
        result = []
        for x in list_x:
            line = []
            for future in self.future_list:
                line.append(x[future])
            result.append(line)


        m_x = Matrix(result)
        result = Counter()
        for ser in self.targets.ser_list:
            result[ser] = self.model[ser].predict(m_x)
            if self.usebase:
                result[ser] += list_x[0][ser]
            if result[ser] < 0:
                result[ser] = 0
        return result


class FutureExtract:

    def __init__(self, ecs_datas, targets, days=None):

        self.ecs_datas = ecs_datas
        self.targets = targets
        self.time_length = targets.end_time - targets.start_time

        days = self.time_length.days if days is None else days
        self.time_delta = dt.timedelta(days=days)

        date_tmp = ecs_datas[0]['date']
        self.train_start = dt.datetime(date_tmp.year, date_tmp.month, date_tmp.day)
        date_tmp = ecs_datas[-1]['date']
        self.train_end = dt.datetime(date_tmp.year, date_tmp.month, date_tmp.day) + dt.timedelta(days=1)

        # self.train_start = ecs_datas[0]['date']
        # self.train_end = ecs_datas[-1]['date']

        self.predict_start = targets.start_time
        self.predict_end = targets.end_time

        self.X = []
        self.Y = []
        self.predict_x = None
        self.get_train_data()

    def find_index(self, index_now, date):
        ecs_datas = self.ecs_datas
        while ecs_datas[index_now]['date'] < ecs_datas[-1]['date'] and ecs_datas[index_now]['date'] < date:
            index_now += 1
        return index_now

    def _get_y(self, ecs_datas, pre=''):

        result = Counter()
        for data in ecs_datas:
            result[pre+data['type']] += 1
        return result

    def stander(self, futures):
        max_num = [max(futures[j][i] for j in range(len(futures))) for i in range(len(futures[0]))]
        min_num = [min(futures[j][i] for j in range(len(futures))) for i in range(len(futures[0]))]
        for i in range(len(futures[0])):
            if max_num[i] == min_num[i] == 0:
                for j in range(len(futures)):
                    futures[j][i] = 0
            elif max_num[i] == min_num[i]:
                for j in range(len(futures)):
                    futures[j][i] = 1
            else:
                for j in range(len(futures)):
                    futures[j][i] = (futures[j][i] - min_num[i]) / (max_num[i] - min_num[i])
        return futures

    def _get_x(self, ecs_datas, time_index):
        futures = Counter()
        flavor = self._get_y(ecs_datas)
        futures.update(flavor)

        for i in range(self.time_length.days):
            day = time_index + dt.timedelta(i) + self.time_length
            futures['week'+str(day.weekday())] += 1
            futures['month' + str(day.month)] += 1
            futures['day' + str(day.day)] += 1

        return futures

    def get_train_data(self):

        time_index = int((self.train_end - self.time_length -self.train_start).days / self.time_delta.days) \
                     * self.time_delta
        time_index = self.train_end - time_index - self.time_length
        index = 0
        X = []

        while time_index + self.time_length < self.train_end - self.time_length:

            index = self.find_index(index, time_index)
            index_end = self.find_index(index, time_index+self.time_length)
            pre_end = self.find_index(index_end, time_index+2*self.time_length)
            X.append(self._get_x(self.ecs_datas[index:index_end], time_index))
            self.Y.append(self._get_y(self.ecs_datas[index_end:pre_end]))
            time_index += self.time_delta

        prex_index = self.find_index(index, self.train_end - self.time_length)
        predict_x = self._get_x(self.ecs_datas[prex_index:], self.train_end - self.time_length)
        X.append(predict_x)
        self.X = X[:-1]
        self.predict_x = X[-1:]


class Place:

    def __init__(self, pre_result, targets):

        self.pre_result = copy.deepcopy(pre_result)
        self.targets = targets
        self.cmp_type = self.targets.type
        self.ana_type = "MEM" if self.cmp_type == "CPU" else "CPU"
        self.ser_dict = Server.server_type
        self.ppre_result = pre_result

    def place(self):

        def cmp_place(ser1, ser2):
            if self.ser_dict[ser1][self.cmp_type] != self.ser_dict[ser2][self.cmp_type]:
                return self.ser_dict[ser1][self.cmp_type] - self.ser_dict[ser2][self.cmp_type]
            else:
                return self.ser_dict[ser1][self.ana_type] - self.ser_dict[ser2][self.ana_type]


        ser_cnt = 0
        ser_list = sorted(self.targets.ser_list, key=lambda x: self.ser_dict[x][self.cmp_type], reverse=True)
        ser_list = list(ser_list)
        pla_result = []
        while ser_cnt < self.pre_result['total']:
            server = Server(self.targets.CPU, self.targets.MEM)
            for ser in ser_list:
                while server.can_place(ser) and self.pre_result[ser] > 0:
                    server.place(ser)
                    self.pre_result[ser] -= 1
                    ser_cnt += 1
            pla_result.append(server)
        return pla_result

    def ratio_place(self):

        ser_cnt = 0

        pla_result = []
        while ser_cnt < self.pre_result['total']:
            if self.pre_result['total'] - ser_cnt < 2:
                pass

            server = Server(self.targets.CPU, self.targets.MEM)

            def cmp_radio(ser):
                ser_info = {'CPU': self.ser_dict[ser]['CPU'], 'MEM': self.ser_dict[ser]['MEM']}
                server_info = {'CPU': server.CPU_remain, 'MEM': server.MEM_remain,
                               'CPUT': server.CPU_num, 'MEMT': server.MEM_num}

                ratio_remain = (server_info['CPU'] - ser_info['CPU']) / (server_info['MEM'] - ser_info['MEM'] + 0.9)
                ratio_server = server_info['CPUT'] / server_info['MEMT']
                score = ser_info[self.cmp_type] * min(ratio_remain, ratio_server) / max(ratio_remain, ratio_server)
                return score

            ser_list = list(sorted(self.targets.ser_list, key=cmp_radio, reverse=True))
            can_place = {ser: True for ser in ser_list}

            while True:
                for ser in ser_list:
                    if server.can_place(ser) and self.pre_result[ser] > 0:
                        server.place(ser)
                        self.pre_result[ser] -= 1
                        ser_cnt += 1
                    else:
                        can_place[ser] = False

                if sum(can_place.values()) == 0:
                    break
                ser_list = list(sorted(self.targets.ser_list, key=cmp_radio, reverse=True))

            pla_result.append(server)
        return pla_result

    def ave_place(self):

        server_info = {'CPU': self.targets.CPU, 'MEM': self.targets.MEM}
        MEM_sum = sum(map(lambda x: self.ser_dict[x]['MEM'] * self.pre_result[x], self.ser_dict))
        CPU_sum = sum(map(lambda x: self.ser_dict[x]['CPU'] * self.pre_result[x], self.ser_dict))

        server_cnt = int(max(MEM_sum / server_info['MEM'], CPU_sum / server_info['CPU'])) + 1
        ser_cnt = 0
        pla_result = [Server(self.targets.CPU, self.targets.MEM) for i in range(server_cnt)]

        def cmp_place(ser1, ser2):
            if self.ser_dict[ser1][self.cmp_type] != self.ser_dict[ser2][self.cmp_type]:
                return self.ser_dict[ser1][self.cmp_type] - self.ser_dict[ser2][self.cmp_type]
            else:
                return self.ser_dict[ser1][self.ana_type] - self.ser_dict[ser2][self.ana_type]

        def cmp_ser(ser1, ser2):
            """
            :type ser1: Server
            :type ser2: Server

            """
            ser1_a, ser1_b = ser1.CPU_remain, ser1.MEM_remain
            ser2_a, ser2_b = ser2.CPU_remain, ser2.MEM_remain
            if self.cmp_type == 'MEM':
                ser1_a, ser1_b = ser1_b, ser1_a
                ser2_a, ser2_b = ser2_b, ser2_a

            if ser1_a != ser2_a:
                return ser1_a - ser2_a
            else:
                return ser1_b - ser2_b

        ser_list = list(sorted(self.targets.ser_list, cmp=cmp_place, reverse=True))

        while True:
            lase_cnt = ser_cnt
            for server in pla_result:
                for ser in ser_list:
                    if server.can_place(ser) and self.pre_result[ser] > 0:
                        server.place(ser)
                        self.pre_result[ser] -= 1
                        ser_cnt += 1
                        break
            if lase_cnt == ser_cnt:
                break
            pla_result.sort(cmp=cmp_ser, reverse=True)

        while ser_cnt < self.pre_result['total']:
            server = Server(self.targets.CPU, self.targets.MEM)
            for ser in ser_list:
                while server.can_place(ser) and self.pre_result[ser] > 0:
                    server.place(ser)
                    self.pre_result[ser] -= 1
                    ser_cnt += 1
            pla_result.append(server)

        cnt = 4
        plan = pla_result[-1]
        if sum(plan.vitu_cnt.values()) < cnt:
            for ser in plan.vitu_cnt:
                self.ppre_result[ser]-=plan.vitu_cnt[ser]
                self.ppre_result['total'] -= plan.vitu_cnt[ser]
            pla_result.pop()
            return pla_result
        n = 0
        while True:
            lastn = n
            for ser in ser_list:
                if plan.can_place(ser):
                    plan.place(ser)
                    self.ppre_result[ser] += 1
                    self.ppre_result['total'] += 1
                    n += 1
                    break
            if n == lastn or n == cnt:
                break


        return pla_result


class Result:

    def __init__(self, pre_result, pla_result, targets):
        self.pre_result = pre_result
        self.pla_result = pla_result
        self.targets = targets
        self.results = []
        self.generate()

    def generate(self):

        self.results.append(self.pre_result['total'])
        for ser in self.targets.ser_list:
            self.results.append(ser + ' ' + str(self.pre_result[ser]))
        self.results.append("")
        self.results.append(len(self.pla_result))
        cnt = 1
        for plan in self.pla_result:
            pla_str = str(cnt)
            cnt += 1
            for ser in self.targets.ser_list:
                pla_str += (' ' + ser + ' ' + str(plan.vitu_cnt[ser]))
            self.results.append(pla_str)


def predict_vm(ecs_lines, input_lines):

    # Do your work from here#
    if ecs_lines is None:
        return []
        print 'ecs information is none'

    if input_lines is None:
        print 'input file information is none'
        return []

    data = Data(ecs_lines, input_lines)
    futures = FutureExtract(data.ecs_datas, data.targets, days=3)

    s_size = dt.timedelta(30).days / futures.time_delta.days

    query={
        'season_size': s_size,
        'number_of_seasons': len(futures.X) / s_size,
        'k': 0.006,
        'usebase': True
    }

    model = Regression(futures.X, futures.Y, data.targets, LwlrModel, **query)
    pre_result = model.predict(futures.predict_x)
    pre_result['total'] = sum(pre_result.values())
    pla_result = Place(pre_result, data.targets).ave_place()
    result = Result(pre_result, pla_result, data.targets).results

    return result
