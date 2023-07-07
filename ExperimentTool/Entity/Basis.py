# 该类用于定义实验执行的基础行为，包括但不限于实验存储位置，分发策略等
import datetime
from enum import Enum, unique


# 基础类，用于记录算法的基本信息
class Basis(object):
    def __init__(self, storgePath, algorithmName):
        self.algorithmName = algorithmName
        self.storgePath = storgePath
        self.date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def __str__(self):
        return '{}-{}'.format(self.date, self.algorithmName)


# 任务类：初始化运行任务并控制任务循环
class Task():
    def __init__(self, params, iterIndex: int, dataName: str, path):
        self.params = params
        self.iterIndex = str(iterIndex)
        self.dataName = str(dataName)
        self.filePath = str(path)

    def __str__(self):
        return '{}-{}'.format(self.dataName, self.iterIndex)


# 记录类型类：用于控制输出的结果类型，该名称会被用于创建输出结果的上级目录
@unique
class RecordType(Enum):
    assignment = 'assignment'
    tree = 'tree'


# 记录类：保存每次输出的结果
class Record():
    def __init__(self):
        self.record = []
        self.cpuTime = []

    # 保存每轮聚类结果，必须明确输出类型，子迭代序数（多轮迭代结果），标签信息和预测结果
    def save_output(self, types: RecordType, label_true: list, label_pred: list, iter=0):
        assert isinstance(types, RecordType), TypeError(
            '输入类型必须为RecordType枚举类，请检查。当前类型为 ' + str(type(types)))
        assert len(label_pred) > 0, \
            TypeError('label_pred必须为list类型，且长度不为0')
        assert len(label_pred) > 0, \
            TypeError('label_true必须为list类型，且长度不为0')
        self.record.append(
            Assignment(types, str(iter), {'label_true': label_true, 'label_pred': label_pred}))

    # 保存每轮所用的时间，最终计算得到总时间。
    def save_time(self, cpuTime):
        assert isinstance(cpuTime, float), TypeError("输入类型必须为float类型，请检查。当前类型为 " + str(type(cpuTime)))
        self.cpuTime.append(cpuTime)


# 分配类，用于记录输出的结果
class Assignment():
    def __init__(self, types: str, iter: str, record: dict):
        self.type = types
        self.iter = iter
        self.record = record


if __name__ == '__main__':
    record = Record()
    record.save_time(0.02)
    record.save_output('assignment', [])
    print(1)
