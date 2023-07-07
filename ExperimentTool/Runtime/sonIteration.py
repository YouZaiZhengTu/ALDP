'''
该类是一个装饰器，用于监控每一轮迭代的cputime、memorySize等，并且定向输出结果到日志包
该类要求：
@:parameter:
outputDirPath: str 中间过程输出日志
dataSet: str 数据集名称
size: 数据集的实例数量
iteration: 迭代序号
computeRandIndex: bool 是否开启兰德指数的计算,默认关闭，直接输出聚类结果到日志包
computeMutualInfo: bool 是否开启互信息指数的计算,默认关闭，直接输出聚类结果到日志包
@:returns
roots 树结构的根节点集,函数会调用树遍历算法求解分配集并计算结果
'''
import logging
import os
import threading
import time
from functools import wraps

import pandas as pd
import psutil

logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s- %(message)s')


class experimentMonitor(object):
    def __init__(self, outputDirPath: str, experimentId: str, algorithmName: str):
        self.baseDirPath = os.path.join(outputDirPath, experimentId, algorithmName)
        self.experimentId = experimentId
        self.algorithmName = algorithmName
        self.parameter = None
        self.dataName = None

    # 使用实例本身进行调用，将对象调用变成函数调用
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.parameter = kwargs['parameter']
            self.dataName = kwargs['dataName']
            if not self.repeat_thread_detection("monitor"):
                t = threading.Thread(target=self.performance_monitoring, name="monitor")
                t.start()
            res = func(*args, **kwargs)
            result = res['result']
            cpu_time = res['cpu_time']
            logging.info('%s 数据集,测试轮次: %d ,迭代轮次 %d,参数 %s,本轮执行%0.3f s ' % (
                kwargs['dataName'], kwargs['iteration'], kwargs['sonIteration'], kwargs['parameter'], cpu_time))
            self.outResult(result, kwargs['parameter'], kwargs['iteration'], kwargs['sonIteration'], kwargs['dataName'])
            self.outputCpuTime(cpu_time, kwargs['parameter'], kwargs['iteration'], kwargs['sonIteration'],
                               kwargs['dataName'])
            return res

        return wrapper

    def outResult(self, result, parameter, iteration, sonIteration, dataName):
        if result:
            for key in result:
                outputPath = os.path.join(self.baseDirPath, str(parameter), 'iteration', str(iteration), 'result',
                                          dataName, key)
                self.makeDir(outputPath)
                pd.DataFrame(result[key]) \
                    .to_csv(outputPath + '/' + str(sonIteration) + '.csv', index=False)

    def outputCpuTime(self, cpu_time, parameter, iteration, sonIteration, dataName):
        data = dict(dataSet=dataName, iteration=sonIteration, cpu_time=cpu_time)
        outputPath = os.path.join(self.baseDirPath, str(parameter), 'iteration', str(iteration), 'evaluation')
        self.lineOutput(outputPath, 'cpu_time', data)

    # 查询线程是否活动
    def repeat_thread_detection(self, tName):
        # 判断 tName线程是否处于活动状态
        for item in threading.enumerate():
            if tName == item.name:  # 如果名字相同，说明tName线程在活动的线程列表里面
                return True
        return False

    # 开启一个线程监控内存变动
    def performance_monitoring(self):
        p = psutil.Process(os.getpid())
        while True:
            current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            cpu_percent = round(((p.cpu_percent(interval=1) / 100) / psutil.cpu_count(logical=False)) * 100, 2)
            mem_percent = round(p.memory_percent(), 2)
            mem_size = p.memory_info().rss
            data = dict(dateTime=current_time, cpu_utilization_ratio=cpu_percent,
                        memory_utilization_ratio=mem_percent,
                        memory_size_mb=round(mem_size / 1024 / 1024, 4))
            outputPath = os.path.join(self.baseDirPath, str(self.parameter), 'log')
            self.lineOutput(outputPath, str(self.dataName), data)
            time.sleep(0.1)

    # 目录创建工具
    def makeDir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    # 输出信息工具
    def lineOutput(self, path, fileName, data: dict):
        self.makeDir(path)
        outputPath = os.path.join(path, fileName + '.csv')
        if not os.path.exists(outputPath):
            pd.DataFrame(data, index=[0]).to_csv(outputPath, index=False, mode='a')
        else:
            pd.DataFrame(data, index=[0]).to_csv(outputPath, index=False, header=False, mode='a')

    # 格式化参数列表
    def getParams(self, data):
        params = ''
        for key in data:
            params = params + str(data[key]) + '-'
        return params[:-1]
