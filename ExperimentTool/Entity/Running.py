import logging
import os
import threading
import time
from functools import wraps

import pandas as pd
import psutil

from ExperimentTool.Entity.Basis import Record, Assignment

logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s- %(message)s')


class ExpMonitor():
    def __init__(self, expId: str, algorithmName: str, storgePath="G:\Experiment"):
        self.task = None
        self.expId = expId
        self.algorithmName = algorithmName
        self.storgePath = storgePath

    # 使用实例本身进行调用，将对象调用变成函数调用
    def __call__(self, func):
        @wraps(func)
        # *args 表示任何多个无名参数，它是一个元组；**kwargs 表示关键字参数，它是一个字典。
        def wrapper(*args, **kwargs):
            self.task = kwargs['task'] if 'task' in kwargs else args[0]
            if not self.repeat_thread_detection("monitor"):
                t = threading.Thread(target=self.out_memory, name="monitor")
                t.start()
            res = func(*args, **kwargs)
            record: Record = res['record']
            self.out_record(record.record)
            self.out_cpu_time(record.cpuTime)
            logging.info('%15s 数据集第 %2d 轮测试完成，运行时间 %10.2f s，参数信息：%10s'
                         % (self.task.dataName.ljust(15)[:15], int(self.task.iterIndex),
                            sum(record.cpuTime), str(self.task.params).ljust(10)[:10]))
            return res

        return wrapper

    def out_record(self, records: list[Assignment]):
        assert len(records) > 0, \
            ValueError('输出结果record没有记录，请检查')
        for record in records:
            path = os.path.join(self.storgePath, self.expId, self.algorithmName, str(self.task.params), self.task.iterIndex,
                                'output',
                                self.task.dataName, record.type.value)
            self.makeDir(path)
            pd.DataFrame(record.record).to_csv(path + '/' + record.iter + '.csv', index=False)

    def out_cpu_time(self, cpuTime: list):
        assert len(cpuTime) > 0, \
            ValueError('cputime 没有记录，请检查')
        path = os.path.join(self.storgePath, self.expId, self.algorithmName, str(self.task.params), self.task.iterIndex,
                            'cpuTime')
        self.makeDir(path)
        self.lineOutput(path, 'cpuTime', dict(dataName=self.task.dataName, cpuTime=sum(cpuTime)))

    # 开启一个线程监控内存变动
    def out_memory(self, ):
        p = psutil.Process(os.getpid())
        while True:
            cpu_percent = round(((p.cpu_percent(interval=1) / 100) / psutil.cpu_count(logical=False)) * 100, 2)
            mem_percent = round(p.memory_percent(), 2)
            mem_size = p.memory_info().rss
            data = dict(cpu_utilization_ratio=cpu_percent,
                        memory_utilization_ratio=mem_percent,
                        memory_size_mb=round(mem_size / 1024 / 1024, 4))
            path = os.path.join(self.storgePath, self.expId, self.algorithmName, str(self.task.params), self.task.iterIndex,
                                'memory')
            self.lineOutput(path, self.task.dataName, data)
            time.sleep(0.1)

    # 输出信息工具
    def lineOutput(self, path, fileName, data: dict):
        self.makeDir(path)
        outputPath = os.path.join(path, fileName + '.csv')
        if not os.path.exists(outputPath):
            pd.DataFrame(data, index=[0]).to_csv(outputPath, index=False, mode='a')
        else:
            pd.DataFrame(data, index=[0]).to_csv(outputPath, index=False, header=False, mode='a')

    # 目录创建工具
    def makeDir(self, path):
        # exist_ok：只有在目录不存在时创建目录，目录已存在时不会抛出异常。
        os.makedirs(path, exist_ok=True)

    # 查询线程是否活动
    def repeat_thread_detection(self, tName):
        # 判断 tName线程是否处于活动状态
        for item in threading.enumerate():
            if tName == item.name:  # 如果名字相同，说明tName线程在活动的线程列表里面
                return True
        return False
