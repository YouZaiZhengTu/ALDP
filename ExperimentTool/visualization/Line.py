import os

import matplotlib.pyplot as plt
import pandas as pd

from ExperimentTool.visualization.Basis import Task, initEvalTask


class Line():
    def __init__(self, task: Task, algorithm=[], major=''):
        self.task = task
        self.true_class = 0
        self.data = {}
        self.algorithm = algorithm
        self.major = major
        self.minY, self.maxY = 10, 0
        self.range = 30
        self.initData()

    def initData(self):
        data = pd.read_csv(self.task.filePath)
        self.true_class = data[['class']].max().values.tolist()[0]
        self.algorithm = self.algorithm if len(self.algorithm) != 0 else data['algorithm'].unique().tolist()
        for a_name, a_data in data.groupby('algorithm'):
            if a_name in self.algorithm:
                for c_name, c_data in a_data.groupby('pred_class'):
                    if a_name not in self.data.keys():
                        self.data[a_name] = {}
                    self.data[a_name][c_name] = c_data['mean'].max()
                    if c_data['mean'].max() > self.maxY: self.maxY = c_data['mean'].max()
                    if c_data['mean'].max() < self.minY: self.minY = c_data['mean'].max()
        vsDataPath = os.path.join(self.task.baseDir, 'visualization', 'data', self.task.type, self.major)
        os.makedirs(vsDataPath, exist_ok=True)
        d = pd.DataFrame(self.data).iloc[int(self.true_class - self.range if self.true_class - self.range > 0 else 0): int(self.true_class + self.range)]
        d = d.sort_index()
        d.to_csv(vsDataPath + '/' + self.task.dataName + '.csv')

    def draw(self):
        if self.major in self.data.keys():
            majorData = self.data.pop(self.major)
            plt.figure(figsize=(15, 10), dpi=100)
            plt.plot(majorData.keys(), majorData.values(), linestyle='-', linewidth=2, label=self.major)
            plt.scatter(majorData.keys(), majorData.values())
            for key in self.data:
                plt.plot(self.data[key].keys(), self.data[key].values(), linestyle=':', label=key)
                plt.scatter(self.data[key].keys(), self.data[key].values())
            plt.vlines(self.true_class, self.minY, self.maxY, linestyles="solid", colors="k")
            plt.legend(loc='best')
            plt.axis(
                [self.true_class - self.range if self.true_class - self.range > 0 else 0, self.true_class + self.range,
                 self.minY,
                 self.maxY])
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.xlabel('cluster', fontdict={'size': 16})
            plt.ylabel(self.task.type, fontdict={'size': 16})
            plt.title(self.task.dataName, fontdict={'size': 20})
            path = os.path.join(self.task.baseDir, 'visualization', self.task.type, self.major)
            os.makedirs(path, exist_ok=True)
            # plt.show()
            plt.savefig(path + '/' + self.task.dataName + '.png')
            plt.close()


if __name__ == "__main__":
    tasks = initEvalTask()
    for task in tasks:
        line = Line(task, ['HKMeans', 'Affinity', 'PERCH', 'DPC', 'streaKHC', 'AHC', 'RSC', 'GIRNCH', 'RNNNoParam',
                           'RNNSearch', 'Munec', 'SCC'], 'RNNSearch')
        line.draw()
    print('输出完成')
