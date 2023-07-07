import os
from enum import Enum, unique

import pandas as pd
from sklearn.metrics import rand_score, adjusted_rand_score, normalized_mutual_info_score
from tqdm import tqdm


@unique
class TaskType(Enum):
    cpuTime = 'cpuTime'
    memory = 'memory'
    assignment = 'assignment'
    tree = 'tree'


@unique
class evaluationType(Enum):
    rand_index = 'ri'
    adjust_rand_index = 'ari'
    normal_mutual_information = 'nmi'


class Task():
    def __init__(self, dataName: str, type: str, algorithmName: str, parameter: str, iteration: str, filePath: str):
        self.dataName = dataName
        self.type = type
        self.algorithmName = algorithmName
        self.parameter = parameter
        self.iteration = iteration
        self.filePath = filePath

    def __str__(self):
        return '{}-{}-{}-{}'.format(self.algorithmName, self.dataName, self.type, self.iteration)


class Basis():
    def __init__(self, expId: list[str], storgePath="G:\Experiment"):
        self.storgePath = storgePath
        self.expId = expId if len(expId) > 0 and expId is not None else os.listdir(self.storgePath)
        self.cpuTimeTasks = []
        self.memoryTasks = []
        self.assignmentTasks = []
        self.treeTasks = []
        self.initTasks()

    def initTasks(self):
        for expId in self.expId if self.expId else os.listdir(self.storgePath):
            rootPath = os.path.join(self.storgePath, expId)
            for algorithmName in os.listdir(rootPath):
                algorithmNamePath = os.path.join(rootPath, algorithmName)
                for parameter in os.listdir(algorithmNamePath):
                    parameterPath = os.path.join(algorithmNamePath, parameter)
                    for iteration in os.listdir(parameterPath):
                        iterationPath = os.path.join(parameterPath, iteration)
                        for fileName in os.listdir(os.path.join(iterationPath, 'cpuTime')):
                            # 读取每个CPUTIME文件并初始化任务
                            self.cpuTimeTasks.append(
                                Task(TaskType.cpuTime.value, TaskType.cpuTime.value, algorithmName, parameter,
                                     iteration,
                                     os.path.join(iterationPath, 'cpuTime', fileName)))
                        # 读取memory文件夹，将内存记录读入
                        if os.path.exists(os.path.join(iterationPath, 'memory')):
                            for fileName in os.listdir(os.path.join(iterationPath, 'memory')):
                                dataName = fileName.split('.')[0]
                                self.memoryTasks.append(
                                    Task(dataName, TaskType.memory.value, algorithmName, parameter, iteration,
                                         os.path.join(iterationPath, 'memory', fileName)))
                        # 读取分配结果文件夹
                        for dataName in os.listdir(os.path.join(iterationPath, 'output')):
                            # 读取簇分配结果
                            sonIterationPath = os.path.join(iterationPath, 'output', dataName, 'assignment')
                            if os.path.exists(sonIterationPath):
                                for sonIterationName in os.listdir(sonIterationPath):
                                    self.assignmentTasks.append(
                                        Task(dataName, TaskType.assignment.value, algorithmName, parameter,
                                             iteration,
                                             os.path.join(sonIterationPath, sonIterationName)))
                            sonIterationPath = os.path.join(iterationPath, 'output', dataName, 'tree')
                            if os.path.exists(sonIterationPath):
                                for sonIterationName in os.listdir(sonIterationPath):
                                    self.treeTasks.append(
                                        Task(dataName, TaskType.tree.value, algorithmName, parameter,
                                             iteration,
                                             os.path.join(sonIterationPath, sonIterationName)))


class Result():
    def __init__(self, data: Basis, base_path="G:/Result"):
        self.base_path = base_path
        self.data = data
        self.result = pd.DataFrame()

    def statistic_cpu_time(self):
        data = pd.DataFrame()
        bar = tqdm(self.data.cpuTimeTasks)
        for cpuTime in bar:
            bar.set_description("正在处理cpuTime记录")
            d = pd.read_csv(cpuTime.filePath)
            d = d[d['cpuTime'] > 0]
            d.insert(loc=d.shape[1], column='parameter', value=cpuTime.parameter)
            d.insert(loc=d.shape[1], column='iteration', value=int(cpuTime.iteration))
            d.insert(loc=d.shape[1], column='algorithm', value=str(cpuTime.algorithmName))
            data = pd.concat((data, d), axis=0)
        for algorithm, aGroup in data.groupby('algorithm'):
            path = os.path.join(self.base_path, 'data', 'cpuTime')
            os.makedirs(path, exist_ok=True)
            aGroup.sort_values(by=['dataName', 'cpuTime'], ascending=False)[
                ['dataName', 'parameter', 'iteration', 'cpuTime']] \
                .to_csv(path + '/' + algorithm + '.csv', index=False)

    def statistic_memory(self):
        data = pd.DataFrame()
        bar = tqdm(self.data.memoryTasks)
        for memory in bar:
            bar.set_description("正在处理内存记录")
            d = pd.read_csv(memory.filePath)
            d = [{'dataName': memory.dataName,
                  'parameter': memory.parameter,
                  'iteration': int(memory.iteration),
                  'algorithm': str(memory.algorithmName),
                  'memory': d[['memory_size_mb']].max()[0]}]
            data = pd.concat((data, pd.DataFrame(d)), axis=0)
        for algorithm, aGroup in data.groupby('algorithm'):
            path = os.path.join(self.base_path, 'data', 'memory')
            os.makedirs(path, exist_ok=True)
            aGroup.sort_values(by=['dataName', 'memory'], ascending=False)[
                ['dataName', 'parameter', 'iteration', 'memory']] \
                .to_csv(path + '/' + algorithm + '.csv', index=False)

    def compute_evaluation(self):
        result = []
        bar = tqdm(self.data.assignmentTasks)
        for assignment in bar:
            bar.set_description('正在评估指标')
            data = pd.read_csv(assignment.filePath)
            for eval in evaluationType:
                value = 0
                # print(assignment.filePath)
                if eval == evaluationType.rand_index:
                    value = rand_score(data['label_true'], data['label_pred'])
                if eval == evaluationType.adjust_rand_index:
                    value = adjusted_rand_score(data['label_true'], data['label_pred'])
                if eval == evaluationType.normal_mutual_information:
                    value = normalized_mutual_info_score(data['label_true'], data['label_pred'])
                result.append({'evaluation': str(eval.value),
                               'dataName': assignment.dataName,
                               'parameter': assignment.parameter,
                               'iteration': int(assignment.iteration),
                               'algorithm': str(assignment.algorithmName),
                               'sample': data['label_true'].shape[0],
                               'class': data['label_true'].drop_duplicates().shape[0],
                               'pred_class': data['label_pred'].drop_duplicates().shape[0],
                               'value': value})
        result = pd.DataFrame(result)
        for a_name, a_data in result.groupby('algorithm'):
            for d_name, d_data in a_data.groupby('dataName'):
                path = os.path.join(self.base_path, 'data', 'evaluation', a_name, d_name)
                os.makedirs(path, exist_ok=True)
                for e_name, e_data in d_data.groupby('evaluation'):
                    out = e_data[['parameter', 'sample', 'class', 'pred_class', 'value']] \
                        .sort_values(by=['value'], ascending=False)
                    out.to_csv(path + '/' + e_name + '.csv', index=False)


class Statistics():
    def __init__(self, data_dir="G:/Result/data", base_path="G:/Result/Statistics"):
        self.data_dir = data_dir
        self.base_path = base_path

    def cpuTime(self):
        objectDir = os.path.join(self.data_dir, TaskType.cpuTime.value)
        record = []
        max = []
        min = []
        mean = []
        for algorithm in os.listdir(objectDir):
            data = pd.read_csv(os.path.join(objectDir, algorithm))
            statistics = []
            for d_name, d_data in data.groupby('dataName'):
                d_statistics = []
                mean.append({'algorithm': algorithm.split('.')[0],
                             'dataName': d_name,
                             TaskType.cpuTime.value: d_data[[TaskType.cpuTime.value]].mean().values[0]})
                for p_name, p_data in d_data.groupby('parameter'):
                    d_statistics.append({'algorithm': algorithm.split('.')[0],
                                         'dataName': d_name,
                                         'parameter': p_name,
                                         TaskType.cpuTime.value: p_data[[TaskType.cpuTime.value]].mean().values[0]})
                d_statistics = sorted(d_statistics, key=lambda x: x[TaskType.cpuTime.value], reverse=True)
                max.append(d_statistics[0])
                min.append(d_statistics[-1])
                statistics.extend(d_statistics)
            record.extend(statistics)
        dir = os.path.join(self.base_path, TaskType.cpuTime.value)
        os.makedirs(dir, exist_ok=True)
        pd.DataFrame(record).to_csv(dir + '/' + 'record.csv', index=False)
        max = pd.DataFrame(max)
        max_out = pd.DataFrame(max[['dataName']].drop_duplicates())
        max_out.index = max_out[['dataName']]
        for a_name, a_data in max.groupby('algorithm'):
            d = a_data[['cpuTime']]
            d.index = a_data[['dataName']]
            d.columns = [a_name]
            max_out = pd.concat((max_out, d), axis=1)
        max_out.to_csv(dir + '/' + 'maxRecord.csv', index=False)
        min = pd.DataFrame(min)
        min_out = pd.DataFrame(min[['dataName']].drop_duplicates())
        min_out.index = min_out[['dataName']]
        for a_name, a_data in min.groupby('algorithm'):
            d = a_data[['cpuTime']]
            d.index = a_data[['dataName']]
            d.columns = [a_name]
            min_out = pd.concat((min_out, d), axis=1)
        min_out.to_csv(dir + '/' + 'minRecord.csv', index=False)
        mean = pd.DataFrame(mean)
        mean_out = pd.DataFrame(mean[['dataName']].drop_duplicates())
        mean_out.index = mean_out[['dataName']]
        for a_name, a_data in mean.groupby('algorithm'):
            d = a_data[['cpuTime']]
            d.index = a_data[['dataName']]
            d.columns = [a_name]
            mean_out = pd.concat((mean_out, d), axis=1)
        mean_out.to_csv(dir + '/' + 'meanRecord.csv', index=False)

    def memory(self):
        objectDir = os.path.join(self.data_dir, TaskType.memory.value)
        record = []
        max = []
        min = []
        mean = []

        for algorithm in os.listdir(objectDir):
            data = pd.read_csv(os.path.join(objectDir, algorithm))
            statistics = []
            for d_name, d_data in data.groupby('dataName'):
                mean.append({'algorithm': algorithm.split('.')[0],
                             'dataName': d_name,
                             TaskType.memory.value: d_data[[TaskType.memory.value]].mean().values[0]})
                d_statistics = []
                for p_name, p_data in d_data.groupby('parameter'):
                    d_statistics.append({'algorithm': algorithm.split('.')[0],
                                         'dataName': d_name,
                                         'parameter': p_name,
                                         TaskType.memory.value: p_data[[TaskType.memory.value]].mean().values[0]})
                d_statistics = sorted(d_statistics, key=lambda x: x[TaskType.memory.value], reverse=True)
                max.append(d_statistics[0])
                min.append(d_statistics[-1])
                statistics.extend(d_statistics)
            record.extend(statistics)
        dir = os.path.join(self.base_path, TaskType.memory.value)
        os.makedirs(dir, exist_ok=True)
        pd.DataFrame(record).to_csv(dir + '/' + 'record.csv', index=False)
        max = pd.DataFrame(max)
        max_out = pd.DataFrame(max[['dataName']].drop_duplicates())
        max_out.index = max_out[['dataName']]
        for a_name, a_data in max.groupby('algorithm'):
            d = a_data[['memory']]
            d.index = a_data[['dataName']]
            d.columns = [a_name]
            max_out = pd.concat((max_out, d), axis=1)
        max_out.to_csv(dir + '/' + 'maxRecord.csv', index=False)
        min = pd.DataFrame(min)
        min_out = pd.DataFrame(min[['dataName']].drop_duplicates())
        min_out.index = min_out[['dataName']]
        for a_name, a_data in min.groupby('algorithm'):
            d = a_data[['memory']]
            d.index = a_data[['dataName']]
            d.columns = [a_name]
            min_out = pd.concat((min_out, d), axis=1)
        min_out.to_csv(dir + '/' + 'minRecord.csv', index=False)
        mean = pd.DataFrame(mean)
        mean_out = pd.DataFrame(mean[['dataName']].drop_duplicates())
        mean_out.index = mean_out[['dataName']]
        for a_name, a_data in mean.groupby('algorithm'):
            d = a_data[['memory']]
            d.index = a_data[['dataName']]
            d.columns = [a_name]
            mean_out = pd.concat((mean_out, d), axis=1)
        mean_out.to_csv(dir + '/' + 'meanRecord.csv', index=False)

    def evaluation(self):
        statistic_data = []
        algorithmPath = os.path.join(self.data_dir, 'evaluation')
        for algorithm in os.listdir(algorithmPath):
            dataPath = os.path.join(algorithmPath, algorithm)
            for dataName in os.listdir(dataPath):
                evaluationPath = os.path.join(dataPath, dataName)
                for evaluationName in os.listdir(evaluationPath):
                    data = pd.read_csv(os.path.join(evaluationPath, evaluationName))
                    for c_name, c_data in data.groupby('pred_class'):
                        for p_name,p_data in c_data.groupby('parameter'):
                            statistic_data.append({'algorithm': algorithm,
                                                   'parameter':p_name,
                                                   'dataName': dataName,
                                                   'eval': evaluationName.split('.')[0],
                                                   'class': c_data[['class']].mean().values.tolist()[0],
                                                   'pred_class': c_name,
                                                   'class_diff': abs(c_name - c_data[['class']].mean().values.tolist()[0]),
                                                   'max': p_data[['value']].max().values.tolist()[0],
                                                   'min': p_data[['value']].min().values.tolist()[0],
                                                   'mean': p_data[['value']].mean().values.tolist()[0],
                                                   'std': p_data[['value']].std().values.tolist()[0]})
        data = pd.DataFrame(statistic_data)
        for d_name, d_data in data.groupby('dataName'):
            eval_base_dir = os.path.join(self.base_path, 'evaluation', d_name)
            os.makedirs(eval_base_dir, exist_ok=True)
            for e_name, e_data in d_data.groupby('eval'):
                out = e_data.sort_values(by=['class_diff', 'mean'], ascending=[True, False])
                out[['algorithm','parameter','class', 'pred_class', 'max','min','mean','std']].to_csv(eval_base_dir + '/' + e_name + '.csv',
                                                                          index=False)


if __name__ == '__main__':
    # basis = Basis(os.listdir('G:\Experiment'))
    # result = Result(basis)
    # result.statistic_memory()
    # result.statistic_cpu_time()
    # result.compute_evaluation()
    statistic = Statistics()
    statistic.evaluation()
    statistic.memory()
    statistic.cpuTime()
    print('处理完成')
