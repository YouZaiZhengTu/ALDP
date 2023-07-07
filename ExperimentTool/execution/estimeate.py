import os

from ExperimentTool.Entity.Estimate import Basis, Result, Statistics
from sklearn.metrics import rand_score, adjusted_rand_score, normalized_mutual_info_score


def calculate_estimate_value(algorithms: list):
    basis = Basis(algorithms)
    result = Result(basis)
    result.statistic_memory()
    result.statistic_cpu_time()
    result.compute_evaluation()
    print('处理完成')


def statistics_data():
    statistic = Statistics()
    statistic.evaluation()
    statistic.memory()
    statistic.cpuTime()
    print('处理完成')

if __name__ == '__main__':
    list = os.listdir('G:/Experiment')
    calculate_estimate_value(['streaKHC','GIRNCH'])
    statistics_data()
