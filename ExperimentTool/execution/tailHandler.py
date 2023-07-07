import os

import numpy as np
import pandas as pd
from arch.unitroot import KPSS
from scipy import stats


def statisticTest(path, out, target):
    for file in os.listdir(path):
        result = []
        type = file.split('.')[0]
        data = pd.read_csv(path + file).fillna(0)
        targetData = data[target]
        testData = data.drop(target, axis=1)
        for tag in targetData.columns:
            for test in testData.columns:
                t_value, p_value = stats.ttest_rel(targetData[tag], testData[test])
                result.append({'target': tag, 'test': test, 't_value': t_value, 'p_value': p_value})
        os.makedirs(out, exist_ok=True)
        pd.DataFrame(result).to_csv(out + type + '.csv', index=False)
    print('完成')

def statisticTest2(path, out, target):
    path = 'G:/science/project/Boosting Clustering By RNN scores/result/ri/'
    outPath = 'G:/science/project/Boosting Clustering By RNN scores/result/evaluation'
    algorithm = ['GIRNCH', 'HKMeans', 'Munec', 'SCC', 'streaKHC']
    for dataset in os.listdir(path):
        evaluation = os.path.join(path, dataset)
        outEvalPath = os.path.join(outPath, dataset)
        for eval in os.listdir(evaluation):
            data = pd.read_csv(os.path.join(evaluation, eval))
            result = data[data['algorithm'].isin(algorithm)]
            result = result.sort_values(by=['algorithm', 'pred_class'])
            os.makedirs(outEvalPath, exist_ok=True)
            result.to_csv(os.path.join(outEvalPath, eval), index=False)
    print('done')
def paramsTest(path,out,target):
    path = 'G:/Result/Statistics/evaluation'
    out = 'G:/science/project/中文期刊/实验结果'
    algorithm = ['RNNSearch']
    result = []
    for dataset in os.listdir(path):
        evaluation = os.path.join(path, dataset)
        for eval in os.listdir(evaluation):
            data = pd.read_csv(os.path.join(evaluation, eval))
            result = data[data['algorithm'].isin(algorithm)]
            result = result.sort_values(by=['parameter','pred_class'])
            os.makedirs(out, exist_ok=True)
            result.to_csv(os.path.join(out, eval), index=False)

def stabilityTest(dirPath,outPath):
    data = pd.read_csv(dirPath).values.tolist()
    out = []
    for d in data:
        type = d[0]
        dataName = d[1]
        kpss = KPSS(d[2:])
        out.append({'type':type,'data':dataName,'t-test':kpss.stat,'p-value':kpss.pvalue})
        # print(kpss.summary().as_text())
        # print('type:%s,data:%s,t-test:%0.4f,p-value:%0.4f,critical%s' % (type,dataName,kpss.stat, kpss.pvalue,kpss.critical_values))
    out = pd.DataFrame(out)
    for type,d in out.groupby('type'):
        d.to_csv(os.path.join(outPath,type+'stability.csv'),index=False)

if __name__ == '__main__':

    # paramsTest('','','')
    # statisticTest('G:/science/project/中文期刊/实验结果/tvalue/新建文件夹/新建文件夹/','G:/science/project/中文期刊/实验结果/tvalue/新建文件夹/结果/',['RNNSearch'])
    stabilityTest('G:\science\project\ALDP扩展\实验数据\参数平稳性检验\data.csv','G:\science\project\ALDP扩展\实验数据\参数平稳性检验')