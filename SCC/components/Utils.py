'''
2022/11/25 16:02 Administrator
description: 用于数据读取和数据预处理，可进行相应配置
'''
import pandas as pd
from sklearn.preprocessing import MinMaxScaler



def data_processing(filePath: str,  # 要读取的文件位置
                    minMaxScaler=True,  # 是否进行归一化
                    drop_duplicates=True,  # 是否去重
                    shuffle=True,  # 是否进行数据打乱
                    label_digit=True):  # 是否标签数字化
    data = pd.read_csv(filePath, header=None)
    if drop_duplicates:
        data = data.drop_duplicates().reset_index(drop=True)
    if shuffle:
        data = data.sample(frac=1).reset_index(drop=True)
    if minMaxScaler:
        data.iloc[:, :-1] = MinMaxScaler().fit_transform(data.iloc[:, :-1])
    if label_digit:
        if data.dtypes.values[-1] != 'int64':
            for index, label in enumerate(data.iloc[:, -1].drop_duplicates().reset_index(drop=True)):
                data.iloc[:, -1] = data.iloc[:, -1].replace(label, index)
    return data.iloc[:, :-1].values.tolist(), data.iloc[:, -1].values.tolist()


# 2022/11/27 14:37 陈斌
# description: 遍历树获取分配结果
def get_assignment(data, true_k):
    pred = None
    flag = True
    for i in range(len(data)):
        if len(list(set(data[i].cluster_assignments))) <= true_k:
            flag = False
            pred = data[i].cluster_assignments
            break
    if flag:
        pred = data[-1].cluster_assignments
    return pred


# 2022/11/27 14:50 陈斌
# description: 对聚类结果进行评估



if __name__ == '__main__':
    data, label = data_processing('../../../data/small/ecoli.txt')
    print(0)
