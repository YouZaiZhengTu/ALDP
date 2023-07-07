'''
2022/11/23 21:42 Administrator
description: 将数据集内数据加以扰动使其满足三角不等式
'''
import numpy as np
import pandas as pd


def perturbed_data(data, path):
    data = np.array(data)
    for i in range(data.shape[1] - 1):
        data[:, i] += 0.00000001 * (i + 1)
    pd.DataFrame(data).to_csv(path, index=False, header=False)


if __name__ == '__main__':
    perturbed_data(pd.read_csv('../../data/big/covtype.csv'), '../../data/big/Covtype.csv')
    print(0)
