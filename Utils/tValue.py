import os

import pandas as pd
from scipy import stats
import numpy as np

def ttest_for_networks():
    datanames = ['beach','mfeat-netsci','jazz','naveford']
    methods = ['RS', 'GN']
    f = open('/Users/wenboxie/Documents/Manuscript/rs-md/RS-IS-reply01/ttest_networks.txt','r')

    m = 0
    rvs1 = []
    rvs2 = []
    for line in f.readlines():
        dim = line.strip('/n').split(',')[1:-1]
        dim = list(filter(lambda x: x and x != '--',dim))
        # print(dim)
        if m % 2 == 0:
            rvs1 = []
            for i in range(len(dim)):
                rvs1.append(float(dim[i]))
            m = m + 1
            continue
        else:
            rvs2 = []
            for i in range(len(dim)):
                rvs2.append(float(dim[i]))

        t_test = stats.ttest_rel(rvs1, rvs2)
        print(datanames[int(m/2)], t_test)
        m = m + 1


def ttest_for_UCI():
    f = open('/Users/wenboxie/Desktop/t-test.csv', 'r')
    methods = ['ET', 'GA', 'CURE', 'Chameleon','PERCH','SCC','ET(100)']
    m = 0
    rvs1 = []
    rvs2 = []
    for line in f.readlines():
        dim = line.strip('/n').split(',')[1:-1]
        dim = list(filter(lambda x: x and x != '--', dim))
        # print(dim)
        if m % 7 == 0:
            rvs1 = []
            for i in range(len(dim)):
                rvs1.append(float(dim[i]))
            m = m + 1
            continue
        else:
            rvs2 = []
            for i in range(len(dim)):
                rvs2.append(float(dim[i]))

        t_test = stats.ttest_rel(rvs1, rvs2)
        print( methods[m % 7], t_test)
        m = m + 1

if __name__ == '__main__':
    # ttest_for_UCI()
    # ttest_for_networks()
