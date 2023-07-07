import os
import time

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from ExperimentTool.Entity.Basis import Record, RecordType, Task
from ExperimentTool.Entity.Running import ExpMonitor
from Utils.FilesHandler import data_processing, print_estimate


def autoselect_dc(max_id, max_dis, min_dis, distances):
    dc = (max_dis + min_dis) / 2

    while True:
        nneighs = sum([1 for v in distances if v < dc]) / max_id ** 2
        if nneighs >= 0.01 and nneighs <= 0.02:
            break
        # binary search
        if nneighs < 0.01:
            min_dis = dc
        else:
            max_dis = dc
        dc = (max_dis + min_dis) / 2
        if max_dis - min_dis < 0.0001:
            break
    return dc


class DPC(object):
    def __init__(self, center_rate=0.6, noise_rate=0.1, dc=None, auto_filt=False):
        self.distance = None
        self.dc = None
        self.rho = None
        self.sigma = None
        self.cluster = None
        self.center = None
        self.tup = None
        self.center_rate = center_rate
        self.noise_rate = noise_rate
        self.auto_filt = auto_filt
        if dc != None:
            self.dc = dc

    def info_cal(self, X):
        distance = euclidean_distances(X)
        if self.dc == None:
            self.dc = autoselect_dc(distance.shape[0], np.max(distance), np.min(distance), distance.flatten())
        rdistance = distance - self.dc
        rho = np.array([rdistance[rdistance[idx] < 0].shape[0] for idx in range(rdistance.shape[0])])
        sigma = np.array(
            [np.min(distance[idx, (rho > value)]) if value < np.max(rho) else np.max(distance[idx]) for idx, value in
             enumerate(rho)])
        map_idx = np.array([np.argwhere(
            distance[idx] == np.min(distance[idx, (rho > value)])).flatten() if value < np.max(rho) else np.argmax(
            distance[idx]).flatten() for idx, value in enumerate(rho)])
        map_idx = np.array([arr[0] for arr in map_idx])
        idx = np.argsort(rho)
        tup = np.array([i for i in zip(rho[idx], sigma[idx], idx)])
        self.distance, self.dc, self.rho, self.sigma = distance, self.dc, rho, sigma
        return tup, map_idx

    def remove_noise(self, cluster, tup, map_idx):
        y_pred = np.array([cluster[idx] for idx in range(map_idx.shape[0])])
        distance = self.distance
        rho = self.rho
        group = {}
        for i in list(set(y_pred)):
            group[i] = np.argwhere(y_pred == i).flatten()
        for label in group.keys():
            max_rho = -1
            for i in group[label]:
                for idx, dis in enumerate(distance[i]):
                    if idx not in group[label] and distance[i, idx] < self.dc and max_rho < rho[i]:
                        max_rho = rho[idx]
                        break
            if max_rho != -1:
                for i in group[label]:
                    if self.rho[i] < max_rho:
                        cluster[i] = -1
        return cluster

    def fit_transform(self, X):
        tup, map_idx = self.info_cal(X)
        self.tup = tup
        origin_center_idx = tup[(tup[:, 0] > np.max(tup[:, 0]) * self.center_rate) & (
                tup[:, 1] > np.max(tup[:, 1]) * self.center_rate)][:, 2].astype("int64")
        self.center = origin_center_idx
        cluster = {}
        for idx, center in enumerate(origin_center_idx):
            cluster[center] = idx
        if not self.auto_filt:
            origin_noise_idx = tup[(tup[:, 0] < np.max(tup[:, 0]) * self.noise_rate)][:, 2].astype("int64")
            for center in origin_noise_idx:
                cluster[center] = -1
        for density, distance, idx in tup[::-1]:
            idx = int(idx)
            if idx in cluster.keys():
                continue
            if map_idx[idx] in cluster.keys():
                cluster[idx] = cluster[map_idx[idx]]
            else:
                cluster[idx] = -1
        if self.auto_filt:
            cluster = self.remove_noise(cluster, tup, map_idx)
        self.cluster = cluster
        y_pred = np.array([cluster[idx] for idx in range(X.shape[0])])
        return y_pred


@ExpMonitor(expId='DPC', algorithmName='DPC', storgePath=os.pardir + '/Experiment')
def run(task, **kwargs):
    record = Record()
    data, label_true, K = data_processing(task.filePath)
    start = time.time()
    dpc = DPC(0.55, 0.1)
    label_pred = dpc.fit_transform(data)
    end = time.time()
    record.save_output(RecordType.assignment, label_true, label_pred, 0)
    print_estimate(label_true, label_pred, task.dataName, task.iterIndex, 0, time.time() - start)
    record.save_time(end - start)
    return {'record': record}


if __name__ == '__main__':
    path = '../data/small'
    for file in os.listdir(path):
        for testIndex in range(5):
            task = Task(str('NoParam'), testIndex, file.split('.')[0], path + '/' + file)
            run(task=task)
