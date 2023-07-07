"""
 Copyright (c) 2021 The authors of SCC All rights reserved.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import math
import os
import time

import numpy as np
from absl import logging

from SCC.components.Utils import get_assignment
from SCC.components.scc import SCC, graph_from_vectors
from ExperimentTool.Entity.Basis import Task, Record, RecordType
from ExperimentTool.Entity.Running import ExpMonitor
from Utils.FilesHandler import data_processing

logging.set_verbosity(logging.INFO)


def SCCAplation(data, label):
    X = np.array(data)
    start = time.time()
    upper = 1.0
    lower = 0.1
    num_rounds = math.ceil(math.log(X.shape[0], 2))
    graph = graph_from_vectors(X, k=600, batch_size=1000)
    taus = np.geomspace(start=upper, stop=lower, num=num_rounds)
    scc = SCC(graph, num_rounds, taus)
    scc.fit()
    end = time.time()
    runTime = end - start
    label_pred = get_assignment(scc.rounds, len(list(set(label))))
    return label_pred, runTime


@ExpMonitor(expId='SCC', algorithmName='SCC', storgePath=os.pardir + '/Experiment')
def run(task):
    record = Record()
    data, label,K = data_processing(task.filePath)
    label_pred, runTime = SCCAplation(data, label)
    # FilesHandler.print_estimate(label, label_pred, task.dataName, int(task.iterIndex), 0, runTime)
    record.save_time(runTime)
    record.save_output(RecordType.assignment, label, label_pred)
    return {'record': record}


if __name__ == "__main__":
    path = '../data/small/'
    for file in os.listdir(path):
        for testIndex in range(50):
            task = Task('noParam', testIndex, file.split('.')[0], path + '/' + file)
            run(task=task)
