import math
import os
from time import time
import igraph as ig
import numpy as np
from sklearn.neighbors import NearestNeighbors

from ExperimentTool.Entity.Basis import Record, Task, RecordType
from ExperimentTool.Entity.Running import ExpMonitor
from FilesHandler import data_processing, print_estimate


class Node():
    def __init__(self, id, data, label):
        # 基础数据
        self.id = id  # 节点id
        self.data = data  # 节点数据
        self.label = label  # 节点标签
        self.label_pred = 0  # 预测标签
        self.children = {}
        self.root = {}

    def addChild(self, iterIndex, node):
        if iterIndex in self.children.keys():
            if node.id in self.children[iterIndex].keys():
                return False
            else:
                self.children[iterIndex][node.id] = node
        else:
            self.children[iterIndex] = {}
            self.children[iterIndex][node.id] = node
        return True

    def setRoot(self, iterIndex, node):
        self.root[iterIndex] = node

    def __repr__(self):
        return str(self.id)


def generatePos(length: int) -> list[int]:
    position = []
    mid = math.ceil(length / 2)
    position.append(mid)
    while mid > 3:
        diff = math.ceil(mid / 2)
        position.append(position[-1] + diff)
        position.insert(0, position[0] - diff)
        mid = diff
    arr = np.array(position)
    return arr[(arr > 0) & (arr < length)]


class ANLDP():
    def __init__(self, data, label):
        self.nodes = {}
        for i in range(len(data)):
            self.nodes[i] = Node(i, data[i], label[i])
        self.label = np.array([-1 for i in range(len(data))])
        self.pred_label = []

    def getNearestNeighbor(self, roots: dict[Node]):
        ids, datas = {}, []
        first, second = {}, {}
        nodes = list(roots.keys())
        for i, key in enumerate(nodes):
            ids[key] = i
            datas.append(roots[key].data)
        neighbors = NearestNeighbors(n_neighbors=3).fit(datas)
        dist = neighbors.kneighbors(datas, return_distance=False).tolist()
        for key in roots:
            first[key] = nodes[dist[ids[key]][1:][0]]
            second[key] = nodes[dist[ids[key]][2:][0]]
        return first, second

    def getNodeOfSct(self, root: Node, iterIndex):
        leaves = {}
        candidates = [root]
        while candidates:
            r = candidates.pop(0)
            leaves[r.id] = r
            for key, value in r.children[iterIndex].items():
                if key not in leaves.keys():
                    leaves[key] = value
                    candidates.append(value)
        for leaf in leaves.values():
            leaf.setRoot(iterIndex, root)
        return leaves

    def constructGraph(self, roots: dict[Node], nearestNeighborIndex, iterIndex):
        candidates = []
        # start = time()
        for key, value in nearestNeighborIndex.items():
            res0 = roots[key].addChild(iterIndex, roots[value])
            res1 = roots[value].addChild(iterIndex, roots[key])
            if not res1:
                candidates.append(roots[key])
        # print(time()-start)
        scts = {}
        rnns = {}
        treeNode = []
        for candidate in candidates:
            leaves = self.getNodeOfSct(candidate, iterIndex)
            treeNode.extend(list(leaves.values()))
            if len(leaves) > 2:
                scts[candidate.id] = {'num': len(leaves), 'leaves': leaves}
            else:
                rnns[candidate.id] = {'num': len(leaves), 'leaves': leaves}
        if len(treeNode) < len(roots):
            diff = np.setdiff1d(np.array([n.id for n in roots.values()]),np.array([n.id for n in treeNode])).tolist()
            while diff:
                candidate = roots[diff[0]]
                leaves = self.getNodeOfSct(candidate, iterIndex)
                if len(leaves) > 2:
                    scts[candidate.id] = {'num': len(leaves), 'leaves': leaves}
                else:
                    rnns[candidate.id] = {'num': len(leaves), 'leaves': leaves}
                diff = np.setdiff1d(np.array(diff),np.array([n.id for n in leaves.values()])).tolist()
        # start = time()
        # graph = ig.Graph(n=len(roots), edges=[(k, v) for k, v in nearestNeighborIndex.items()])
        # graph.vs['id'] = [i for i in roots.keys()]
        # sonGraph = graph.decompose(mode='weak')
        # print(time() - start)
        return rnns, scts

    def refactorRnn(self, rnns, scts, second, iterIndex):
        newEdges = []
        Scts = scts.copy()
        Scts.update(rnns)
        for key, rnn in rnns.items():
            rnnSecondNeighbor = {}
            for id, n in rnn['leaves'].items():
                rnnSecondNeighbor[id] = {'neighbor': self.nodes[second[id]],
                                         'root': self.nodes[second[id]].root[iterIndex].id,
                                         'num': len(
                                             list(Scts[self.nodes[second[id]].root[iterIndex].id]['leaves'].values()))}
            # 方案1：选择点数量最少的连接,如果重复就距离最小连接
            # print(key)
            if len(rnnSecondNeighbor) > 1:
                rnn1, rnn0 = list(rnnSecondNeighbor.keys())[0], list(rnnSecondNeighbor.keys())[1]
                neighbor1, neighbor0 = rnnSecondNeighbor[rnn0]['neighbor'], rnnSecondNeighbor[rnn1]['neighbor']
                if neighbor0.id == neighbor1.id:
                    newEdges.append((key, rnnSecondNeighbor[rnn0]['root']))
                else:
                    # if rnnSecondNeighbor[rnn0]['num'] == rnnSecondNeighbor[rnn1]['num']:
                    # dist0 = np.linalg.norm(np.array(self.nodes[rnn0].data) - np.array(neighbor0.data))
                    # dist1 = np.linalg.norm(np.array(self.nodes[rnn1].data) - np.array(neighbor1.data))
                    # if dist0 > dist1:
                    #     newEdges.append((key, rnnSecondNeighbor[rnn1]['root']))
                    # elif dist0 < dist1:
                    #     newEdges.append((key, rnnSecondNeighbor[rnn0]['root']))
                    # else:
                    #     newEdges.append((key, rnnSecondNeighbor[rnn1]['root']))
                    if rnnSecondNeighbor[rnn0]['num'] <= rnnSecondNeighbor[rnn1]['num']:
                        newEdges.append((key, rnnSecondNeighbor[rnn0]['root']))
                    else:
                        newEdges.append((key, rnnSecondNeighbor[rnn1]['root']))
            else:
                newEdges.append((key, rnnSecondNeighbor[list(rnnSecondNeighbor.keys())[0]]['root']))
        for edge in newEdges:
            if edge[0] == edge[1]:
                our = Scts.pop(edge[0])['leaves']
                Scts[edge[0]] = {'num': len(our), 'leaves': our}
            elif edge[1] in Scts.keys():
                our = Scts.pop(edge[0])['leaves']
                obj = Scts.pop(edge[1])['leaves']
                obj.update(our)
                Scts[edge[1]] = {'num': len(obj), 'leaves': obj}
        return Scts

    def detectRoot(self, scts: dict, iterIndex):
        roots = []
        for key, sct in scts.items():
            distance_metrics = []
            data = list(sct['leaves'].values())
            score = {}
            for d in data:
                score[d.id] = 0
            for i in range(len(data)):
                for j in range(i + 1, len(data)):
                    distance_metrics.append(dict(x=data[i].id, y=data[j].id,
                                                 dist=np.linalg.norm(np.array(data[i].data) - np.array(data[j].data))))
            distance_metrics.sort(key=lambda d: d['dist'])
            for pos in generatePos(len(distance_metrics)):
                for dist in distance_metrics[:pos]:
                    score[dist['x']] += 1
                    score[dist['y']] += 1
            score = sorted(score.items(), key=lambda e: e[1], reverse=True)
            root = sct['leaves'][score[0][0]]
            if len(score) > 1:
                if score[0][1] == score[1][1]:
                    if len(sct['leaves'][score[0][0]].children[iterIndex].values()) >= \
                            len(sct['leaves'][score[1][0]].children[iterIndex].values()):
                        root = sct['leaves'][score[0][0]]
                    else:
                        root = sct['leaves'][score[1][0]]
            else:
                root = sct['leaves'][score[0][0]]
            roots.append({'root': root, 'leaves': sct['leaves']})
        return roots

    def distribution(self, clusters: list[dict]):
        roots = {}
        for cluster in clusters:
            for leaf in cluster['leaves'].values():
                self.label[leaf.id] = cluster['root'].id
            self.label[cluster['root'].id] = -1
            roots[cluster['root'].id] = cluster['root']
        return roots

    def getLabels(self):
        pred_label = [-1 for i in range(self.label.shape[0])]
        for origin in np.where(self.label == -1)[0]:
            leaves = [origin]
            while leaves:
                r = leaves.pop(0)
                pred_label[r] = origin
                leaves.extend(np.where(self.label == r)[0].tolist())
        return pred_label

    def clustering(self, K):
        roots = self.nodes
        iterIndex = 0
        while len(roots) > K and len(roots) > 3:
            first, second = self.getNearestNeighbor(roots)
            rnns, scts = self.constructGraph(roots, first, iterIndex)
            scts = self.refactorRnn(rnns, scts, second, iterIndex)
            roots = self.detectRoot(scts, iterIndex)
            roots = self.distribution(roots)
            self.pred_label.append(self.getLabels())
            iterIndex += 1

# @ExpMonitor(expId='ALDPOPT', algorithmName='ALDPOPT', storgePath='G:/Experiment')
def run(task, **kwargs):
    record = Record()
    data, label, K = data_processing(task.filePath, shuffle=False)
    start = time()
    anldp = ANLDP(data, label)
    anldp.clustering(K)
    end = time()
    print_estimate(label, label, task.dataName, task.iterIndex, 0, end - start)
    for i, assignment in enumerate(anldp.pred_label):
        # print_estimate(label, assignment, task.dataName, task.iterIndex, i, end - start)
        record.save_output(RecordType.assignment, label, assignment, i)
    record.save_time(end - start)
    return {'record': record}


if __name__ == '__main__':
    path = '../../../data/random'
    for file in os.listdir(path):
        for testIndex in range(5):
            task = Task(str('NoParam'), testIndex, file.split('.')[0], path + '/' + file)
            run(task=task)
