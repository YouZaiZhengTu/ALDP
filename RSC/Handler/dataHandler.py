from pyecharts import options as opts
from sklearn import metrics

from Entities.Node import Node


# 将数据转为数据对象
# input:dict {data,label}
# output: list[Node]
def data_to_object(dataArray):
    data = dataArray['data']
    label = dataArray['label']
    nodeList = []
    for i in range(0, len(data)):
        nodeList.append(Node(i, data[i], label[i], 0))
    return nodeList


# 将数据对象的数据提取用于kdTree计算
# input: list[Node]
# output: list: data
def extract_data_from_Node(nodeList: list[Node]):
    dataList = []
    for node in nodeList:
        dataList.append(node.data)
    return dataList


# 将数据对象转换为可视化的数据格式
# input Node
# output dict{TreeItem}
colorList = ['#222831', '#00adb5', '#6a2c70', '#ff2e63', '#3f72af',
             '#3490de', '#6639a6', '#d59bf6', '#52616b', '#00adb5']

def tree_traversal(root):
    children = []
    root.isVisited = True
    for key in root.adjoinList:
        n = root.adjoinList[key]
        if n.isVisited:
            continue
        else:
            left = opts.TreeItem(name=str(n.label),  children=tree_traversal(n),value = n.label,
                    label_opts = opts.LabelOpts(position='bottom', color=colorList[n.iteration]))
        children.append(left)
    return children

def extract_tree(root):
    return opts.TreeItem(name=str(root.label),  children=tree_traversal(root),value = root.label,
                    label_opts = opts.LabelOpts(position='bottom', color=colorList[root.iteration]))

# 生成对应的label_true
def init_label_true(labels):
    label = list(set(labels))
    labelskey = {}
    for i in range(len(label)):
        labelskey[label[i]] = i
    label_true = [-1 for i in range(len(labels))]
    for i in range(len(labels)):
        label_true[i] = labelskey[labels[i]]
    return label_true

# 生成对应的预测结果
def init_label_pred(roots:list[Node],label_pred):
    for i in range(len(roots)):
        current = [roots[i]]
        isVisited = []
        while current:
            n = current.pop(0)
            isVisited.append(n.id)
            if n.id < len(label_pred):
                label_pred[n.id] = i
            for key in n.adjoinList:
                if key not in isVisited:
                    current.append(n.adjoinList[key])
    return label_pred

# 计算调整兰德指数
def compute_adjust_rand_index(label_true,label_pred,iteration,categorysize):
    adjust_rand_index = metrics.adjusted_rand_score(label_true, label_pred)
    print("第%d次聚类，当前共%d个类,调整兰德系数为%f"%(iteration,categorysize,adjust_rand_index))
