from Entities.Node import Node


# 将处理后的数据转换为对象数组
def data_to_node(data, label):
    nodeList = []
    for i in range(len(label)):
        nodeList.append(Node(i, data[i], label[i], label[i]))
    return dict(originList=nodeList, true_k=len(list(set(label))))


# 读取文件数据转为数组
# return dict data and label
def read_file_to_array(filePath: str):
    with open(filePath, 'r') as f1:
        list1 = f1.readlines()
    for i in range(0, len(list1)):
        list1[i] = list1[i].rstrip('\n')
    data = [tuple(o.split(',')) for o in list1]
    data = list(set(data))
    labels = [o[-1] for o in data]
    datas = [o[:-1] for o in data]
    category = list(set(labels))
    l = {}
    for i in range(len(category)):
        l[category[i]] = chr(i + 96)
    for i in range(len(labels)):
        labels[i] = l[labels[i]]
    return {'data': datas, 'label': labels}
