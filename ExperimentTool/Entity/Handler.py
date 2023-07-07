import os.path

import pandas as pd
from tqdm import tqdm


class Handler():
    def __init__(self, outDirPath: str, dirpath: str, algorithmName: str, parameter: str):
        self.outDirPath = outDirPath
        self.dataDirPath = 'G:/Project/Python/hierarchical-clustering/data/all'
        self.dirpath = dirpath
        self.algorithmName = algorithmName
        self.parameter = parameter
        self.dirFolderPath = os.path.join(self.outDirPath, self.algorithmName, self.algorithmName,self.parameter)

    def lineOutput(self, path, fileName, data: dict):
        os.makedirs(path, exist_ok=True)
        outputPath = os.path.join(path, fileName + '.csv')
        if not os.path.exists(outputPath):
            pd.DataFrame(data, index=[0]).to_csv(outputPath, index=False, mode='a')
        else:
            pd.DataFrame(data, index=[0]).to_csv(outputPath, index=False, header=False, mode='a')

    def handle_time(self):
        path = os.path.join(self.dirpath, 'time')
        if os.path.exists(path):
            bar = tqdm(os.listdir(path))
            for item in bar:
                bar.set_description("正在处理时间数据")
                data = None
                try:
                    data = pd.read_csv(os.path.join(path, item), header=None,encoding="ISO-8859-15").iloc[:,0].values.tolist()
                    for i, d in enumerate(data):
                        self.lineOutput(os.path.join(self.dirFolderPath, str(i), 'cpuTime'), 'cpuTime',
                                        {'dataName': item.split('.')[0], 'cpuTime': d})
                except Exception as e:
                    print(e, type(e))
                    if (isinstance(e, pd.errors.EmptyDataError)):
                        print("time file is empty, please check it, file name is " + item)

        else:
            print('time directory not exists,please check it')
    def handle_result(self):
        path = os.path.join(self.dirpath, 'result')
        if os.path.exists(path):
            bar = tqdm(os.listdir(path))
            for item in bar:
                bar.set_description("正在处理结果数据")
                data = None
                try:
                    iter = item.split('$')[0]
                    file = item.split('$')[1]
                    label_true = None
                    label_pred = pd.read_csv(os.path.join(path, item), header=None, encoding="ISO-8859-15").iloc[:,0].values.tolist()
                    for origin in os.listdir(self.dataDirPath):
                        if origin.split('.')[0] == file.split('.')[0]:
                            label_true = pd.read_csv(os.path.join(self.dataDirPath, origin), header=None, encoding="ISO-8859-15").iloc[:,-1].values.tolist()
                            break
                    if label_true is None:
                        print('can not find label_true file, please check it, file name is ' + file.split('.')[0])
                        continue
                    out = os.path.join(self.dirFolderPath, iter, 'output',file.split('.')[0],'assignment')
                    os.makedirs(out, exist_ok=True)
                    pd.DataFrame({'label_true':label_true,'label_pred':label_pred}).to_csv(os.path.join(out,'0.csv'),index=False)
                except Exception as e:
                    print(e, type(e))
                    if (isinstance(e, pd.errors.EmptyDataError)):
                        print("time file is empty, please check it, file name is " + item)

        else:
            print('time directory not exists,please check it')
    def execution(self):
        self.handle_result()
        self.handle_time()


if __name__ == '__main__':
    handeler = Handler('G:/Experiment', 'G:/Project/C/Munec/1/munecsource', 'Munec', 'noParameter')
    handeler.execution()
