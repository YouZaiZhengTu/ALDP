import os


class Task():
    def __init__(self, type: str, dataName, filePath, baseDir='G:/Result/', ):
        self.baseDir = baseDir
        self.filePath = filePath
        self.type = type
        self.dataName = dataName


def initEvalTask(baseDir='G:/Result/Statistics',
                 evaluation='evaluation'):
    evaluationDir = os.path.join(baseDir, evaluation)
    tasks = []
    for dataName in os.listdir(evaluationDir):
        evalPath = os.path.join(evaluationDir, dataName)
        for evalName in os.listdir(evalPath):
            tasks.append(Task(evalName.split('.')[0], dataName, os.path.join(evalPath, evalName)))
    return tasks


if __name__ == "__main__":
    tasks = initEvalTask()
