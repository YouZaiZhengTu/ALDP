from ExperimentTool.visualization.Basis import initTask
from ExperimentTool.visualization.Line import Line

if __name__ == '__main__':
    tasks = initTask()
    for task in tasks:
        line = Line(task)
        line.draw(['EXP100'])
