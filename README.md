# Cost-effective hierarchical clustering with local density peak detection

* Type: Implementation
* Group1: School of Computer Science, Southwest Petroleum University, Chengdu 610500, China
* Group1: School of Computer and Software Engineering, Xihua University, Chengdu 610039, China
* Author: Wen-Bo Xie1⋆, Bin Chen1, Xun Fu1, Jun-Hao Shi1, Yan-Li Lee2, and Xin Wang1
* Year: 2023
---

This is a project running the Aldp (short for clustering by Aggregating Local Density Peaks) algorithm.

### Environment

Developed on windows 11 using Python 3.9.

Be sure to install the dependencies before executing the project code. 
```
python3 -m pip install -r requirements.txt
```

### How to use

Our proposed algorithms are in the aldp and ALDPnp folders, 
in addition ExperimentTool folder is an experimental testing tool, 
Utils folder is a data reading and processing tool, 
the data folder holds the dataset and the rest of the files are comparison algorithms.
A main.py file is included in each algorithm's folder to run the code, 
and the output files are stored in the storgePath
A skeleton of typical clustering program is like this: 
```python
from ExperimentTool.Entity.Basis import Record, RecordType, Task
from ExperimentTool.Entity.Running import ExpMonitor
from Utils.FilesHandler import data_processing
# Record, save and return the output
# Task, sets the run parameters
# ExpMonitor, experimental hosting
# data_processing, data pre-processing and reading
@ExpMonitor(expId='HAC', algorithmName='HAC', storgePath=os.pardir+'/Experiment')
# Using decorator functions to host running code
def run(task: Task,k, **kwargs):
    record = Record()
    data, label, K = data_processing(task.filePath)
    start = time()
    clustering = AgglomerativeClustering(n_clusters=k, linkage='average').fit(data)
    end = time()
    record.save_time(end - start)
    label_pred, label_true = construct_tree(clustering, label)
    record.save_output(RecordType.tree, label_true, label_pred)
    record.save_output(RecordType.assignment, label, clustering.labels_)
    return {'record': record}
```

We also provide a tool for data analysis in the ExperimentTool\execution\estimate.py file
```python
    list = os.listdir('G:/Experiment') # Enter the catalogue of experimental data
    calculate_estimate_value(['streaKHC','GIRNCH'])  #Inside the list are the names of the algorithms to be analysed
    statistics_data() # output
```


