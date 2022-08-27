# GRAPHS HOMEWORK


## PART 1: Pytorch Geometric Installation

Ensure that at least PyTorch 1.11.0 is installed:
`python -c "import torch; print(torch.__version__)"`

Find the CUDA version PyTorch was installed with:
`python -c "import torch; print(torch.version.cuda)"`


Install the relevant packages:

```$ pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
$ pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
$ pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
$ pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
$ pip install torch-geometric
```
where `${CUDA}` and `${TORCH}` should be replaced by the specific CUDA version `(cpu, cu102, cu113, cu115)` and PyTorch version `(1.11.0, 1.12.0)`, respectively. For example, for PyTorch 1.12.* and CUDA 11.6, type:

```
$ pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
$ pip install torch-sparse -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
$ pip install torch-cluster==latest+cu116 -f https://pytorch-geometric.com/whl/torch-1.12.0.html
$ pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.12.0.html
$ pip install torch-geometric
```

For more details or issues during installation go to: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html


## PART 2: Dataset exploration (1 Point)

In this part you will explore the [datasets](data). In this [article](https://arxiv.org/pdf/1909.05310.pdf) you will find a description of the three datasets available (BBBP, ESOL and FreeSolv). To get the complete point you have to append a table in your report with descriptive features about each one of the datasets and answer the following questions:

**What is the task for each one of the datasets?**

**How are the graph features organized?**

**Which is the evaluation metric of this dataset?**

## PART 3: Implementation (1 point)

In this part of the homework your task is to implement some state of the art GCN's. You will also dive inside the code and implement some missing lines only by modifying the [graph_conv.py](https://github.com/juanitapuentes/Graphs-Homeworks/blob/main/src/graph_conv.py). Please just modify the parts of the code that explicitly ask to do so, otherwise the whole method would stop working. You have to read the files inside this [folder](https://github.com/juanitapuentes/Graphs-Homeworks/tree/main/src). You do not have to understand everything but you need to have a general idea of what is happening inside the code. In your report you need to answer the following questions:

**Is this method evaluating the whole training set in every iteration?**

**How is the GCN aggregating information from neighboring nodes?**

**Explain in detail what process or algorithm is carried out in each of the lines that you added in the code. What would happen if these lines were not added? How would this affect the output of the network?**

## PART 4: Hyperparameter and Loss Function experimentation (2 points)

**At this point you must decide which of the dataset you want to use**

To run geo-GCN on MNISTSuperpixels with default parameters, go to `src` and use the command:

```python
python train_models.py MNISTSuperpixels
```
 
 To use chemical data:
 
 ```python
from torch_geometric.data import DataLoader
from chem import load_dataset

batch_size = 64
dataset_name = ...  # 'freesolv' / 'esol' / 'bbbp'

train_dataset = load_dataset(dataset_name, 'train')
val_dataset = load_dataset(dataset_name, 'val')
test_dataset = load_dataset(dataset_name, 'test')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# training loop
... 
```
It is important to note that this network does not work properly because it does not have a favorable loss function to solve this problem. Hence, your task is to modify the **loss function**. 

Then you need to choose some tunable hyperparameters and run at least **6 experiments**. For reference, it should take aproximately **one and a half** hours to run one experiment.

To have the complete points, you need to attach a table with all the experiments to your report. In addition, you should discuss how each of the hyperparameters that you modify affects the performance of the network. Finally, you must explain why the new loss function that you implemented improves (or not) the metrics.

## PART 5: Convolutional Layers experimentation (1 points)

Finally, you should choose the best model found in **PART 4** and experiment with the number of convolutional layers. You must perform at least 2 experiments. In the report attach the table with the results and discuss them. In addition, propose future enhancements to this GCN.


## BONUS: Build your own GCN (1 point)

In [cora](cora_data) you will find a citation network dataset that contains 2708 scientific publications and a total of 5429 links. Each publication is classified in one out of seven topics and is is described by the absence/presence of certain words from a dictionary. In [this link](https://graphsandnetworks.com/the-cora-dataset/) you will find more information about the dataset.

Based on what you learned in class and considering the model you already worked with implement your own node classification model with the cora dataset. Start by exploring the dataset, understand the elements that will make up the graphs in this problem and load the data in order to develop your model using GCNs. To get the total marks you will have to attach the model and the training archives to the [new_model](new_model) file and report your model's accuracy in the PDF to be uploaded.

# Report
Please upload to your repository a PDF file named `lastname_graphsHW.pdf`.
