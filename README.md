# GRAPHS HOMEWORK


## Part 1: Pytorch Geometric Installation

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


## Part 2: Dataset exploration (0.5 points)

In this part you will be downloading and exploring the OGBN-Proteins dataset. Following the instructions from the official website produce a script (And name it ogbn_proteins_explore.py) that will download and explore the dataset (Use some of the functions learned in the tutorial). To get the points you have to append a table in your report with descriptive features about the dataset and answer the following questions:

What is the task for this dataset?
How are the graph features organized?
Which is the evaluation metric of this dataset and how is it analytically calculated?
