# FedTail-DT

A Dual-Teacher Framework for Long-Tailed Heterogeneous FL

## Requirements
### Installation
Create a conda environment and install dependencies:
```python
conda env create -f env_cuda_latest.yaml

```
### Dataset
Here we provide the implementation on  Cifar10, Cifar100 The  Cifar10 and Cifar100 datasets will be automatically downloaded in your datadir. 




### Parameters
| Parameter        | Description                                                                                           |
|------------------|-------------------------------------------------------------------------------------------------------|
| `m`              | The model architecture. Options: `dnn`, `mobilenetv2`.                                                |
| `lbs`            | Local batch size.                                                                                     |
| `lr`             | Learning rate.                                                                                        |
| `nc`             | Number of clients.                                                                                    |
| `jr`             | The client joining rate.                                                                              |
| `nb`             | Number of classes.                                                                                    |
| `data`           | Dataset to use. Options:  `cifar10 `, `cifar100`.                            |
| `algo`           | Algorithm to use.                                                                                     |
| `gr`             | Global_rounds.                                                                                        |
| `did`            | Device id.                                                                                            |
| `partition`      | The data partitioning strategy.                                                                       |
| `al`             | The Dirichlet distribution coefficient.                                                               |
| `ls`             | local_epochs.                                                                                         |

### Usage
Here is an example to run FedVLS on CIFAR10 with mobilenetv2:
```
python -u main.py -lbs 64 -nc 20 -jr 0.4 -nb 10 -data cifar10 -m mobilenetv2 -algo FedTail-DT -gr 100 -did 0 -aug True -lr 0.01 -partition noniid -al 0.1 -ls 10 -ed 1e-5
```





