# WCMC-SAPD
The folder includes the following python file for Fair Classification experiment:
1.algorithms.py
  including functions 
  SAPD-VR, this paper
  SMDA-VR, https://nips.cc/Conferences/2021/ScheduleMultitrack?event=26760
  SREDA, https://proceedings.neurips.cc/paper/2020/hash/ecb47fbb07a752413640f82a945530f8-Abstract.html
2.algorithms_CIFAR10.py
  including functions 
  SAPD-VR, this paper
  SMDA-VR, https://nips.cc/Conferences/2021/ScheduleMultitrack?event=26760
  SREDA, https://proceedings.neurips.cc/paper/2020/hash/ecb47fbb07a752413640f82a945530f8-Abstract.html
3.model.py
  including model class for MNIST and FMNIST for Fair Classification experiment
4.model_CIFAR10
  including model class for CIFAR10 for Fair Classification experiment
5.Opitmization_Method.py
  including a function that is the projection onto the unit simplex
6.functions.py
  including shadow error bar plot function


The folder includes the following python jupyter note book file:
1.main_MNIST.ipynb
  The main file for the Fair Classification for MNIST data set
2.main_FMNIST.ipynb
  The main file for the Fair Classification experiment for FMNIST data set
3.main_CIFAR10.ipynb
  The main file for the Fair Classification experiment for CIFAR10 data set
4.main plot test.ipynb【Produce Figures】
  The main file for ploting the three figures of Fair Classification experiment 


Data set:
MNIST,FMNIST,CIFAR10 downloaded from pytorch dataset


Required device: GPU
