# WCMC-SAPD
The folder includes the following python file for Distributionally Robust Optimization experiment:

1.algorithms.py
<br />
including functions 
  <br />
  SAPD,SAPD-VR, https://arxiv.org/abs/2205.15084
  <br />
  SMDA,SMDA-VR, https://nips.cc/Conferences/2021/ScheduleMultitrack?event=26760
  <br />
  SREDA, https://proceedings.neurips.cc/paper/2020/hash/ecb47fbb07a752413640f82a945530f8-Abstract.html
  <br />
  PASGDA, https://arxiv.org/abs/2007.13605
  <br />
  
5.model.py

  including model class for DRO experiment
  
3.Opitmization_Method.py

  including a function that is the projection onto the unit simplex
  
6.functions.py

  including shadow error bar plot function
  
7.dataclass.py

  including a data class
<br />
<br />
<br />
<br />
The folder includes the following python jupyter note book file:

1.main_a9a.ipynb
  The main file for the Distributionally Robust Optimization experiment for a9a data set
  
2.main_gisette.ipynb
  The main file for the Distributionally Robust Optimization experiment for gisette data set
  
3.main_sido0.ipynb
  The main file for the Distributionally Robust Optimization experiment for sido0 data set
  
4.main plot test.ipynb
  The main file for ploting the three figures of Distributionally Robust Optimization experiment 
<br />
<br />
<br />
<br />
 Data set:
 
 1.a9a and gisette: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
 
 2.sido0 http://www.causality.inf.ethz.ch/challenge.php?page=datasets
 
 Note that the downloaded data needs to be put in \data\name\data_file
<br />
<br />
<br />
<br />
Required device: GPU
