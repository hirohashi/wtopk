# Learning Weighted Top-k Support Vector Machine

This repository provides the core implementation of our paper entitled 
"Learning Weighted Top-k Support Vector Machine" presented in 
[ACML 2019](http://www.acml-conf.org/2019/).

### Dependencies
This implementation requires the following softwares.
* Python3
* Numpy (version >= 1.15 is required for "take_along_axis" function)
* Scipy

### Usage
Basically, the weighted top-k SVM training for the dummy data (k.demo450_01.mat) with the reguralization parameter "C=10.0', and "exponentially decreased weights of k=3" can be executed by the following command.
```
$ python train_wtopk.py --dataset k.demo450_01.mat --c_svm 10.0 --rho_dist topk_exp --rho_param 3
```
For technical details, please check our paper.
