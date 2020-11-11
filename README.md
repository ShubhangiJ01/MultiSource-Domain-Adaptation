# MultiSource-Domain-Domain-Adaptation

![](https://img.shields.io/badge/pytorch-1.6.0-blue.svg?style=flat) ![](https://img.shields.io/badge/python-3.7.1-green.svg?style=flat) 


## Introduction:

Conventional unsupervised domain adaptation (UDA) assumes that training data are sampled from a single domain.
This neglects the more practical scenario where training
data are collected from multiple sources, requiring multisource domain adaptation. To address this problem, a paper was published [Moment Matching for Multi-Source
Domain Adaptation](http://ai.bu.edu/M3SDA/) in which a dataset called Domain
Net, which contain six domains and 0.6 million images distributed among 345 categories, was created. In the same paper they proposed a new deep learning approach, Moment
Matching for Multi-Source Domain Adaptation (M3SDA).
As part of this project, I have extended work of this paper. I have written code for image’s dataset and changed
the distance function used in original paper to calculate loss.
Two different distance functions are used to train our model: 
1. Dynamic Partial Distance Function
1. Mahalanobis distance

## Execution:

To run the code 
```console
$ ./experiment.sh <parameter1> <parameter2> <parameter3> <parameter4>
```

where parameter1 =Target Domain,
parameter2 = max_epoch, 
parameter3 = GPUID,
parameter4 = record_folder



All model weights can be found [here](https://drive.google.com/file/d/1aZtT1rrFXTfDd4XCeauvwgi24axLng9V/view?usp=sharing)
# Report: 

For detail understanding refer [documentation](https://github.com/ShubhangiJ01/MultiSource-Domain-Adaptation/blob/master/Final%20Report.pdf)
