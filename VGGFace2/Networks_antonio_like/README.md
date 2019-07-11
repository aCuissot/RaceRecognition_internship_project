Here we are using MobileNet, VGG16, Resnet, VGGFace(with VGG16 model) and NASNet networks.

For each networks, initial learning rate is set to 0.005 and is divided by 2 each 6 epochs

You can tune the batch composition in `dataset_tool.py`, here we have batches representative of the used dataset

The trains are done on 2.000.000 images from VGGFace2 dataset and stoped when accuracy seem stable

Weights gave by training are available in the folder `trained_networks`

The matrix structure:

|  | African predicted    | Asian predicted      | Latin/Caucasian predicted       |    Indian predicted  |
|:----:|:----:|:-----:|:------:|:----:|
|Real Africans|   |     |     |     |
|Real Asian|    |   |     |    |
|Real Latin/Caucasian|    |    |   |   |
|Real Indians|     |      |     |   |

#### 1. Batches respecting proportions

Until we have big steps in ethnicities proportions in the dataset, we try to keep the same proportions in each batch, so here we got around 6% Africans, 8% Asians, 80% Latin/Caucasian and 6% Indians in each batch

##### 1.1. Results for ResNet

After 16 epochs (around 36 hours on the workstation), we got `loss: 0.1684 - acc: 0.9406 - val_loss: 0.2173 - val_acc: 0.9319`.

This is the confusion matrix computer on testSet:

|      |       |        |      |
|:----:|:-----:|:------:|:----:|
| 7774 |  112  |  1296  |  82  |
|  290 | 16214 |  2950  |  171 |
|  619 |  1053 | 129441 | 2395 |
|  79  |   77  |  2016  | 4455 |

And this is a normalized version of this matrix:

|      |       |        |      |
|:----:|:-----:|:------:|:----:|
| 0.839|   0.012  |   0.140  |  0.009  |
|  0.015 |  0.826 |  0.150  |  0.009 |
|  0.005 |   0.008 | 0.969 | 0.018 |
|  0.012  |    0.012  |  0.304  | 0.672 |

![alt text](data/resnet_train.png "evolution of loss and accuracy during training")


##### 1.2. Results for MobileNet

After 16 epochs.

This is the confusion matrix computer on testSet:

|      |       |        |      |
|:----:|:-----:|:------:|:----:|
| 7925 |  60  |  1168  |  111  |
|  251 | 16517 |  2650  |  270 |
|  832 |  1096 | 129019 | 2561 |
|  41  |   61  |  1763  | 4762 |

And this is a normalized version of this matrix:

|      |       |        |      |
|:----:|:-----:|:------:|:----:|
| 0.855|   0.006  |   0.126  |  0.013  |
|  0.013 |  0.842 |  0.135  |  0.010 |
|  0.006 |   0.008 | 0.966 | 0.019 |
|  0.006  |    0.009  |  0.266  | 0.719 |

##### 1.3. Results for VGG16

After 17 epochs, we got `loss: 0.1570 - acc: 0.9457 - val_loss: 0.2124 - val_acc: 0.9381`.

This is the confusion matrix computer on testSet:

|      |       |        |      |
|:----:|:-----:|:------:|:----:|
| 7852 |  88  |  1252  |  72  |
|  198 | 16917 |  2396  |  114 |
|  614 |  867 | 129628 | 2399 |
|  48  |   43  |  1831  | 4705 |

And this is a normalized version of this matrix:

|      |       |        |      |
|:----:|:-----:|:------:|:----:|
| 0.848|   0.009  |   0.135  |  0.008  |
|  0.010 |  0.862 |  0.122  |  0.006 |
|  0.005 |   0.006 | 0.971 | 0.018 |
|  0.007  |    0.007  |  0.276  | 0.710 |

![alt text](data/vgg_train.png "evolution of loss and accuracy during training")

##### 1.4. Results for VGGFace

After 9 epochs

This is the confusion matrix computer on testSet:

|      |       |        |      |
|:----:|:-----:|:------:|:----:|
| 7927 |  61  |  1146  |  130  |
|  185 | 16914 |  2327  |  199 |
|  767 |  930 | 128992 | 2819 |
|  33  |   39  |  1664  | 4891 |

And this is a normalized version of this matrix:

|      |       |        |      |
|:----:|:-----:|:------:|:----:|
| 0.856|   0.006  |   0.124  |  0.014  |
|  0.009 |  0.862 |  0.119  |  0.010 |
|  0.006 |   0.007 | 0.966 | 0.021 |
|  0.005  |    0.006  |  0.251  | 0.738 |

![alt text](data/vggface_train_bal.png "evolution of loss and accuracy during training")

#### 2. Batches homogeneous

Here we tried to use homogeneous ethnicity repartition in each batch, so each batch should contain 25% of each ethnicity.

##### 2.4. Results for VGGFace

After 9 epochs

This is the confusion matrix computer on testSet:

|      |       |        |      |
|:----:|:-----:|:------:|:----:|
| 8369 |  101  |  657  |  137  |
|  301 | 18087 |  964  |  273 |
|  1976 |  2329 | 124545 | 4658 |
|  59  |   106  |  768  | 5694 |

And this is a normalized version of this matrix:

|      |       |        |      |
|:----:|:-----:|:------:|:----:|
| 0.903|   0.011  |   0.071  |  0.015  |
|  0.015 |  0.922 |  0.049  |  0.014 |
|  0.015 |   0.017 | 0.933 | 0.035 |
|  0.009  |    0.016  |  0.116  | 0.859 |

![alt text](data/vggface_train_bal.png "evolution of loss and accuracy during training")
