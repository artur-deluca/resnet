# ResNet
Study notes and implementation in tensorflow from [He et al.'s Residual Networks (2015)](https://arxiv.org/abs/1512.03385) of a ResNet for the CIFAR-10 dataset.

# How to use
To setup project:
 ```
make create environment
#activate virtual environment (either by conda or virtualenv)
make requirements
```

To start to train a ResNet (default ResNet 20)
 ```
make train
```

To check on the result on Tensorboard:
 ```
make tensorboard
```

Project Organization
------------

    ├── Makefile                 <- Use `make help` for more info
    ├── data_loader.py           <- Dataset download and manipulation
    ├── resnet.py                <- ResNet implementation
    ├── main.py                  <- intializing, training and checkpoint saving
    ├── test_environment.py      <- Combined with Makefile, checks if python used is 3
    ├── requirements.txt         <- requirements file
    


