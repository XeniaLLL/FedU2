# FedU2
This repository is an official PyTorch implementation of paper:
[FedU2: Rethinking the Representation in Federated Unsupervised Learning with Non-IID Data](https://openaccess.thecvf.com/content/CVPR2024/html/Liao_Rethinking_the_Representation_in_Federated_Unsupervised_Learning_with_Non-IID_Data_CVPR_2024_paper.html).
CVPR 2024 (Poster).

Thanks to [@Huabin](https://github.com/zhb2000/fedbox.git) for providing a robust and practical implementation framework.

# Rethinking the Representation in Federated Unsupervised Learning with Non-IID Data

## Abstract
Federated learning achieves effective performance in modeling decentralized data. In practice client data are not well-labeled which makes it potential for federated unsupervised learning (FUSL) with non-IID data. However the performance of existing FUSL methods suffers from insufficient representations i.e. (1) representation collapse entanglement among local and global models and (2) inconsistent representation spaces among local models. The former indicates that representation collapse in local model will subsequently impact the global model and other local models. The latter means that clients model data representation with inconsistent parameters due to the deficiency of supervision signals. In this work we propose FedU2 which enhances generating uniform and unified representation in FUSL with non-IID data. Specifically FedU2 consists of flexible uniform regularizer (FUR) and efficient unified aggregator (EUA). FUR in each client avoids representation collapse via dispersing samples uniformly and EUA in server promotes unified representation by constraining consistent client model updating. To extensively validate the performance of FedU2 we conduct both cross-device and cross-silo evaluation experiments on two benchmark datasets i.e. CIFAR10 and CIFAR100.

## Implementation 
This project backbone is contributively implemented by my co-authors, Huabin, Pengyang, and Fengyuan.


###Step 1: install FL toolbox implemented by my co-author Huabin
```
git clone https://github.com/zhb2000/fedbox.git
cd fedbox
pip install .
```
###Step 2: train model
Assign parameters for `train_xxx.py` that is called in `src/main.py`
```
# execute FedU2
python src/main.py 
```


## Citation
If you find HyperFed useful or inspiring, please consider citing our paper:
```bibtex
@InProceedings{Liao_2024_CVPR,
    author    = {Liao, Xinting and Liu, Weiming and Chen, Chaochao and Zhou, Pengyang and Yu, Fengyuan and Zhu, Huabin and Yao, Binhui and Wang, Tao and Zheng, Xiaolin and Tan, Yanchao},
    title     = {Rethinking the Representation in Federated Unsupervised Learning with Non-IID Data},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {22841-22850}
}
```

