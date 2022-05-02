# Asym-Siam: On the Importance of Asymmetry for Siamese Representation Learning

<p align="center">
  <img src="https://user-images.githubusercontent.com/2420753/161443048-ed1751ed-8a32-4d7d-85b7-024a6dc09067.png" width="300">
</p>

This is a PyTorch implementation of the [Asym-Siam paper](https://arxiv.org/abs/2204.00613), CVPR 2022:
```
@inproceedings{wang2022asym,
  title     = {On the Importance of Asymmetry for Siamese Representation Learning},
  author    = {Xiao Wang and Haoqi Fan and Yuandong Tian and Daisuke Kihara and Xinlei Chen},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022}
}
```
The pre-training code is built on [MoCo](https://github.com/facebookresearch/moco), with additional designs described and analyzed in the paper.

The linear classification code is from [SimSiam](https://github.com/facebookresearch/simsiam), which uses LARS optimizer.


## Installation

1. [`Install git`](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) 

2. Install PyTorch and ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet).

3. Install [apex](https://github.com/NVIDIA/apex) for the LARS optimizer used in linear classification. If you find it hard to install apex, it suffices to just copy the [code](https://github.com/NVIDIA/apex/blob/master/apex/parallel/LARC.py) directly for use.

4. Clone the repository: 
```
git clone https://github.com/facebookresearch/asym-siam & cd asym-siam
```


## 1 Unsupervised Training

This implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported.

### 1.1 Our MoCo Baseline (BN in projector MLP)
To do unsupervised pre-training of a ResNet-50 model on ImageNet in an 8-gpu machine, run:
```
python main_moco.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 256 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders]
```
This script uses all the default hyper-parameters as described in the MoCo v2 paper. We only upgrade the projector to a MLP with BN layer.

### 1.2 MoCo + MultiCrop
```
python main_moco.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 256 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders] --enable-multicrop
```
By simply setting  **--enable-multicrop** to true, we can have asym MultiCrop on source side.

### 1.3 MoCo + ScaleMix
```
python main_moco.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 256 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders] --enable-scalemix
```
By simply setting  **--enable-scalemix** to true, we can have asym ScaleMix on source side.

### 1.4 MoCo + AsymAug
```
python main_moco.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 256 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders] --enable-asymm-aug
```
By simply setting  **--enable-asymm-aug** to true, we can have Stronger Augmentation on source side and Weaker Augmentation on target side.

### 1.5 MoCo + AsymBN
```
python main_moco.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 256 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders] --enable-asym-bn
```
By simply setting  **--enable-asym-bn** to true, we can have asym BN on target side (sync BN for target).

### 1.6 MoCo + MeanEnc
```
python main_moco.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 256 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders] --enable-mean-encoding
```
By simply setting  **--enable-mean-encoding** to true, we can have MeanEnc on target side.


## 2 Linear Classification

With a pre-trained model, to train a supervised linear classifier on frozen features/weights, run:
```
python main_lincls.py \
  -a resnet50 \
  --lars \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --pretrained [your checkpoint path] \
  [your imagenet-folder with train and val folders]
```

Linear classification results on ImageNet using this repo with 8 NVIDIA V100 GPUs :
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Method</th>
<th valign="bottom">pre-train<br/>epochs</th>
<th valign="bottom">pre-train<br/>time</th>
<th valign="bottom">top-1</th>
<th valign="bottom">model</th>
<th valign="bottom">md5</th>
<!-- TABLE BODY -->
<tr><td align="left">Our MoCo</td>
<td align="center">100</td>
<td align="center">23.6h</td>
<td align="center">65.8</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/asym-siam/checkpoint_baseline_100ep.pth">download</a></td>
<td align="center"><tt>e82ede</tt></td>
</tr>

<tr><td align="left">MoCo<br/> +MultiCrop</td>
<td align="center">100</td>
<td align="center">50.8h</td>
<td align="center">69.9</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/asym-siam/checkpoint_multicrop_100ep.pth">download</a></td>
<td align="center"><tt>892916</tt></td>
</tr>
  
<tr><td align="left">MoCo<br/> +ScaleMix</td>
<td align="center">100</td>
<td align="center">30.7h</td>
<td align="center">67.6</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/asym-siam/checkpoint_scalemix_100ep.pth">download</a></td>
<td align="center"><tt>3f5d79</tt></td>
</tr>

<tr><td align="left">MoCo<br/> +AsymAug</td>
<td align="center">100</td>
<td align="center">24.0h</td>
<td align="center">67.2</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/asym-siam/checkpoint_aug_100ep.pth">download</a></td>
<td align="center"><tt>d94e24</tt></td>
</tr>
  
<tr><td align="left">MoCo<br/> +AsymBN</td>
<td align="center">100</td>
<td align="center">23.8h</td>
<td align="center">66.3</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/asym-siam/checkpoint_asym_bn_100ep.pth">download</a></td>
<td align="center"><tt>2bf912</tt></td>
</tr>
  
<tr><td align="left">MoCo<br/> +MeanEnc</td>
<td align="center">100</td>
<td align="center">32.2h</td>
<td align="center">67.7</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/asym-siam/checkpoint_mean_enc_100ep.pth">download</a></td>
<td align="center"><tt>599801</tt></td>
</tr>
</tbody></table>


### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

