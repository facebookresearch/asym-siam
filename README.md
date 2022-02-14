# Asym: On the Importance of Asymmetry for Siamese Representation Learning

<p align="center">
  <img src="https://github.com/fairinternal/asym-siam/blob/main/figure/teaser.png" width="300">
</p>

This is a PyTorch implementation of the [Asym paper]():
```
@Article{wang2021asym,
  author  = {Xiao Wang and Haoqi Fan and Yuandong Tian and Daisuke Kihara and Xinlei Chen},
  title   = {On the Importance of Asymmetry for Siamese Representation Learning},
  journal = {},
  year    = {2021},
}
```


## Installation

1. [`Install git`](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) 

2. Install PyTorch and ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet).

3. Clone the repository in your computer 
```
git clone https://github.com/facebookresearch/asym-siam & cd asym-siam
```


## 1 Unsupervised Training

This implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported.

### 1.1 MoCo BN Baseline
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

### 1.2 MoCo BN + MultiCrop
```
python main_moco.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 256 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders] --enable-multicrop
```
By simply setting  **--enable-multicrop** to true, we can have asym MultiCrop on source side.

### 1.3 MoCo BN + ScaleMix
```
python main_moco.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 256 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders] --enable-scalemix
```
By simply setting  **--enable-scalemix** to true, we can have asym ScaleMix on source side.

### 1.4 MoCo BN + AsymAug
```
python main_moco.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 256 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders] --enable-asymm-aug
```
By simply setting  **--enable-asymm-aug** to true, we can have Stronger Augmentation on source side and Weaker Augmentation on target side.

### 1.5 MoCo BN + AsymBN
```
python main_moco.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 256 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders] --enable-asym-bn
```
By simply setting  **--enable-asym-bn** to true, we can have asym BN on target side (sync BN for target).

### 1.6 MoCo BN + MeanEnc
```
python main_moco.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 256 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders] --enable-mean-encoding
```
By simply setting  **--enable-mean-encoding** to true, we can have MeanEnc on target side.



### Linear Classification

With a pre-trained model, to train a supervised linear classifier on frozen features/weights in an 8-gpu machine, run:
```
python main_lincls.py \
  -a resnet50 \
  --lr 30.0 \
  --batch-size 256 \
  --pretrained [your checkpoint path] \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
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
<th valign="bottom">top-5</th>
<th valign="bottom">model</th>
<th valign="bottom">md5</th>
<!-- TABLE BODY -->
<tr><td align="left">MoCo BN</td>
<td align="center">100</td>
<td align="center"></td>
<td align="center"></td>
<td align="center"></td>
<td align="center">[checkpoint](https://dl.fbaipublicfiles.com/asym-siam/checkpoint_baseline_100ep.pth)</td>
<td align="center">e82ede</td>
</tr>

<tr><td align="left">MoCo BN<br/> +MultiCrop</td>
<td align="center">100</td>
<td align="center"></td>
<td align="center"></td>
<td align="center"></td>
<td align="center">[checkpoint](https://dl.fbaipublicfiles.com/asym-siam/checkpoint_multicrop_100ep.pth)</td>
<td align="center">892916</td>
</tr>
  
<tr><td align="left">MoCo BN<br/> +ScaleMix</td>
<td align="center">100</td>
<td align="center"></td>
<td align="center"></td>
<td align="center"></td>
<td align="center">[checkpoint](https://dl.fbaipublicfiles.com/asym-siam/checkpoint_scalemix_100ep.pth)</td>
<td align="center">3f5d79</td>
</tr>

<tr><td align="left">MoCo BN<br/> +AsymAug</td>
<td align="center">100</td>
<td align="center"></td>
<td align="center"></td>
<td align="center"></td>
<td align="center">[checkpoint](https://dl.fbaipublicfiles.com/asym-siam/checkpoint_aug_100ep.pth)</td>
<td align="center">d94e24</td>
</tr>
  
<tr><td align="left">MoCo BN<br/> +AsymBN</td>
<td align="center">100</td>
<td align="center"></td>
<td align="center"></td>
<td align="center"></td>
<td align="center">[checkpoint](https://dl.fbaipublicfiles.com/asym-siam/checkpoint_asym_bn_100ep.pth)</td>
<td align="center">2bf912</td>
</tr>
  
<tr><td align="left">MoCo BN<br/> +MeanEnc</td>
<td align="center">100</td>
<td align="center"></td>
<td align="center"></td>
<td align="center"></td>
<td align="center">[checkpoint](https://dl.fbaipublicfiles.com/asym-siam/checkpoint_mean_enc_100ep.pth)</td>
<td align="center">599801</td>
</tr>
  
  
</tbody></table>


### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

