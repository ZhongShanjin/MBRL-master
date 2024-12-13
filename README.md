# Mixup and Balanced Relation Learning (MBRL)

## Install
Please refer to [INSTALL](INSTALL.md).

## Dataset
Please refer to [DATASET](DATASET.md).


## Object Detector

### Download Pre-trained Detector

In generally SGG tasks, the detector is pre-trained on the object bounding box annotations on training set. We directly use the [pre-trained Faster R-CNN](https://onedrive.live.com/embed?cid=22376FFAD72C4B64&resid=22376FFAD72C4B64%21779870&authkey=AH5CPVb9g5E67iQ) provided by [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch), because our 20 category setting and their 50 category setting have the same training set.

After you download the [Faster R-CNN model](https://onedrive.live.com/embed?cid=22376FFAD72C4B64&resid=22376FFAD72C4B64%21779870&authkey=AH5CPVb9g5E67iQ), please extract all the files to the directory `$INSTALL_DIR/checkpoints/pretrained_faster_rcnn`. To train your own Faster R-CNN model, please follow the next section.

The above pre-trained Faster R-CNN model achives 38.52/26.35/28.14 mAp on VG train/val/test set respectively.

### Pre-train Your Own Detector

In this work, we do not modify the Faster R-CNN part. The training process can be referred to the [origin code](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/README.md).


### VG150-Training
For our pre-trained models and enhanced datasets, please refer to [MODEL_ZOO.md](MODEL_ZOO.md)

1. PREDCLS

```sh
# train a supervised model
bash cmds/50/motif/predcls/sup/train.sh
# conduct identifies triplets with semantic similarities
bash cmds/50/motif/predcls/lt/internal/relabel.sh
bash cmds/50/motif/predcls/lt/internal/relabel_cut.sh
# train a new model
bash cmds/50/motif/predcls/lt/combine/train.sh
# evaluate a model
bash cmds/50/motif/predcls/lt/combine/val.sh
```

Do not forget to specify the **OUTPUT_PATH** in **val.sh** as the path to the model directory you want to evaluate.


Note that the `motif` can replaced with other models we provide (e.g.  `vctree`, and `transformer`).


2. SGCLS

```sh
# train a new model
bash cmds/50/motif/sgcls/lt/combine/train.sh
# evaluate a model
bash cmds/50/motif/sgcls/lt/combine/val.sh
```

3. SGDET

```sh
# train a new model
bash cmds/50/motif/sgdet/lt/combine/train.sh
# evaluate a model
bash cmds/50/motif/sgdet/lt/combine/val.sh
```

### GQA200-Training
The overall training is similar with VG150.

1. PREDCLS

```sh
# train a supervised model
bash cmds/gqa/motif/predcls/sup/train.sh
# conduct identifies triplets with semantic similarities
bash cmds/gqa/motif/predcls/lt/internal/relabel.sh
bash cmds/gqa/motif/predcls/lt/internal/relabel_cut.sh
# train a new model
bash cmds/gqa/motif/predcls/lt/combine/train.sh
# evaluate a model
bash cmds/gqa/motif/predcls/lt/combine/val.sh
```

2. SGCLS

```sh
# train a new model
bash cmds/gqa/motif/sgcls/lt/combine/train.sh
# evaluate a model
bash cmds/gqa/motif/sgcls/lt/combine/val.sh
```

3. SGDET

First go to the **OUTPUT_PATH** of your model.
```sh
# train a new model
bash cmds/gqa/motif/sgdet/lt/combine/train.sh
# evaluate a model
bash cmds/gqa/motif/sgdet/lt/combine/val.sh
```


## Bugs or questions?
If you have any questions related to the code or the paper, feel free to email Shanjin Zhong (`2022024943@m.scnu.edu.cn`). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!

## Acknowledgement
The code is built on [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch), [IETrans-SGG.pytorch](https://github.com/waxnkw/IETrans-SGG.pytorch), which are also licensed under the MIT License.

Thanks for their excellent codes.

## References

Tang, K. (2020). A Scene Graph Generation Codebase in PyTorch [Computer software]. GitHub. Retrieved July 15, 2024, from https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch

Zhang, A. (2022). IETrans-SGG.pytorch [Computer software]. GitHub. Retrieved July 15, 2024, from https://github.com/waxnkw/IETrans-SGG.pytorch

