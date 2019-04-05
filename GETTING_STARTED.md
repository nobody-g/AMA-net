# Using Detectron

This document provides brief tutorials covering AMA-net for inference and training on the DensePose-COCO dataset.
This document is a modified version of the [`detectron/GETTING_STARTED.md`](https://github.com/facebookresearch/Detectron/blob/master/GETTING_STARTED.md).

- For general information about AMA-net, please see [`README.md`](README.md).
- For installation instructions, please see [`INSTALL.md`](INSTALL.md).


## Testing with Pretrained Models

Make sure that you have downloaded the DensePose evaluation files as instructed in [`INSTALL.md`](INSTALL.md). 
This example shows how to run an end-to-end trained AMA-net model from the model zoo using a single GPU for inference. As configured, this will run inference on all images in `coco_2014_minival` (which must be properly installed).

```
python2 tools/test_net.py \
    --cfg configs/coco_exp_configs/DensePose_ResNet50_FPN_cascade_mask_dp_s1x-e2e_all.yaml \
    TEST.WEIGHTS <The path of Pretrained model> \
    NUM_GPUS 1
```

## Training a Model

This example shows how to train a model using the DensePose-COCO dataset. The model will be an end-to-end trained AMA-net using a ResNet-50-FPN backbone. 

```
python2 tools/train_net.py \
    --cfg configs/coco_exp_configs/DensePose_ResNet50_FPN_cascade_mask_dp_s1x-e2e_all.yaml \
    OUTPUT_DIR /tmp/detectron-output
```


