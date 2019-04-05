# Installing AMA-net

The AMA-net is implemented within the [`detectron`](https://github.com/facebookresearch/Detectron) framework and [`DensePose`](https://github.com/facebookresearch/Densepose) project. This document is based on the Detectron installation instructions, for troubleshooting please refer to the [`detectron installation document`](https://github.com/facebookresearch/Detectron/blob/master/INSTALL.md).

**Requirements:**

- NVIDIA GPU, Linux, Python2
- Caffe2, various standard Python packages, and the COCO API; Instructions for installing these dependencies are found below

**Notes:**

- Detectron operators currently do not have CPU implementation; a GPU system is required.
- Detectron has been tested extensively with CUDA 8.0 and cuDNN 6.0.21.

## Caffe2

To install Caffe2 with CUDA support, follow the [installation instructions](https://caffe2.ai/docs/getting-started.html) from the [Caffe2 website](https://caffe2.ai/). **If you already have Caffe2 installed, make sure to update your Caffe2 to a version that includes the [Detectron module](https://github.com/caffe2/caffe2/tree/master/modules/detectron).**

Please ensure that your Caffe2 installation was successful before proceeding by running the following commands and checking their output as directed in the comments.

```
# To check if Caffe2 build was successful
python2 -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"

# To check if Caffe2 GPU build was successful
# This must print a number > 0 in order to use Detectron
python2 -c 'from caffe2.python import workspace; print(workspace.NumCudaDevices())'
```

If the `caffe2` Python package is not found, you likely need to adjust your `PYTHONPATH` environment variable to include its location (`/path/to/caffe2/build`, where `build` is the Caffe2 CMake build directory).

## Other Dependencies

Install the [COCO API](https://github.com/cocodataset/cocoapi):

```
# COCOAPI=/path/to/clone/cocoapi
git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
cd $COCOAPI/PythonAPI
# Install into global site-packages
make install
# Alternatively, if you do not have permissions or prefer
# not to install the COCO API into global site-packages
python2 setup.py install --user
```

Note that instructions like `# COCOAPI=/path/to/install/cocoapi` indicate that you should pick a path where you'd like to have the software cloned and then set an environment variable (`COCOAPI` in this case) accordingly.

## AMA-net

Clone the AMA-net repository:

```
# AMANET=/path/to/clone/AMA-net
git clone https://github.com/nobody-g/AMA-net $AMANET
```

Install Python dependencies:

```
pip install -r $AMANET/requirements.txt
```

Set up Python modules:

```
cd $AMANET && make
```

Check that Detectron tests pass (e.g. for [`SpatialNarrowAsOp test`](tests/test_spatial_narrow_as_op.py)):

```
python2 $AMANET/detectron/tests/test_spatial_narrow_as_op.py
```

Build the custom operators library:

```
cd $AMANET && make ops
```

Check that the custom operator tests pass:

```
python2 $AMANET/detectron/tests/test_zero_even_op.py
```
### Fetch DensePose data.
Get necessary files to run, train and evaluate AMA-net.
```
cd $AMANET/DensePoseData
bash get_densepose_uv.sh
```
For training, download the DensePose-COCO dataset:
```
bash get_DensePose_COCO.sh
```
For evaluation, get the necessary files:
```
bash get_eval_data.sh
```
## Setting-up the COCO dataset.

Create a symlink for the COCO dataset in your `datasets/data` folder.
```
ln -s /path/to/coco $DENSEPOSE/detectron/datasets/data/coco
```

Create symlinks for the DensePose-COCO annotations

```
ln -s $DENSEPOSE/DensePoseData/DensePose_COCO/densepose_coco_2014_minival.json $DENSEPOSE/detectron/datasets/data/coco/annotations/
ln -s $DENSEPOSE/DensePoseData/DensePose_COCO/densepose_coco_2014_train.json $DENSEPOSE/detectron/datasets/data/coco/annotations/
ln -s $DENSEPOSE/DensePoseData/DensePose_COCO/densepose_coco_2014_valminusminival.json $DENSEPOSE/detectron/datasets/data/coco/annotations/
```

Your local COCO dataset copy at `/path/to/coco` should have the following directory structure:

```
coco
|_ coco_train2014
|  |_ <im-1-name>.jpg
|  |_ ...
|  |_ <im-N-name>.jpg
|_ coco_val2014
|_ ...
|_ annotations
   |_ instances_train2014.json
   |_ ...
```

