## Image Classification Model & API

Image classification (CNN) with Keras + TensorFlow on Tiny ImageNet dataset

- Custom developed model
- Trained model available at [here](https://storage.cloud.google.com/bp-ml-models/180301_tiny_imagenet_weights.h5) (might require a blupig.net account)
- Data from https://tiny-imagenet.herokuapp.com/

### Program files

| File          | Description                                             |
|---------------|---------------------------------------------------------|
| model.py      | CNN model with multi GPU support                        |
| train.py      | Training script for model development                   |
| predit.py     | Prediction script for model development                 |
| data_utils.py | Data IO, image pre-processing, class name mapping, etc. |
| api_server.py | Serves trained model as HTTP API                        |

### Environment Setup

Assuming `python3` and `virtualenv` are installed.

```bash
virtualenv venv
source venv/bin/activate

# To train on GPU, use `tensorflow-gpu` in requirements.txt (CUDA required).
pip3 install -r requirements.txt
```

## Run API Server

A trained model is required, update `trained_model_path` in `api_server.py`.

```
python3 api_server.py
```

The API accepts `POST` requests:

```
curl -F 'image=@cat.jpg' http://IP:PORT/images/annotate
```

## Development

### Training

Training takes around 3 hours on 2 [`NVIDIA Tesla P100`](http://www.nvidia.com/object/tesla-p100.html) GPUs.

```
python3 train.py
```

It saves the model as `weights.h5` in current directory.

### Prediction

A trained model is required, update `trained_model_path` in `predit.py`.

Test data will be read from **subdirectories** in `test_images/`.

```
python3 predit.py
```
