## CIFAR-10

CIFAR-10 classification (CNN) using TFLearn

- Modified from https://github.com/tflearn/tflearn/blob/master/examples/images/convnet_cifar10.py
- Trained model available at [here](https://storage.cloud.google.com/bp-ml-models/180212_cifar_10_model.tflearn.tar.gz) (requires a `blupig.net` account to access).

### Environment Setup

Assuming `python3` and `virtualenv` are installed.

```bash
virtualenv venv
source venv/bin/activate

# To train on GPU, use `tensorflow-gpu` in requirements.txt (additional setup required).
pip3 install -r requirements.txt
```

### Training

Training takes less than 20 minutes on a single [`NVIDIA Tesla K80`](http://www.nvidia.com/object/tesla-k80.html) GPU (AWS [`p2.xlarge`](https://aws.amazon.com/ec2/instance-types/p2/) instance).

Or roughly 2-3 hours on an Intel [i7-7567U](https://ark.intel.com/products/97541/Intel-Core-i7-7567U-Processor-4M-Cache-up-to-4_00-GHz) CPU.

```
python3 train.py
```

It saves the model as `model.tflearn` in current directory.

### Classification

A trained model `model.tflearn` is required in `models` directory.

```
python3 classify.py test/1.jpg
```
