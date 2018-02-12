## MNIST

MNIST classification (CNN) using TensorFlow

Code slightly modified from [this tutorial](https://www.tensorflow.org/tutorials/layers).

Added ability for predicting from arbitrary number (sort of) image files from the command line.

### Environment Setup

Assuming `python3` and `virtualenv` are installed.

```bash
virtualenv venv
source venv/bin/activate

# To train on GPU, use `tensorflow-gpu` in requirements.txt (additional setup required).
pip3 install -r requirements.txt
```

### Training

Training takes less than 5 minutes on a single GCP [`n1-standard-2`](https://cloud.google.com/compute/docs/machine-types#standard_machine_types) instance with 1 [`NVIDIA Tesla K80`](http://www.nvidia.com/object/tesla-k80.html) GPU attached.

Or around 1 hour on an Intel [i7-7567U](https://ark.intel.com/products/97541/Intel-Core-i7-7567U-Processor-4M-Cache-up-to-4_00-GHz) CPU.

```
python3 train.py
```

It saves the model as `mnist_model` in current directory.

### Classification

A trained model `mnist_model` is required in working directory.

```
python3 classify.py test/*.png
```
