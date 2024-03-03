# Classiferris

An MNIST classifier using a simple neural network written in Rust.

## Pre-requisites

- Rust 1.31.0 or later
- Cargo
- Python 3.6 or later (for downloading the MNIST dataset)

### Downloading the MNIST dataset

You may use the `data/download.py` script to download the MNIST dataset. This script requires Python 3.6 or later.

```sh
cd data
# Create a new virtual environment and activate it
  # GNU/Linux and macOS
  python3 -m venv .venv
  source .venv/bin/activate
  # Windows
  python -m venv .venv
  .venv\Scripts\activate
# Install the required packages
pip install -r requirements.txt
# Download the dataset
python download.py
# Deactivate the virtual environment
deactivate
cd .. # Return to the project root
```

The script will download the following files to the `data` directory:

- [train-images-idx3-ubyte.gz](http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz)
- [train-labels-idx1-ubyte.gz](http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz)
- [t10k-images-idx3-ubyte.gz](http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz)
- [t10k-labels-idx1-ubyte.gz](http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz)

These files are then unpacked and stored as a CSV with the labels prepended to the image data.
All values are `u8` and the image data is flattened to a single row of 784 values.
