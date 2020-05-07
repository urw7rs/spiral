# SPIRAL

## Overview

This repository contains environments described in the ICML'18
paper ["Synthesizing Programs for Images using Reinforced Adversarial Learning"](http://proceedings.mlr.press/v80/ganin18a.html).
For the time being, the authors of the paper are providing two simulators:
one based on [`libmypaint`](https://github.com/mypaint/libmypaint) and one
based on [`Fluid Paint`](https://github.com/dli/paint) (**NOTE:** the authors' 
implementation is written in `C++` whereas the original is in `javascript`).

If you feel an immediate urge to dive into the code the most relevant files are:

| Path | Description |
| :--- | :--- |
| [`spiral/envs/libmypaint.py`](spiral/environments/libmypaint.py) | The `libmypaint`-based environment |
| [`spiral/envs/fluid.py`](spiral/environments/fluid.py) | The `Fluid Paint`-based environment |

## Installation

This section describes how to build and install the package on Ubuntu (16.04 or
newer). The following instructions (with slight modifications) might also work
for other Linux distributions.

Clone this repository and fetch the external submodules:

```shell
git clone https://github.com/urw7rs/spiral.git
cd spiral
git submodule update --init --recursive
```

Install required packages:

```shell
apt-get install cmake pkg-config protobuf-compiler libjson-c-dev intltool libpython3-dev python3-pip
pip3 install six setuptools numpy scipy
```

**WARNING:** Make sure that you have `cmake` **3.14** or later since we rely
on its capability to find `numpy` libraries. If your package manager doesn't
provide it follow the installation instructions from
[here](https://cmake.org/install/). You can check the version by
running `cmake --version `.

Finally, run the following command to install the SPIRAL package itself:

```shell
python3 setup.py develop --user
```

You will also need to obtain the brush files for the `libmypaint` environment
to work properly. These can be found
[here](https://github.com/mypaint/mypaint-brushes). For example, you can
place them in `third_party` folder like this:

```shell
wget -c https://github.com/mypaint/mypaint-brushes/archive/v1.3.0.tar.gz -O - | tar -xz -C third_party
```

Finally, the `Fluid Paint` environment depends on the shaders from the original
`javascript` [implementation](https://github.com/dli/paint). You can obtain
them by running the following commands:

```shell
git clone https://github.com/dli/paint third_party/paint
patch third_party/paint/shaders/setbristles.frag third_party/paint-setbristles.patch
```

Optionally, in order to be able to try out the package in the provided
`jupyter` [notebook](notebooks/spiral-demo.ipynb), you’ll need to install
the following packages:

```shell
pip3 install matplotlib jupyter
```

## Usage

For a basic example of how to use the package see [original repo](https://github.com/deepmind/spiral)
