# piperaceae-features

* Requirements
* How to run `main.py`
* Details of implementation

---

# Requirements

### Python
```
$ conda --version
conda 24.1.0
...
$ python --version
Python 3.11.5
...
$ pip install requirements.txt
...
```

### How to run `surf_lbp.py`
```
$ conda create --name opencv python=3.11
...
$ conda activate opencv
...
$ wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
...
$ wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.x.zip
...
$ unzip opencv.zip
...
$ unzip opencv_contrib.zip
...
$ mkdir -p build && cd build
...
$ cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.x/modules ../opencv-4.x -D PYTHON_DEFAULT_EXECUTABLE=$(/home/xandao/miniconda3/envs/openvcv/python3)
...
$ cmake --build .
...
$ sudo make install
...
$ python3 surf_lbp.py -i ...
```
Replace the `x` for the version chosen.

---
# How to run `main.py`

### 1- Execute `preprocess.py` script to

#### 1.1- Before
```
$ ls input -R
input
labelA       labelC
labelB       labelD
```

#### 1.1- After the ran `preprocess.py` script
```
$ python preprocess.py -i input/ -z 0
$ ...
$ ls output -R
output
info.csv

output
f01       f03
f02       f04
```

In this above example, the parameter `z` indicates the count of zero before each number. The `preprocess.py` generates `info.csv` file that contains two columns:

1. `input`: folder name that contains the images.

2. `output`: folder name with new format name.

Other results after execute the program are:

* images in the folder `labelA` are in `f01` folder
* images in the folder `labelB` are in `f02` folder
* ...

### 2- Execute `main.py` script
```
$ python main.py
$ ...
$ ls output/26-05-2024-18-05-00/features/ -R
features/
dataset.csv
samples.csv
...
features/npy/f1/:
fold-1_patches-3.npy
...
features/npz/f1/:
fold-1_patches-3.npz
```

* The `26-05-2024-18-05-00` is a fictional folder name.
  * Each execution generates a folder name that indicates the date of execution of the program.
* The `npy` or `npz` file is a matrix that contains features and the label.
* `dataset.csv`: a file containing information about the dataset.
  * `fold`: the number of folds present in the dataset.
  * `patches`: the number of splits applied to the images.
  * `features`: number of features.
  * `samples`: number of samples that contain the dataset.
  * `samples+patch`:  number of samples (with patches) that contain the dataset.
* `samples.csv`: a file containing details about the images of the extracted features.
  * `filename`: feature of the filename (image).
  * `fold`: a label that belongs to those images.

---

# Details of implementation

The pipeline folder contains two block diagrams.

1. `preprocess`: is a block diagram of a `preprocess.py`

2. `extract-features`: is a block diagram of the `main.py`