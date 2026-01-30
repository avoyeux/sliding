# 'sliding' package

This python package contains utilities to compute a sliding sigma clipping, where the kernel can
contain weights and the data can contain NaN values. \
This package was created to have a relatively (compared to using scipy.ndimage.generic_filter) fast
way of computing the sliding mean, sliding median, the sliding standard deviation and to apply
a sliding sigma clipping to N-dimensional data (speed up of ~10/20 times depending on the input
choices).


## Install package

As usual, given that the code was set up as a package, to use it, you first import the git
repository and then install it using the 'pyproject.toml' file, i.e.

#### Get the code:
```bash
git clone https://github.com/avoyeux/sigma_clipping_tests.git
cd sigma_clipping_tests
```
#### (**OPTIONAL**) Create and activate a python virtual environnement:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
or on Windows OS:
```bash
python -m venv .venv
source .venv/Scripts/activate
```
#### Install package in virtual environnement (or on bare-metal - wouldn't recommend):
```bash
pip install .
```
You do need to be in the root of the git repository for the last command to work.


## Functions

The *'sliding'* package has 5 different classes:
- **'Convolution'** which lets you perform a convolution (no NaN handling).
- **'SlidingMean'** which performs a sliding mean (NaN handling done).
- **'SlidingMedian'** which performs a sliding median (NaN handling done).
- **'SlidingStandardDeviation'** which performs a sliding standard deviation (NaN handling done).
- **'SlidingSigmaClipping'** which performs a sliding sigma clipping (NaN handling done).

#### Example
```python
# IMPORTs
import numpy as np
from sliding import SlidingMean

# CREATE fake data
fake_data = np.random.rand(36, 1024, 128).astype(np.float64)
fake_data[10:15, 100:200, 50:75] = 1.3
fake_data[7:, 40:60, 70:] = 1.7

# KERNEL
kernel = np.ones((5,) * fake_data.ndim, dtype=fake_data.dtype)
kernel[2, 2, 2] = 0.

# MEAN sliding
mean = SlidingMean(
    data=fake_data,
    kernel=kernel,
    borders='reflect',
    threads=1,
).mean
```

## IMPORTANT

Before using this package some information is needed:

- **float64** values for the data (recommended): While the formula used for the standard
deviation is numerically stable, there is still a need to do one squared operation. Hence, there is
the possibility to get a wrong standard deviation in really rare cases (from personal tests, with
5000 FITs files - usual 2D but sometimes 3D data - had ~1 pixel out of 4 million that was wrong for
~5% of the images when using 'SlidingSigmaClipping' with float32 data).
- **weights**: weighted kernels haven't yet been extensively tested. While I was able to properly
test weights for 'Convolution' and 'SlidingMedian' I still haven't created a sure weighted kernel
test for the other functions.
- **'Borders=None'**: used when adaptative kernels are needed at the borders (i.e. borders are
added as NaN values). While all the other border choices were extensively tested, I still wasn't
able to properly test this option (do not know of any Python function that lets you do so).
- **high memory usage**: while being numerically stable, the 'SlidingStandardDeviation'
implementation does use a lot memory. Did use a buffer to minimize the memory allocations and usage.
That being said, for a numerically stable standard deviation in python, the buffer is of size
kernel.size * data.nbytes. E.g. for a 3x3x3 kernel (so 3D input data), 'SlidingStandardDeviation'
uses at little over 27 times the memory usage than the initial input data.
- **Python3.12+**: because of the type annotations that I have used, the code only runs on Python
3.12 or later. The actual computations need Python3.10+ (because of the := operator).


## Tests
To see what was actually tested, you can check the tests inside the 'tests' folder.

For further information, look at the code. The code is extensively type annotated and all functions should have proper docstrings.
