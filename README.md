# Intersection Over Union (IOU) in NumPy

## Description
This is vectorized implementation of the [IOU (Jaccard index)](<https://en.wikipedia.org/wiki/Jaccard_index>) calculation in NumPy.

![IOU Rectangles](https://upload.wikimedia.org/wikipedia/commons/thumb/c/c7/Intersection_over_Union_-_visual_equation.png/300px-Intersection_over_Union_-_visual_equation.png)

## How to run
Current IOU implementation is based on [numpy](<https://numpy.org/>) module, so make sure that this module is installed.

Project uses virtual environment (**venv**) to run python code. 
You can use helper commands from Makefile to create **venv** and install all necessary dependencies.

```bash
git clone https://github.com/kserhii/iou-numpy.git
cd iou-numpy
make venv
source venv/bin/activate
make update
```    

Function `iou` expects two batches of rectangles as numpy arrays with shape [batch size, 4].
Here "rectangle" is a 4 dimensional vector with positive values [x_min, y_min, x_max, y_max].

```python
import numpy as np

from iou import iou

rect_batch1 = np.array([
    [0, 0, 5, 10],
    [5, 10, 15, 20],
    [0, 5, 5, 10],
    [5, 5, 10, 10]
])

rect_batch2 = np.array([
    [0, 5, 10, 10],
    [0, 0, 5, 5],
    [0, 0, 5, 10],
    [0, 5, 5, 10],
])

rect_iou = iou(rect_batch1, rect_batch2)

# rect_iou: array([0.33333334, 0. , 0.5 , 0. ], dtype=float32)
```

## Run tests

Unit tests are placed int the `tests.py`. Just call `make test` command to run the tests

```bash
make test
```

## Alternative implementations
1. <https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/>
2. <https://medium.com/@venuktan/vectorized-intersection-over-union-iou-in-numpy-and-tensor-flow-4fa16231b63d>
3. <https://codereview.stackexchange.com/questions/204017/intersection-over-union-for-rotated-rectangles>
