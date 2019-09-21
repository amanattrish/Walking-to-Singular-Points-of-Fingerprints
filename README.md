# Walking to Singular Points of Fingerprints

It's the python implementation of paper [Zhu, En & Guo, Xifeng & Yin, Jianping. (2016). Walking to Singular Points of Fingerprints. Pattern Recognition. 56. 10.1016/j.patcog.2016.02.015.](https://www.researchgate.net/publication/297615926_Walking_to_Singular_Points_of_Fingerprints)

## Requirements

Opencv, numpy and scipy

## Usage

Put your images in **test_images** directory and check results in **results** directory.
Make changes in line-24  and line-42 of **example.py** accordingly!

```python
...
#line-24 is
im = cv2.imread('img_path',0)
...

...
#line-42 is
cv2.imwrite('path_to_save_img.ext', stacked_img) # ext = png, jpg, bmp, tif
...
```

Then run the following command in terminal
```bash
python example.py
```

## Result
![](https://github.com/amanattrish/Walking-to-Singular-Points-of-Fingerprints/blob/master/src_img/example.bmp)
![](https://github.com/amanattrish/Walking-to-Singular-Points-of-Fingerprints/blob/master/src_img/example_sp.bmp)

## Resources
[Ridge Orient](https://github.com/Utkarsh-Deshmukh/Fingerprint-Enhancement-Python/blob/master/src/ridge_orient.py) and [Ridge Segment](https://github.com/Utkarsh-Deshmukh/Fingerprint-Enhancement-Python/blob/master/src/ridge_segment.py) functions are taken from github repo [Fingerprint-Enhancement-Python](https://github.com/Utkarsh-Deshmukh/Fingerprint-Enhancement-Python) written by [Utkarsh Deshmukh](https://github.com/Utkarsh-Deshmukh).
The other functions are python implementation of [Peter Kovesi](https://www.peterkovesi.com/)'s matlab functions.



## License
### For ridge_orient.py and ridge_segment
###### BSD 2-Clause License
###### Copyright (c) 2017, Utkarsh-Deshmukh
###### All rights reserved.

## Acknowledgements
The author would like to thank Dr. [Peter Kovesi](https://www.peterkovesi.com/) (This code is a python implementation of his work). The author would also like to thank Mr. [Nagasai Bharat](https://github.com/NagasaiBharat) for his contribution on code structuring.
