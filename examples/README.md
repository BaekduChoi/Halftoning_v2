# Digital halftoning using deep AR models

This is an implementation of digital halftoning using deep AR models in [PyTorch](https://pytorch.org/). 

Digital halftoning is a image translation problem where the image is represented using only 0 or 1 as pixel values instead of continuous (ignoring quantization) pixel values.
The aim is to preserve texture in the original image as much as possible while simultaneously minimizing noise inherent due to the binarization.

For selected results, please refer to the 'examples' directory. The output images from the cGAN is thresholded wrt 0.5.

Please note that halftone images cannot be viewed or saved with lossy compression and needs to be viewed only in magnifications 100%, 200%, 300%, etc.



This project aims to mimic the results of digital halftoning from [DBS](https://ieeexplore.ieee.org/document/877215) algorithm; it is known to generate high quality halftone images.
The images generated with DBS algorithm are marked as 'GT' in the example folder.

One known method of mimicking DBS algorithm is to use a DBS-generated screen, which is generated following description in [this paper](https://ieeexplore.ieee.org/document/559555). However, it results in rather noisy halftone image.

Our deep AR model generates halftone images comparable to those generated via DBS.


The training images are generated using [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) as the original images. The images are first divided into 256x256 patches and then DBS is performed on them.


