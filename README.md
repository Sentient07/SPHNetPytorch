# SPHNetPytorch
Pytorch implementation of Effective Rotation-invariant Point CNN with Spherical Harmonics kernels


## Usage

1) Follow the notebook example to train the model architecture discussed in the paper for ModelNet40 classification. 

2) In addition, `from layers import SPHConvNet` can be used outside this package as a custom Pytorch layer for plug-and-play into other models.

3) You can download the ModelNet40 dataset for classification at : https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip and change the path in the notebook

4) The code has been tested on `torch==1.9.1` and Cuda 11.1


## Disclaimer 

1) This implementation is an unofficial one. For reproducing the original results, I strongly advise to use the code in Keras : https://github.com/adrienPoulenard/SPHnet.

2) That being said, this code has been tested on the `O/A` setting of Table 1 in the paper and acheives an accuracy of 86% (in comparison to 86.6% with Keras).


## Acknowledgement

Thanks to Adrien Poulenard, the author of the paper, for his help in porting the code.


## Citation
If you use this codebase, please cite paper.
```
@article{poulenard2019effective,
  title={Effective Rotation-invariant Point CNN with Spherical Harmonics kernels},
  author={Poulenard, Adrien and Rakotosaona, Marie-Julie and Ponty, Yann and Ovsjanikov, Maks},
  journal={arXiv preprint arXiv:1906.11555},
  year={2019}
}
```
