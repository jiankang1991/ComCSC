# Learning Convolutional Sparse Coding on Complex Domain for Interferometric Phase Restoration

[Jian Kang](https://github.com/jiankang1991), [Danfeng Hong](https://sites.google.com/view/danfeng-hong), [Jialin Liu](https://www.math.ucla.edu/~liujl11/), [Gerald Baier](https://www.geoinformatics2018.com/member/geraldbaier/), [Naoto Yokoya](https://naotoyokoya.com/), [Begüm Demir](https://begumdemir.com/)

---

This repo contains the codes for the [TNNLS paper](https://arxiv.org/pdf/2003.03440.pdf). we propose a complex convolutional sparse coding (ComCSC) algorithm and its gradient regularized version (ComCSC-GR) for interferometric phase restoration. Our method outperforms the state-of-the-art methods. The codes are modified from [SPORCO](http://brendt.wohlberg.net/software/SPORCO/) for adapting to complex images.

![alt text](./Selection_001.png)
**10, 000 Monte-Carlo simulations for evaluating the compared methods on the expected values and standard deviations of step function approximation. The amplitude is constant and the coherence value is set as 0.3.**

## Usage

`train_cpx_dic.m` uses ComCSC and ComCSC-GR to train the complex convolutional dictionaries based on the simulated interferograms `train_cpxs.mat`.

`recon_peaks_cpx_comp.mlx` gives a demo using the learned convolutional dictionaries to restore the clean interferogram.


## Citation
```
@article{kang2020comcsc,
  title={{Learning Convolutional Sparse Coding on Complex Domain for Interferometric Phase Restoration}},
  author={Kang, Jian and Hong, Danfeng and Liu, Jialin and Baier, Gerald and Yokoya, Naoto and Demir, Begüm},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2020},
  note={DOI:10.1109/TNNLS.2020.2979546}
  publisher={IEEE}
}

```
