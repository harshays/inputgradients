# Do input gradients highlight discriminative features? 

### Summary

This repository consists of code primitives and Jupyter notebooks that can be used to replicate and extend the findings presented in the paper "Do input gradients highlight discriminative features? " ([link](https://arxiv.org/abs/2102.12781)). In addition to the modules in ```scripts/```, we provide two Jupyter notebooks to reproduce the findings presented in our paper: 


1. ```01_fig_in_real_data.ipynb``` uses our attribution quality evaluation framework (described in section 3 of the [paper](https://arxiv.org/abs/2102.12781)) to study input gradient attributions of standard and robust ResNets trained on benchmark image classification datasets. We show that standard models exhibit Feature Inversion in Gradients (FIG), whereas adversarially robust models fix feature inversion (RF-FIG). 
2. ```02_fig_in_synthetic_data.ipynb``` shows that standard and robust fully-connected networks (FCNs) trained on a simple synthetic data distribution exhibit FIG (feature inversion in gradients) and RF-FIG (robustness fixes FIG) as well. In addition to substantiating our counter-intuitive empirical findings on real data, the synthetic dataset enables us to better understand FIG and RF-FIG empirically as well as theoretically; check out section 5 of the [paper](https://arxiv.org/abs/2102.12781) for more information.

### Setup
1. Our code and Jupyter notebooks require Python 3.7.3, Torch 1.1.0, Torchvision 0.3.0, Ubuntu 18.04.2 LTS and additional packages listed in `requirements.txt`.
2. The first notebook ```01_fig_in_real_data.ipynb``` requires standard and robust models that are trained on original and attribution-masked CIFAR-10 and ImageNet-10 datasets. Our models can be downloaded using this [link](https://www.dropbox.com/s/xn3xyyxug3w1jzb/inputgradients.tar.gz?dl=0), which includes ~20 models trained on original datasets and ~400 models trained on masked datasets. Check out ```01_fig_in_real_data.ipynb``` to know more about training standard and robust models on original and attribution-masked datasets from scratch. 

--- 
If you find this project useful in your research, please consider citing the following paper:
```
@article{shah2021input,
      title={Do Input Gradients Highlight Discriminative Features?}, 
      author={Harshay Shah and Prateek Jain and Praneeth Netrapalli},
      year={2021},
      eprint={2102.12781},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
