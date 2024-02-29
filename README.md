# Image Classification Using Swin Transformer With RandAugment, CutMix, and MixUp


 <div align="center">
    <a href="https://colab.research.google.com/github/reshalfahsi/image-classification-augmentation/blob/master/Image_Classification_Using_Swin_Transformer_With_RandAugment_CutMix_and_MixUp.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="colab"></a>
    <br />
 </div>


In this project, we will explore three distinct Swin Transformers, i.e., without augmentation, with augmentation, and without using the pre-trained weight (or from scratch). Here, the augmentation is undertaken with RandAugment, CutMix, and MixUp. We are about to witness the consequences of utilizing augmentation and pre-trained weight (transfer learning) on the models on the imbalanced dataset, i.e., Caltech-256. The dataset is split per category with a ratio of ``81``:``9``:``10`` for the training, validation, and testing sets. For the from scratch model, each category is truncated to ``100`` instances. Applying the augmentation and pre-trained weight clearly boosts the performance of the model. Not to mention the pre-trained weight insanely pushes the model to effectively predict the right label in the top-1 and top-5.


## Experiment

Check out this [notebook](https://github.com/reshalfahsi/image-classification-augmentation/blob/master/Image_Classification_Using_Swin_Transformer_With_RandAugment_CutMix_and_MixUp.ipynb) to see and ponder the full implementation.


## Result

## Quantitative Result

The result below shows the performance of three different Swin Transformer models: without augmentation, with augmentation, and from scratch, quantitatively.

Model | Loss | Top-1 Acc. | Top-5 Acc. |
------------ | ------------- | ------------- | ------------- |
No Augmentation |  0.369 | 90.17% | 97.68% |
Augmentation | 0.347 | 91.57% | 98.75% |
From Scratch | 4.544 | 11.58% | 27.09% |


## Validation Accuracy and Loss Curve

<p align="center"> <img src="https://github.com/reshalfahsi/image-classification-augmentation/blob/master/assets/val_acc_curve.png" alt="acc_curve" > <br /> Accuracy curves of the models on the validation set. </p>

<p align="center"> <img src="https://github.com/reshalfahsi/image-classification-augmentation/blob/master/assets/val_loss_curve.png" alt="loss_curve" > <br /> Loss curves of the models on the validation set. </p>


## Qualitative Result

The following collated pictures visually delineate the quality of the prediction of the three models.

<p align="center"> <img src="https://github.com/reshalfahsi/image-classification-augmentation/blob/master/assets/no_aug_qualitative.png" alt="no_aug_qualitative" > <br /> The prediction result of Swin Transformer without augmentation. </p>

<p align="center"> <img src="https://github.com/reshalfahsi/image-classification-augmentation/blob/master/assets/aug_qualitative.png" alt="aug_qualitative" > <br /> The prediction result of Swin Transformer with augmentation. </p>

<p align="center"> <img src="https://github.com/reshalfahsi/image-classification-augmentation/blob/master/assets/scratch_qualitative.png" alt="scratch_qualitative" > <br /> The prediction result of Swin Transformer from scratch (no pre-trained). </p>


## Credit

- [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/pdf/2103.14030.pdf)
- [TorchVision's Swin Transformer](https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py)
- [Image classification with Swin Transformers](https://keras.io/examples/vision/swin_transformers/)
- [Caltech-256 Object Category Dataset](https://authors.library.caltech.edu/records/5sv1j-ytw97)
- [TorchVision's Caltech256 Dataset](https://github.com/pytorch/vision/blob/main/torchvision/datasets/caltech.py)
- [RandAugment: Practical automated data augmentation with a reduced search space](https://arxiv.org/pdf/1909.13719.pdf)
- [RandAugment for Image Classification for Improved Robustness](https://keras.io/examples/vision/randaugment/)
- [CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features](https://arxiv.org/pdf/1905.04899.pdf)
- [CutMix data augmentation for image classification](https://keras.io/examples/vision/cutmix/)
- [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/pdf/1710.09412.pdf)
- [MixUp augmentation for image classification](https://keras.io/examples/vision/mixup/)
- [Multi-head or Single-head? An Empirical Comparison for Transformer Training](https://arxiv.org/pdf/2106.09650.pdf)
- [Getting 95% Accuracy on the Caltech101 Dataset using Deep Learning](https://debuggercafe.com/getting-95-accuracy-on-the-caltech101-dataset-using-deep-learning/)
- [How to use CutMix and MixUp](https://pytorch.org/vision/main/auto_examples/transforms/plot_cutmix_mixup.html)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/latest/)
