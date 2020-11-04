# TF-2.3.0: TraVelGAN

* Reference: [TraVeLGAN: Image-to-image Translation by Transformation Vector Learning](https://arxiv.org/pdf/1902.09631.pdf)
* Reference: [from Pytorch](https://github.com/clementabary/travelgan)

## Implementation

* Tensorflow  2.3.0
* python 3.5.5
* Ubuntu 18.04

## Train

* Use LSGAN loss instead of WGAN (Different from orginal paper)
* Input size:  256 x 256 x 3 (original paper is 512 x 512 x 3)
* "FLAG" part
  * "load_size": Input size before cropping
  * "input_size": After crop the load size
  * "num_classes": Number of classes in training dataset
  * "A_txt_path":  Train A text path
  * "A_img_path":  Train A image path
  * "B_txt_path": Train B text path
  * "B_img_path": Train B image path
  * "train": Set "True" in bool type
  * "pre_checkpoint": Set "False" in bool type (if True, train continue)
  * "pre_checkpoint_path": if "pre_checkpoint" is True,  Set path where weight files were saved
  * "save_checkpoint": Path where weight files will be saved
  * "save_images": Path where sample images will be saved

## Test

* Add code that part of generate the sample images
