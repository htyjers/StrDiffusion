# StrDiffusion

This repository is the official code for the paper "Structure Matters: Tackling the Semantic Discrepancy in Diffusion Models for Image Inpainting" by Haipeng Liu (hpliu_hfut@hotmail.com), Yang Wang (corresponding author: yangwang@hfut.edu.cn), Biao Qian, Meng Wang, Yong Rui. *CVPR 2024, Seattle, USA*
#

#
## Dependenices

* OS: Ubuntu 20.04.6
* nvidia :
	- cuda: 12.3
	- cudnn: 8.5.0
* python3
* pytorch >= 1.13.0
* Python packages: `pip install -r requirements.txt`

## Train Structure Denoising Model
1. Dataset Preparation:
   
   Download mask and image datasets, then get into the `StrDiffusion/train/structure` directory and modify the dataset paths in option files in `/config/inpainting/options/train/ir-sde.yml`
   * *You can set the mask path in [here](https://github.com/htyjers/StrDiffusion/blob/5749a214bb39754be165fa2bf76f96f13bc3e4a3/train/structure/config/inpainting/options/train/ir-sde.yml#L15)*
   * *You can set the image path in [here](https://github.com/htyjers/StrDiffusion/blob/5749a214bb39754be165fa2bf76f96f13bc3e4a3/train/structure/config/inpainting/options/train/ir-sde.yml#L22)*

2. Run the following command:
```
Python3 ./train/structure/config/inpainting/train.py
```

## Train Texture Denoising Model
1. Dataset Preparation:
   
   Download mask and image datasets, then get into the `StrDiffusion/train/texture` directory and modify the dataset paths in option files in `/config/inpainting/options/train/ir-sde.yml`
   * *You can set the mask path in [here](https://github.com/htyjers/StrDiffusion/blob/5749a214bb39754be165fa2bf76f96f13bc3e4a3/train/texture/config/inpainting/options/train/ir-sde.yml#L15)*
   * *You can set the image path in [here](https://github.com/htyjers/StrDiffusion/blob/5749a214bb39754be165fa2bf76f96f13bc3e4a3/train/texture/config/inpainting/options/train/ir-sde.yml#L22)*

2. Run the following command:
```
Python3 ./train/texture/config/inpainting/train.py
```

## Train Discriminator Network
1. Dataset Preparation:

   Download mask and image datasets, then get into the `StrDiffusion/train/discriminator` directory and modify the dataset paths in option files in `/config/inpainting/options/train/ir-sde.yml`
   * *You can set the mask path in [here](https://github.com/htyjers/StrDiffusion/blob/5749a214bb39754be165fa2bf76f96f13bc3e4a3/train/discriminator/config/inpainting/options/train/ir-sde.yml#L15)*
   * *You can set the image path in [here](https://github.com/htyjers/StrDiffusion/blob/5749a214bb39754be165fa2bf76f96f13bc3e4a3/train/discriminator/config/inpainting/options/train/ir-sde.yml#L22)*
     
2. Run the following command:
```
Python3 ./train/discriminator/config/inpainting/train.py
```

## Test StrDiffusion
1. Dataset Preparation:

   Download mask and image datasets, then get into the `StrDiffusion/test/texture` directory and modify the dataset paths in option files in `/config/inpainting/options/test/ir-sde.yml`
   * *You can set the mask path in [here](https://github.com/htyjers/StrDiffusion/blob/5749a214bb39754be165fa2bf76f96f13bc3e4a3/test/texture/config/inpainting/options/test/ir-sde.yml#L15)*
   * *You can set the image path in [here](https://github.com/htyjers/StrDiffusion/blob/5749a214bb39754be165fa2bf76f96f13bc3e4a3/test/texture/config/inpainting/options/test/ir-sde.yml#L23)*
     
2. Pre-trained models:
   Download the pre-trained model of [Places2, T=400](https://pan.baidu.com/s/1vxZ57te6TratZwKsuUYV8Q?pwd=n8dr), [Places2, T=100](https://pan.baidu.com/s/1tJIDNg1je6OBebViq-4wyA?pwd=pr8o) or [PSV, T=100](https://pan.baidu.com/s/1qX0FyehsM1Vl54PPv5N5yA?pwd=63se), then get into the `StrDiffusion/test/texture` directory and modify the dataset paths in option files in `/config/inpainting/options/test/ir-sde.yml`
   * *You can set the path of Texture Denoising Model in [here](https://github.com/htyjers/StrDiffusion/blob/5749a214bb39754be165fa2bf76f96f13bc3e4a3/test/texture/config/inpainting/options/test/ir-sde.yml#L44)*
   * *You can set the path of Structure Denoising Model in [here](https://github.com/htyjers/StrDiffusion/blob/5749a214bb39754be165fa2bf76f96f13bc3e4a3/test/texture/config/inpainting/options/test/ir-sde.yml#L45)*
   * *You can set the path of Discriminator Network in [here](https://github.com/htyjers/StrDiffusion/blob/5749a214bb39754be165fa2bf76f96f13bc3e4a3/test/texture/config/inpainting/options/test/ir-sde.yml#L46)*
    
3. For different T, you can set the corresponding the hyperparameters of resampling strategy in [here](https://github.com/htyjers/StrDiffusion/blob/5bb611b8e2586039b88608fd6494aef7cd5db3a9/test/texture/config/inpainting/utils/sde_utils.py#L302-L313)
   
4. Run the following command:
```
Python3 ./test/texture/config/inpainting/test.py
```


#
## Example Results

- Visual comparison between our method and the competitors.

![](https://github.com/htyjers/StrDiffusion/tree/main/image/image1.bmp)

- Visualization of the denoised results for IR-SDE and StrDiffusion during the denoising process,

![](https://github.com/htyjers/StrDiffusion/tree/main/image/image2.png)
