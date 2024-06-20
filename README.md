# StrDiffusion

### [Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Structure_Matters_Tackling_the_Semantic_Discrepancy_in_Diffusion_Models_for_CVPR_2024_paper.html) | [BibTex](https://github.com/htyjers/StrDiffusion?tab=readme-ov-file#citation)

This repository is the official code for the paper "Structure Matters: Tackling the Semantic Discrepancy in Diffusion Models for Image Inpainting" by Haipeng Liu (hpliu_hfut@hotmail.com), Yang Wang (corresponding author: yangwang@hfut.edu.cn), Biao Qian, Meng Wang, Yong Rui. *CVPR 2024, Seattle, USA*

#
## Introduction
In this paper, we propose a novel structure-guided diffusion model for image inpainting (namely **StrDiffusion**), which reformulates the conventional texture denoising process under the guidance of the structure to derive a simplified denoising objective (**_Eq.11_**) for inpainting, while revealing:  1) the semantically sparse structure is beneficial to tackle the semantic discrepancy in the early stage, while the dense texture generates the reasonable semantics in the late stage;  2) the semantics from the unmasked regions essentially offer the time-dependent guidance for the texture denoising process, benefiting from the time-dependent sparsity of the structure semantics. For the denoising process, a structure-guided neural network is trained to estimate the simplified denoising objective by exploiting the consistency of the denoised structure between masked and unmasked regions. Besides, we devise an adaptive resampling strategy as a formal criterion on whether the structure is competent to guide the texture denoising process, while regulate their semantic correlations. 


![](image/image3.png)
<p align="center">Figure 1. Illustration of the proposed StrDiffusion pipeline.</p>

![](image/image4.png)
<p align="center">Figure 2. Illustration of the adaptive resampling strategy.</p>

In summary, our StrDiffusion reveals: 
- the semantically _sparse_ structure encourages the consistent semantics for the denoised results in the early stage, while the _dense_ texture carries out the semantic generation in the late stage; 
- The semantics from the unmasked regions essenially offer the time-dependent guidance for the texture denoising process, benefiting from the time-dependent sparsity of the structure semantics.
- We remark that whether the structure guides the texture well greatly depends on the semantic correlation between them. As inspired, an adaptive resampling strategy comes up to monitor the semantic correlation and regulate it via the resampling iteration

  
#
## Dependencies

* OS: Ubuntu 20.04.6
* nvidia :
	- cuda: 12.3
	- cudnn: 8.5.0
* python3
* pytorch >= 1.13.0
* Python packages: `pip install -r requirements.txt`

## Train-[Structure Denoising Model]
1. Dataset Preparation:
   
   Download mask and image datasets, then get into the `StrDiffusion/train/structure` directory and modify the dataset paths in option files in `/config/inpainting/options/train/ir-sde.yml`
   * *You can set the mask path in [here](https://github.com/htyjers/StrDiffusion/blob/5749a214bb39754be165fa2bf76f96f13bc3e4a3/train/structure/config/inpainting/options/train/ir-sde.yml#L15)*
   * *You can set the image path in [here](https://github.com/htyjers/StrDiffusion/blob/5749a214bb39754be165fa2bf76f96f13bc3e4a3/train/structure/config/inpainting/options/train/ir-sde.yml#L22)*

2. Run the following command:
```
Python3 ./train/structure/config/inpainting/train.py
```

## Train-[Texture Denoising Model]
1. Dataset Preparation:
   
   Download mask and image datasets, then get into the `StrDiffusion/train/texture` directory and modify the dataset paths in option files in `/config/inpainting/options/train/ir-sde.yml`
   * *You can set the mask path in [here](https://github.com/htyjers/StrDiffusion/blob/5749a214bb39754be165fa2bf76f96f13bc3e4a3/train/texture/config/inpainting/options/train/ir-sde.yml#L15)*
   * *You can set the image path in [here](https://github.com/htyjers/StrDiffusion/blob/5749a214bb39754be165fa2bf76f96f13bc3e4a3/train/texture/config/inpainting/options/train/ir-sde.yml#L22)*

2. Run the following command:
```
Python3 ./train/texture/config/inpainting/train.py
```

## Train-[Discriminator Network]
1. Dataset Preparation:

   Download mask and image datasets, then get into the `StrDiffusion/train/discriminator` directory and modify the dataset paths in option files in `/config/inpainting/options/train/ir-sde.yml`
   * *You can set the mask path in [here](https://github.com/htyjers/StrDiffusion/blob/5749a214bb39754be165fa2bf76f96f13bc3e4a3/train/discriminator/config/inpainting/options/train/ir-sde.yml#L15)*
   * *You can set the image path in [here](https://github.com/htyjers/StrDiffusion/blob/5749a214bb39754be165fa2bf76f96f13bc3e4a3/train/discriminator/config/inpainting/options/train/ir-sde.yml#L22)*
     
2. Run the following command:
```
Python3 ./train/discriminator/config/inpainting/train.py
```

## Test-[StrDiffusion]
1. Dataset Preparation:

   Download mask and image datasets, then get into the `StrDiffusion/test/texture` directory and modify the dataset paths in option files in `/config/inpainting/options/test/ir-sde.yml`
   * *You can set the mask path in [here](https://github.com/htyjers/StrDiffusion/blob/5749a214bb39754be165fa2bf76f96f13bc3e4a3/test/texture/config/inpainting/options/test/ir-sde.yml#L15)*
   * *You can set the image path in [here](https://github.com/htyjers/StrDiffusion/blob/5749a214bb39754be165fa2bf76f96f13bc3e4a3/test/texture/config/inpainting/options/test/ir-sde.yml#L23)*
     
2. Pre-trained models:
   
   Download the pre-trained model of [Places2, T=400](https://pan.baidu.com/s/1vxZ57te6TratZwKsuUYV8Q?pwd=n8dr), [Places2, T=100](https://pan.baidu.com/s/1yw8lE3Vk_3IsR3HoVf84Ww?_at_=1713468801558#list/path=%2F), then get into the `StrDiffusion/test/texture` directory and modify the model paths in option files in `/config/inpainting/options/test/ir-sde.yml`
   * *You can set the path of Texture Denoising Model in [here](https://github.com/htyjers/StrDiffusion/blob/5749a214bb39754be165fa2bf76f96f13bc3e4a3/test/texture/config/inpainting/options/test/ir-sde.yml#L44)*
   * *You can set the path of Structure Denoising Model in [here](https://github.com/htyjers/StrDiffusion/blob/5749a214bb39754be165fa2bf76f96f13bc3e4a3/test/texture/config/inpainting/options/test/ir-sde.yml#L45)*
   * *You can set the path of Discriminator Network in [here](https://github.com/htyjers/StrDiffusion/blob/5749a214bb39754be165fa2bf76f96f13bc3e4a3/test/texture/config/inpainting/options/test/ir-sde.yml#L46)*
    
3. For different [T](https://github.com/htyjers/StrDiffusion/blob/5749a214bb39754be165fa2bf76f96f13bc3e4a3/test/texture/config/inpainting/options/test/ir-sde.yml#L10), you can set the corresponding hyperparameters of adaptive resampling strategy in [here](https://github.com/htyjers/StrDiffusion/blob/772dd44b5595293d95efa5d92745ce48db185d66/test/texture/config/inpainting/utils/sde_utils.py#L295-L296)
   
4. Run the following command:
```
Python3 ./test/texture/config/inpainting/test.py
```


#
## Example Results

- Visual comparison between our method and the competitors.

![](image/image1.png)

- Visualization of the denoised results for IR-SDE and StrDiffusion during the denoising process,

![](image/image2.png)


#
## Citation

If any part of our paper and repository is helpful to your work, please generously cite with:

```
@InProceedings{Liu_2024_CVPR,
    author    = {Liu, Haipeng and Wang, Yang and Qian, Biao and Wang, Meng and Rui, Yong},
    title     = {Structure Matters: Tackling the Semantic Discrepancy in Diffusion Models for Image Inpainting},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {8038-8047}
}
```

This implementation is based on / inspired by:

* [https://github.com/Algolzw/image-restoration-sde](https://github.com/Algolzw/image-restoration-sde) (Image Restoration SDE)
* [https://github.com/andreas128/RePaint](https://github.com/andreas128/RePaint) (RePaint)
