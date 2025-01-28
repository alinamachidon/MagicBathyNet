Modified README of: MagicBathyNet: A Multimodal Remote Sensing Dataset for Benchmarking Learning-based Bathymetry and Pixel-based Classification in Shallow Waters


[MagicBathyNet](https://www.magicbathy.eu/magicbathynet.html) is a benchmark dataset made up of image patches of Sentinel-2, SPOT-6 and aerial imagery, bathymetry in raster format and seabed classes annotations. Dataset also facilitates unsupervised learning for model pre-training in shallow coastal areas. It is developed in the context of MagicBathy project.


## Downloading the dataset

For downloading the dataset and a detailed explanation of it, please visit the MagicBathy Project website at [https://www.magicbathy.eu/magicbathynet.html](https://www.magicbathy.eu/magicbathynet.html)

## Dataset structure
The folder structure should be as follows:
```
┗ 📂 magicbathynet/
  ┣ 📂 agia_napa/
  ┃ ┣ 📂 img/
  ┃ ┃ ┣ 📂 aerial/
  ┃ ┃ ┃ ┣ 📜 img_339.tif
  ┃ ┃ ┃ ┣ 📜 ...
  ┃ ┃ ┣ 📂 s2/
  ┃ ┃ ┃ ┣ 📜 img_339.tif
  ┃ ┃ ┃ ┣ 📜 ...
  ┃ ┃ ┣ 📂 spot6/
  ┃ ┃ ┃ ┣ 📜 img_339.tif
  ┃ ┃ ┃ ┣ 📜 ...
  ┃ ┣ 📂 depth/
  ┃ ┃ ┣ 📂 aerial/
  ┃ ┃ ┃ ┣ 📜 depth_339.tif
  ┃ ┃ ┃ ┣ 📜 ...
  ┃ ┃ ┣ 📂 s2/
  ┃ ┃ ┃ ┣ 📜 depth_339.tif
  ┃ ┃ ┃ ┣ 📜 ...
  ┃ ┃ ┣ 📂 spot6/
  ┃ ┃ ┃ ┣ 📜 depth_339.tif
  ┃ ┃ ┃ ┣ 📜 ...
  ┃ ┣ 📂 gts/
  ┃ ┃ ┣ 📂 aerial/
  ┃ ┃ ┃ ┣ 📜 gts_339.tif
  ┃ ┃ ┃ ┣ 📜 ...
  ┃ ┃ ┣ 📂 s2/
  ┃ ┃ ┃ ┣ 📜 gts_339.tif
  ┃ ┃ ┃ ┣ 📜 ...
  ┃ ┃ ┣ 📂 spot6/
  ┃ ┃ ┃ ┣ 📜 gts_339.tif
  ┃ ┃ ┃ ┣ 📜 ...
  ┃ ┣ 📜 [modality]_split_bathymetry.txt
  ┃ ┣ 📜 [modality]_split_pixel_class.txt
  ┃ ┣ 📜 norm_param_[modality]_an.txt
  ┃
  ┣ 📂 puck_lagoon/
  ┃ ┣ 📂 img/
  ┃ ┃ ┣ 📜 ...
  ┃ ┣ 📂 depth/
  ┃ ┃ ┣ 📜 ...
  ┃ ┣ 📂 gts/
  ┃ ┃ ┣ 📜 ...
  ┃ ┣ 📜 [modality]_split_bathymetry.txt
  ┃ ┣ 📜 [modality]_split_pixel_class.txt
  ┃ ┣ 📜 norm_param_[modality]_pl.txt


## Installation Guide
The requirements are easily installed via Anaconda (recommended):

`conda env create -f environment.yml`

After the installation is completed, activate the environment:

`conda activate magicbathynet`

## Train and Test the bathymetry models
To train and test the **bathymetry** models use **run_bathymetry.ipynb**.

## Pre-trained Deep Learning Models
The code and model weights for the following deep learning models that have been pre-trained on MagicBathyNet for bathymetry tasks:

### Learning-based Bathymetry
| Model Name | Modality | Area | Pre-Trained PyTorch Models                                                                                                                | 
| ----------- |----------| ---- |----------------------------------------------------------------------------------------------------------------------------------------------|
| Modified U-Net for bathymetry | Aerial | Agia Napa | [bathymetry_aerial_an.zip](https://drive.google.com/file/d/1-qUlQMHdZDZKkeQ4RLX54o4TK6juwOqD/view?usp=sharing) |
| Modified U-Net for bathymetry | Aerial | Puck Lagoon         | [bathymetry_aerial_pl.zip](https://drive.google.com/file/d/1SN8YH-WZIdR4e5Zl0uQK4OM62z_WNCks/view?usp=sharing)            |
| Modified U-Net for bathymetry | SPOT-6 | Agia Napa        | [bathymetry_spot6_an.zip](https://drive.google.com/file/d/1giG-MrJQZ2YLDzjOd2h-u2vr9gfI1jO0/view?usp=sharing)            |
| Modified U-Net for bathymetry | SPOT-6 | Puck Lagoon      | [bathymetry_spot6_pl.zip](https://drive.google.com/file/d/1Cf1gAsEUfACkBep4i_0gB-pp_L0bvaU_/view?usp=sharing)      |
| Modified U-Net for bathymetry | Sentinel-2 | Agia Napa    | [bathymetry_s2_an.zip](https://drive.google.com/file/d/15esoghCHHHilQJxTBBjmHpAAde-AHdtE/view?usp=sharing)   | 
| Modified U-Net for bathymetry | Sentinel-2 | Puck Lagoon    | [bathymetry_s2_pl.zip](https://drive.google.com/file/d/1oCnD5ePwVW3ORix4GWRcMUp_kSL5p9Se/view?usp=sharing)   |

To achieve the results presented in the paper, use the parameters and the specific train-evaluation splits provided in the dataset. Parameters can be found [here](https://drive.google.com/file/d/1gkIG99WFI6LNP7gsRvae9FZWU3blDPgv/view?usp=sharing) while train-evaluation splits are included in the dataset.

## Example testing results
Example patch of the Agia Napa area (left) predicted bathymetry obtained by MagicBathy-U-Net. 
![depth_410_aerial](https://github.com/pagraf/MagicBathyNet/assets/35768562/7995efd7-f85e-4411-8037-4a68c9780bfb)


