# ISPRS P&PS 2025: CGSL: Commonality Graph Structure Learning for Unsupervised Multimodal Change Detection

![Paper](https://img.shields.io/badge/Paper-ISPRS%20P&RS-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)

This repository provides the official implementation of the paper:

Jianjian Xu, Tongfei Liu*, Tao Lei, Hongruixuan Chen, Naoto Yokoya, Zhiyong Lv, and Maoguo Gong[J]. [CGSL: Commonality Graph Structure Learning for Unsupervised Multimodal Change Detection](https://www.sciencedirect.com/science/article/pii/S0924271625003223),
**ISPRS Journal of Photogrammetry and Remote Sensing, 2025**, 229:92-106.

---

## 🚀 Features

- ✅ Unsupervised multimodal change detection  
- ✅ Graph-based commonality structure learning  
- ✅ Siamese graph encoder with probabilistic latent representation  
- ✅ Composite loss (reconstruction, KL divergence, commonality)  
- ✅ Support for **optical, SAR, NDVI**, and other multimodal combinations  

---

## 📖 Abstract

Multimodal change detection (MCD) has attracted a great deal of attention due to its significant advantages in processing heterogeneous remote sensing images (RSIs) from different sensors (e.g., optical and synthetic aperture radar). The major challenge of MCD is that it is difficult to acquire the changed areas by directly comparing heterogeneous RSIs. Although many MCD methods have made important progress, they are still insufficient in capturing the modality-independence complex structural relationships in the feature space of heterogeneous RSIs. To this end, we propose a novel commonality graph structure learning (CGSL) for unsupervised MCD, which aims to extract potential commonality graph structural features between heterogeneous RSIs and directly compare them to detect changes. In this study, heterogeneous RSIs are first segmented and constructed as superpixel-based heterogeneous graph structural data consisting of nodes and edges. Then, the heterogeneous graphs are input into the proposed CGSL to capture the commonalities of graph structural features with modality-independence. The proposed CGSL consists of a Siamese graph encoder and two graph decoders. The Siamese graph encoder maps heterogeneous graphs into a shared space and effectively extracts potential commonality in graph structural features from heterogeneous graphs. The two graph decoders reconstruct the mapped node features as original node features to maintain consistency with the original graph features. Finally, the changes between heterogeneous RSIs can be detected by measuring the differences in commonality graph structural features using the mean squared error. In addition, we design a composite loss with regularization to guide CGSL in effectively excavating the potential commonality graph structural features between heterogeneous graphs in an unsupervised learning manner. Extensive experiments on seven MCD datasets show that the proposed CGSL outperforms the existing state-of-the-art methods, demonstrating its superior performance in MCD.

The framework of the proposed CGSL is presented as follows:
![Framework of our proposed CGSL](https://github.com/TongfeiLiu/CGSL-for-MCD/blob/main/Figures/Freamwork%20of%20CGSL.png)

---

## 📊 Results Preview
**Change Maps**
![Change maps of the proposed CGSL](https://github.com/TongfeiLiu/CGSL-for-MCD/blob/main/Figures/BCIs.png)

Note: If you want to get the difference map and binary map in our paper, you can get them directly through the "Results of Our CGSL" folder.

## 📦 Installation

### Requirements
- Python 3.8+  
- PyTorch 1.9+  
- scikit-image  
- scikit-learn  
- NumPy  

### Install Dependencies
```bash
pip install torch torchvision scikit-image scikit-learn numpy
```

---

## 📁 Dataset Preparation

Organize your datasets as follows:

```text
root_dir/
├── dataset1/
│   ├── t1_image.tif
│   ├── t2_image.tif
│   └── ground_truth.tif
├── dataset2/
│   ├── ...
```

- Supported image types: `.tif`, `.png`, `.jpg`  
- Each dataset should contain:  
  - **T1 image** (pre-event)  
  - **T2 image** (post-event)  
  - **Ground truth image** (if available)  

---

## 🛠 Usage

### Training and Inference

Run the “main.py” script with appropriate arguments:

```bash
python main.py \
  --root_dir /path/to/root \
  --load_data_dir /path/to/dataset \
  --result_folder ./results \
  --img_t1_name t1_image.tif \
  --img_t2_name t2_image.tif \
  --ref_name ground_truth.tif \
  --type_a optical \
  --type_b sar \
  --n_seg 1900 \
  --com 32 \
  --item 1 \
  --epochs 50 \
  --lr 0.0001 \
  --weight_decay 0.0001 \
  --lambda_reg 0.001
```

### Parameters

| Argument       | Type  | Default  | Description |
|----------------|-------|----------|-------------|
| `root_dir`     | str   | `''`     | Root directory for datasets |
| `load_data_dir`| str   | `''`     | Subdirectory of the dataset |
| `result_folder`| str   | `''`     | Folder to save results |
| `img_t1_name`  | str   | `''`     | Pre-event image filename |
| `img_t2_name`  | str   | `''`     | Post-event image filename |
| `ref_name`     | str   | `''`     | Ground truth image filename |
| `type_a`       | str   | `''`     | Modality of first image (e.g., optical, sar) |
| `type_b`       | str   | `''`     | Modality of second image (e.g., optical, sar) |
| `n_seg`        | int   | `0`      | Number of superpixels for SLIC |
| `com`          | float | `0`      | Compactness parameter for SLIC |
| `item`         | int   | `5`      | Dataset index (for logging) |
| `epochs`       | int   | `50`     | Number of training epochs |
| `lr`           | float | `0.0001` | Learning rate |
| `weight_decay` | float | `0.0001` | Weight decay for optimizer |
| `lambda_reg`   | float | `0.001`  | L2 regularization coefficient |

---

## 📊 Recommended Hyperparameters

Based on the paper, here are the suggested parameters for each dataset:

| Dataset | n_seg | com |
|---------|-------|-----|
| #1      | 1900  | 32  |
| #2      | 2800  | 0.4 |
| #3      | 1100  | 30  |
| #4      | 1100  | 42  |
| #5      | 1400  | 30  |
| #6      | 1800  | 0.4 |
| #7      | 1500  | 32  |

---

## 📈 Results Output

The method outputs:
- **Difference Image (DI)**  
- **Binary Change Image (BCI)** via Otsu thresholding  
- Evaluation metrics: **OA, Kappa, F1, AUR, AUP**  

CGSL outperforms multiple SOTA methods (e.g., SR-GCAE, CACD, PRBCD-Net, BGAAE).  
For detailed results, please check **Tables 2–3** and **Figures 4–6** in the paper.

---

## 📜 Citation

If you use this code or our paper, please cite:

```bibtex
@article{ISPRS2025CGSL,
  title={CGSL: Commonality Graph Structure Learning for Unsupervised Multimodal Change Detection},
  author={Xu, Jianjian and Liu, Tongfei and Lei, Tao and Chen, Hongruixuan and Yokoya, Naoto and Lv, Zhiyong and Gong, Maoguo},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={229},
  pages={92--106},
  year={2025},
  publisher={Elsevier}
}

@article{TIP2025CFRL,
  author={Liu, Tongfei and Zhang, Mingyang and Gong, Maoguo and Zhang, Qingfu and Jiang, Fenlong and Zheng, Hanhong and Lu, Di},
  journal={IEEE Transactions on Image Processing}, 
  title={Commonality Feature Representation Learning for Unsupervised Multimodal Change Detection}, 
  year={2025},
  volume={34},
  number={},
  pages={1219-1233},
  keywords={Feature extraction;Image reconstruction;Training;Data mining;Autoencoders;Representation learning;Image sensors;Electronic mail;Decoding;Clustering algorithms;Multimodal change detection;unsupervised change detection;heterogeneous images;representation learning;commonality feature},
  doi={10.1109/TIP.2025.3539461}}
```

---

## 📮 Contact

For questions or suggestions, please open an issue or contact:

- Jianjian Xu: xujianjian.Leo@sust.edu.cn  
- Tongfei Liu: liutongfei_home@hotmail.com  

---

