# FACES: Facial Analysis with Compressed Efficient Systems

[![Paper](https://img.shields.io/badge/Paper-ScienceDirect-blue.svg)](https://www.sciencedirect.com/science/article/pii/S2405959526000299)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.icte.2026.02.008-B31B1B.svg)](https://doi.org/10.1016/j.icte.2026.02.008)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of the paper:  
**"[FACES: Facial analysis with compressed efficient systems](https://www.sciencedirect.com/science/article/pii/S2405959526000299)"** (ICT Express, 2026).

*Authors: Romanela Lajić, Peter Peer, Vitomir Štruc, Dong Seog Han, Blaž Meden, Žiga Emeršič.*

---

## 📌 Overview

Due to their promising performance, Vision Transformers (ViTs) are increasingly being incorporated into various biometric solutions, particularly in the domain of face analysis. However, their size and computational expense remain a significant challenge for deployment on resource-constrained devices. 

**FACES** introduces a framework for compressing these models, reducing their computational footprint and memory requirements while preserving high accuracy for facial analysis tasks. 

---

## 🚀 Getting Started

### Prerequisites

The code has been developed and tested with **Python 3.9** and **PyTorch 2.2.0** (CUDA 12.1). We recommend using [Anaconda](https://www.anaconda.com/) to manage your environment.

### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/romanela-lajic/faces.git](https://github.com/romanela-lajic/faces.git)
   cd faces
2. **Create the enviroment:**
   ```bash
    conda create -n faces-env python=3.9 -y
    conda activate faces-env

3. **Install the requirements:**
   ```bash
     pip install torch==2.2.0 torchvision==0.17.0 --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
     pip install -r requirements.txt

4. **Run MS1MV2 training:**
     ```bash
     python train_tf_pruned.py
5. **Run CelebA training:**
     ```bash
     python attributes_tf.py

6. **Run CelebA testing:**
     ```bash
     python attributes_tf.py --eval

## ✍️ Citation
## If you use this code or models in your research, please cite our paper:


```bibtex
@article{lajic2026faces,
  title={FACES: Facial analysis with compressed efficient systems},
  author={Laji{\'c}, Romanela and Peer, Peter and {\v{S}}truc, Vitomir and Han, Dong Seog and Meden, Bla{\v{z}} and Emer{\v{s}}i{\v{c}}, {\v{Z}}iga},
  journal={ICT Express},
  year={2026},
  publisher={Elsevier},
  doi={10.1016/j.icte.2026.02.008}
}

     
     
