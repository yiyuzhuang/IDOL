# **IDOL: Instant Photorealistic 3D Human Creation from a Single Image**  

[![Website](https://img.shields.io/badge/Project-Website-0073e6)](https://yiyuzhuang.github.io/IDOL/)
[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/pdf/2412.14963)
[![Live Demo](https://img.shields.io/badge/Live-Demo-34C759)](https://yiyuzhuang.github.io/IDOL/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)


---

<p align="center">
  <img src="./asset/images/Teaser_v2.png" alt="Teaser Image for IDOL" width="85%">
</p>

---

## **Abstract**

This work introduces **IDOL**, a feed-forward, single-image human reconstruction framework that is fast, high-fidelity, and generalizable. 
Leveraging a large-scale dataset of 100K multi-view subjects, our method demonstrates exceptional generalizability and robustness in handling diverse human shapes, cross-domain data, severe viewpoints, and occlusions. 
With a uniform structured representation, the reconstructed avatars are directly animatable and easily editable, providing a significant step forward for various applications in graphics, vision, and beyond.

In summary, this project introduces:

- **IDOL**: A scalable pipeline for instant photorealistic 3D human reconstruction using a simple yet efficient feed-forward model.
- **HuGe100K Dataset**: We develop a data generation pipeline and present \datasetname, a large-scale multi-view human dataset featuring diverse attributes, high-fidelity, high-resolution appearances, and a well-aligned SMPL-X model.
- **Application Support**: Enabling 3D human reconstruction and downstream tasks such as editing and animation.


---
## 📰 **News** 

- **2024-12-18**: Paper is now available on arXiv.
- **2025-01-02**: The demo dataset containing 100 samples is now available for access. The remaining dataset is currently undergoing further cleaning and review.
- **2025-03-01**: 🎉 Paper accepted by CVPR 2025.
- **2025-03-01**: 🎉 We have released the inference code! Check out the [Code Release](#code-release) section for details.
- **2025-04-01**: 🔥 Full HuGe100K dataset is now available! See the [Dataset Access](#dataset-demo-access) section.
- **2025-04-05**: 🔥 Training code is now available! Check out the [Training Code](#training-code) section for details.

## 🚧 **Project Status**   

We are actively working on releasing the following resources:  

| Resource                    | Status              | Expected Release Date      |
|-----------------------------|---------------------|----------------------------|
| **Dataset Demo**            | ✅ Available        | **Now Live! (2025.01.02)**      |
| **Inference Code**             | ✅ Available        | **Now Live! (2025.03.01)**   |
| **Full Dataset Access**     | ✅ Available        | **Now Live! (2025.04.01)**   |
| **Online Demo**             | 🚧 In Progress      | **Before April  2025**   |
| **Training Code**                    | ✅ Available     | **Now Live! (2025.04.05)**   |

Stay tuned as we update this section with new releases! 🚀  



## 💻 **Code Release** 

### Installation & Environment Setup

Please refer to [env/README.md](env/README.md) for detailed environment setup instructions.

### Quick Start
Run demo with different modes:
```bash
# Reconstruct the input image
python run_demo.py --render_mode reconstruct --low_ram

# Generate novel poses (animation)
python run_demo.py --render_mode novel_pose --low_ram

# Generate 360-degree view
python run_demo.py --render_mode novel_pose_A --low_ram
```

### Training

#### Data Preparation

1. **Dataset Structure**: First, prepare your dataset with the following structure:
   ```
   dataset_root/
   ├── deepfashion/
   │   ├── image1/
   │   │   ├── videos/
   │   │   │   ├── xxx.mp4
   │   │   │   └── xxx.jpg
   │   │   └── param/
   │   │       └── xxx.npy
   │   └── image2/
   │       ├── videos/
   │       └── param/
   └── flux_batch1_5000/
       ├── image1/
       │   ├── videos/
       │   └── param/
       └── image2/
           ├── videos/
           └── param/
   ```

2. **Process Dataset**: Run the data processing script to generate cache files:
   ```bash
   # Process the dataset and generate cache files
   # Please modify the dataset path and the sample number in the script
   bash data_processing/process_datasets.sh
   ```

   This will generate cache files in the `processed_data` directory:
   - `deepfashion_train_140.npy`
   - `deepfashion_val_10.npy`
   - `deepfashion_test_50.npy`
   - `flux_batch1_5000_train_140.npy`
   - `flux_batch1_5000_val_10.npy`
   - `flux_batch1_5000_test_50.npy`

3. **Configure Cache Path**: Update the cache path in your config file (e.g., `configs/idol_v0.yaml`):
   ```yaml
     params:
       cache_path: [
         ./processed_data/deepfashion_train_140.npy,
         ./processed_data/flux_batch1_5000_train_140.npy
       ]
   ```

#### Training

1. **Single-Node Training**: For single-node multi-GPU training:
   ```bash
   python train.py \
     --base configs/idol_v0.yaml \
     --num_nodes 1 \
     --gpus 0,1,2,3,4,5,6,7
   ```

2. **Multi-Node Training**: For multi-node training, specify additional parameters:
   ```bash
   python train.py \
     --base configs/idol_v0.yaml \
     --num_nodes <total_nodes> \
     --node_rank <current_node_rank> \
     --master_addr <master_node_ip> \
     --master_port <port_number> \
     --gpus 0,1,2,3,4,5,6,7
   ```

   Example for a 2-node setup:
   ```bash
   # On master node (node 0):   
   python train.py --base configs/idol_v0.yaml --num_nodes 2 --node_rank 0 --master_addr 192.168.1.100 --master_port 29500 --gpus 0,1,2,3,4,5,6,7

   # On worker node (node 1):
   python train.py --base configs/idol_v0.yaml --num_nodes 2 --node_rank 1 --master_addr 192.168.1.100 --master_port 29500 --gpus 0,1,2,3,4,5,6,7
   ```

3. **Resume Training**: To resume training from a checkpoint:
   ```bash
   python train.py \
     --base configs/idol_v0.yaml \
     --resume PATH/TO/MODEL.ckpt \
     --num_nodes 1 \
     --gpus 0,1,2,3,4,5,6,7
   ```

4. **Test and Evaluate Metrics**:
   ```bash
   python train.py \
     --base configs/idol_v0.yaml \                # Main config file (model)
     --num_nodes 1 \
     --gpus 0,1,2,3,4,5,6,7 \
     --test_sd /path/to/model_checkpoint.ckpt \   # Path to the .ckpt model you want to test
      --test_dataset ./configs/test_dataset.yaml   # (Optional) Dataset config used specifically for testing
   ```

## Notes
- Make sure all GPUs have enough memory for the selected batch size
- For multi-node training, ensure network connectivity between nodes
- Monitor training progress using the logging system
- Adjust learning rate and other hyperparameters in the config file as needed


## 🌐 **Key Links** 

- 📄 [**Paper on arXiv**](https://arxiv.org/pdf/2412.02684)  
- 🌐 [**Project Website**](https://yiyuzhuang.github.io/IDOL/)  
- 🚀 [**Live Demo**](https://your-live-demo-link.com) (Coming Soon!)  

---

## 📊 **Dataset Demo Access**   

We introduce **HuGe100K**, a large-scale multi-view human dataset, supporting 3D human reconstruction and animation research.  

### ▶ **Watch the Demo Video**
<p align="center">
  <img src="./asset/videos/dataset.gif" alt="Dataset GIF" width="85%">
</p>

### 📋 **Dataset Documentation**
For detailed information about the dataset format, structure, and usage guidelines, please refer to our [Dataset Documentation](dataset/README.md).

### 🚀 **Access the Dataset**   

<div align="center">
  <p><strong>🔥 HuGe100K - The largest multi-view human dataset with 100,000+ subjects! 🔥</strong></p>
  <p>High-resolution • Multi-view • Diverse poses • SMPL-X aligned</p>
  

  <a href="https://docs.google.com/forms/d/e/1FAIpQLSeVqrA9Mc_ODdcTZsB3GgrxgSNZk5deOzK4f64N72xlQFhvzQ/viewform?usp=dialog">
    <img src="https://img.shields.io/badge/Apply_for_Access-HuGe100K_Dataset-FF6B6B?style=for-the-badge&logo=googleforms&logoColor=white" alt="Apply for Access" width="300px">
  </a>
  <p><i>Complete the form to get access credentials and download links!</i></p>
</div>

### ⚖️ **License and Attribution**

This dataset includes images derived from the **DeepFashion** dataset, originally provided by MMLAB at The Chinese University of Hong Kong. The use of DeepFashion images in this dataset has been explicitly authorized by the original authors solely for the purpose of creating and distributing this dataset. **Users must not further reproduce, distribute, sell, or commercially exploit any images or derived data originating from DeepFashion.** For any subsequent or separate use of the DeepFashion data, users must directly obtain authorization from MMLAB and comply with the original [DeepFashion License](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html).

---

## 📝 **Citation**   

If you find our work helpful, please cite us using the following BibTeX:

```bibtex
@article{zhuang2024idolinstant,                
  title={IDOL: Instant Photorealistic 3D Human Creation from a Single Image}, 
  author={Yiyu Zhuang and Jiaxi Lv and Hao Wen and Qing Shuai and Ailing Zeng and Hao Zhu and Shifeng Chen and Yujiu Yang and Xun Cao and Wei Liu},
  journal={arXiv preprint arXiv:2412.14963},
  year={2024},
  url={https://arxiv.org/abs/2412.14963}, 
}
```



## **License** 

This project is licensed under the **MIT License**.

- **Permissions**: This license grants permission to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software.
- **Condition**: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
- **Disclaimer**: The software is provided "as is", without warranty of any kind.

For more information, see the full license [here](https://opensource.org/licenses/MIT).

## **Support Our Work** ⭐

If you find our work useful for your research or applications:

- Please ⭐ **star our repository** to help us reach more people
- Consider **citing our paper** in your publications (see [Citation](#citation) section)
- Share our project with others who might benefit from it

Your support helps us continue developing open-source research projects like this one!

## 📚 **Acknowledgments**

This project is majorly built upon several excellent open-source projects:

- [E3Gen](https://github.com/olivia23333/E3Gen): Efficient, Expressive and Editable Avatars Generation
- [SAPIENS](https://github.com/facebookresearch/sapiens): High-resolution visual models for human-centric tasks
- [GeoLRM](https://github.com/alibaba-yuanjing-aigclab/GeoLRM): Large Reconstruction Model for High-Quality 3D Generation
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting): Real-Time 3DGS Rendering

We thank all the authors for their contributions to the open-source community.
