# MedCAM: A Hybrid Model for Medical Image Segmentation

### Introduction
MedCAM is a hybrid model designed for precise medical image segmentation by integrating **Convolutional Neural Networks (CNNs)** and **Vision Transformers (ViTs)**. The model leverages the strengths of CNNs for local feature extraction and ViTs for capturing global context, providing a robust solution for clinical diagnostics.

Medical image segmentation is crucial for identifying tumors, lesion boundaries, and other critical regions, aiding doctors in faster and more accurate decision-making. MedCAM addresses the limitations of conventional methods by creating a computationally efficient model that balances global and local feature understanding.

---

### Features
- **Hybrid Architecture**: Combines MaxViT-based encoder and CAM-based decoder for enhanced segmentation.
- **Advanced Preprocessing**: Includes intensity normalization, resizing, and augmentation for robust training.
- **Visualization Tools**: Graphical insights into segmentation boundaries and model predictions.
- **Flexible Implementation**: Supports Synapse and ACDC datasets for segmentation tasks.

---

### Model Architecture

#### MaxViT Encoder
- Hybrid convolution + attention design for local and global feature extraction.
- Hierarchical multi-resolution blocks ensure detailed representation.
- Efficient axial and block attention mechanisms reduce computational overhead.

#### CAM Decoder
- **Channel Attention Mechanism** dynamically focuses on relevant feature channels.
- Precise transposed convolutions for accurate upsampling.
- Skip connections for enhanced boundary segmentation and feature refinement.

---

### Data Preprocessing
1. **Data Loading**: CT images and segmentation masks loaded from `.nii.gz` files using `nibabel`.
2. **Clipping and Normalization**: Intensity values clipped between `-125` to `275` and normalized to `[0, 1]`.
3. **Resizing**: All slices resized to `256x256` pixels for uniformity.
4. **Augmentation**:
   - Random rotations (`-20°` to `+20°`).
   - Horizontal and vertical flipping.
   - Scaling to simulate size variations.
5. **Saving and Visualization**: Preprocessed data stored as `.npz` and `.h5` files, with visualization for verification.

---

### Results
- **Preliminary Results**:
  - Effective boundary segmentation achieved.
- **Visualization**:
  - Segmentation masks color-coded for organ identification.
  - Statistical analysis of organ diversity and area coverage.

---

### Future Work
- Fine-tune hyperparameters, including learning rate, weight decay, and loss weighting.
- Train and evaluate the model on the **ACDC dataset** for broader validation.
- Experiment with additional datasets to ensure generalizability.

---

### Tech Stack
- **Programming Languages**: Python  
- **Deep Learning Frameworks**: TensorFlow, PyTorch  
- **Data Handling**: NumPy, Pandas  
- **Data Preprocessing**: nibabel, OpenCV, scikit-image  
- **Visualization Tools**: Matplotlib, Seaborn  
- **Optimization Techniques**: Adam Optimizer, Gradient Clipping, Batch Normalization  
- **Augmentation Tools**: Albumentations, custom pipelines  
- **Data Formats**: NIfTI (`.nii.gz`), NumPy arrays (`.npz`), HDF5 (`.h5`)  
- **Development Tools**: Jupyter Notebooks, Google Colab  
- **Version Control**: Git, GitHub  

---

You have to refer to this citation for any use of the ACDC database

O. Bernard, A. Lalande, C. Zotti, F. Cervenansky, et al.
"Deep Learning Techniques for Automatic MRI Cardiac Multi-structures Segmentation and Diagnosis: Is the Problem Solved ?" in IEEE Transactions on Medical Imaging, vol. 37, no. 11, pp. 2514-2525, Nov. 2018
doi: 10.1109/TMI.2018.2837502




