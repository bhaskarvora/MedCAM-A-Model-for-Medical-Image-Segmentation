# Medical Image Segmentation using Deep Learning

# Abstract 
 Medical image segmentation plays a very important role in clinical diagnosis for exact identification of
 tumors, lesions, and anatomical structures. Traditional CNNs, while effective for local feature extraction
 but they are not good in capturing long pixel dependencies; ViTs are developed for wider context but
 face inefficiency in extracting local features. This paper presents the MedCAM model, a hybrid approach
 that involves both CNNs and transformers. MedCAM adopts a multi-stage encoder for hierarchical multi
scale feature extraction and a Convolutional Attention Mixing (CAM) decoder that refines segmentation
 maps using a variety of state-of-the-art attention mechanisms like channel and spatial attention. Preliminary
 evaluations on the Synapse multi-organ dataset look promising, yet further refinements are in order to make
 these results more accurate. It balances accuracy with computational efficiency, hence, has the potential to
 revolutionize diagnostic capabilities in health care. Advanced attention mechanisms are built into MedCAM
 to capture local and global dependencies effectively for a significant improvement of medical images. This
 model has demonstrated strength in adapting to anatomical variation and is flexible with data variety. The
 methodology to be proposed has scalability; hence, it could also be deployed clinically.

## Introduction
This project focuses on medical image segmentation using a deep learning-based approach. The goal is to segment anatomical structures in medical images using a **Convolutional Attention Mixing Transformer (CAM) Decoder** integrated with a **customized CNN-based encoder**. The model is evaluated on standard medical datasets to demonstrate its effectiveness.

## Dataset
We used the following datasets for training and evaluation:
- **Synapse Multi-Organ Segmentation** dataset

## DATA SELECTION AND SAMPLING
 We have explored the Synapse 1 multi-organ dataset for this
 research. This dataset offers a comprehensive collection of
 medical images which allows our model to learn and segment
 images effectively. The dataset contains 3D medical images
 wherein different anatomical structures may be viewed. The
 quality of the data, the variety, and the presence of ground
 truth labels, which enable efficient training and testing of our
 model, were the selection criteria.
 The Synapse multi-organ dataset has 30 abdominal CT
 scans and their corresponding segmentation masks, where
 each CT scan has around 85-198 2D slices adding up to 3779
 axial contrast-enhanced abdominal slices. Each 2D slice is
 of resolution 512*512 has eight organs such as the aorta,
 gallbladder (GB), left kidney (KL), right kidney (KR), liver,
 pancreas (PC), spleen (SP), and stomach (SM), for segmenta
tion. We divide the dataset randomly into two sets: a training
 set consisting of 18 CT scans and a validation set with the
 remaining 12 CT scans.

 # Model Architecture
### **A. Convolutional Attention Mixing (CAM) Decoder**
The CAM decoder consists of sequential decoder blocks along with a bottleneck block. The key steps are:

- **Layer Normalization and Upsampling**: Each decoder block begins with layer normalization, followed by upsampling the input by a factor of two. This ensures that the output size is restored to match the encoder's resolution, reversing the progressive reduction during encoding.
- **Skip Connections and Depth-Wise Convolution (DWC)**: The upsampled input is concatenated with the corresponding encoder output via skip connections. It then passes through **depth-wise convolution (DWC)** for feature extraction and channel dimension reduction, helping to preserve fine-grained spatial details essential for accurate segmentation.
- **Multi-Head Self-Attention (MSA)**: Standard linear projection-based MSA is computationally expensive, so MedCAM replaces it with convolutional projections. This reduces the number of parameters while maintaining spatial detail and improving computational efficiency.
- **Channel Attention (CA) and Spatial Attention (SA)**: These attention modules focus on important regions for segmentation, capturing broader dependencies and providing better pixel-wise context.
- **Output Processing**: The decoder generates outputs in four stages, passing them through convolutional layers to match the number of segmentation classes. **Deep supervision** is applied to the last three outputs, ensuring high-quality segmentation results.

### **B. Encoder**
The encoder is responsible for hierarchical feature extraction at both local and global levels. The architecture includes the following steps:

- **Feature Extraction via Convolutional Blocks**: Two convolutional blocks, each comprising **3×3 convolutions, batch normalization, and LeakyReLU activation**, extract initial features from the input image.
- **MaxPooling**: This operation reduces pixel dimensions, extracting higher-level features while downsampling the input.
- **Skip Connections**: The output from the second convolution block is stored and later used in the decoder for precise localization, retaining spatial information.

```
```

# Data Visualization
![image](https://github.com/user-attachments/assets/14646750-64e4-4ebe-8931-0bdbf565c748)
The left image shows a raw CT scan, while the right image displays the segmented organs color-coded according to the legend.


![image](https://github.com/user-attachments/assets/451b6ace-2457-418d-93d6-ffd0744c0c15)
Donut Chart providing organ area coverage 

![image](https://github.com/user-attachments/assets/5b2b598e-bb22-4183-85dd-673d34a1ad41)
Bar Chart providing organ area coverage 


# Results


![image](https://github.com/user-attachments/assets/c0f88084-f366-4ff6-9955-c68515eb3d85)

![image](https://github.com/user-attachments/assets/3687cb43-6e6c-4e41-ad38-468f467f413a)

![image](https://github.com/user-attachments/assets/6c85b5e3-e618-496a-a8b7-9acafc2d316c)

![image](https://github.com/user-attachments/assets/d327ac1b-1413-4ef9-82f8-91b32c210dda)

![image](https://github.com/user-attachments/assets/ceabbe0c-9ad8-4184-9258-6bd4f82838db)


The model achieves the following performance:
| Dataset  | Dice Score | IoU |
|----------|------------|------|
| Synapse  | **0.8214**   | 0.7688 |

The qualitative results indicate precise segmentation, even for complex anatomical structures.

---

**Contributors:** Bhaskar Vora & Group 13
Research Project 
WIlfrid Laurier University
