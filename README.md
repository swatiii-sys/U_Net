
### Overview
This repository provides a specialized implementation of the U-Net architecture for the semantic segmentation of neuronal membranes. Utilizing the ISBI 2012 Challenge dataset, this project leverages TensorFlow and Keras to automate the identification of cellular boundaries in electron microscopy (EM) images. By employing a symmetric encoder-decoder framework with long-range skip connections, the model is capable of producing high-resolution biological masks crucial for connectomics and brain tissue mapping.

### Features
Biomedical-Centric Architecture: A U-Net backbone tailored for 256x256 grayscale EM imagery, ensuring high-fidelity extraction of thin membrane structures.

KaggleHub Integration: Seamless dataset acquisition using the kagglehub API for the ISBI-2012 challenge files.

Feature Preservation: Integrated concatenation layers that recover fine-grained spatial details lost during deep feature extraction.

Binarized Diagnostics: Post-processing logic to transform sigmoid probability maps into discrete, actionable binary neuronal masks.

### Introduction to the ISBI 2012 Challenge
The ISBI (International Symposium on Biomedical Imaging) 2012 Challenge focuses on the segmentation of neuronal structures in EM stacks. Accurate mapping of these membranes is a foundational requirement in connectomics, allowing researchers to reconstruct the 3D connectivity of neural circuits.

The U-Net architecture is the industry standard for this task because of its unique "U" shape:

The Contracting Path (Encoder): Captures the global context and identifies the presence of biological structures.

The Expansive Path (Decoder): Reconstructs the precise spatial localization of those structures, ensuring the final mask aligns perfectly with the original anatomical boundaries.

### Technical Architecture
Following the implementation provided in the source code, the network is built with a symmetric modular design:
Encoder: Two stages of double 3x3 convolutions ($16 \to 32$ filters) with ReLU activation and Max-Pooling to extract hierarchical features.
Bottleneck: A 64-filter convolutional bridge capturing the highest level of abstract latent features.
Decoder: Up-sampling blocks integrated with Concatenation Layers (Skip Connections) to restore spatial resolution.
Output: A 1x1 convolution with a Sigmoid activation, mapping results to a pixel-wise probability map.

### Results & Inference
The model produces a continuous probability map. To generate the final biological mask used in research, we apply a binarization threshold:


