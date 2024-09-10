# **Solving a Dataset using *Paper Replication***

## Problem Statement -> Solve the challenging dataset Food101 using a Convolutional Neural Network Model written from scratch using PyTorch.‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎

## Dataset -> Food101‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎

[Link to the dataset.](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
![enter image description here](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/static/img/food-101.jpg)

- The Food-101 is dataset of 101 food categories with 101,000 images. For each class, 250 manually reviewed test images are provided as well as 750 training images. On purpose, the training images were not cleaned, and thus still contain some amount of noise.  All images were rescaled to have a maximum side length of 512 pixels. The images include variations in lighting, perspective, and background, making it a challenging dataset. ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎

## Choosing & Replicating the paper of the Model

### My Model Architecture is inspired by **VGG-16**

- VGG-16  is renowned for its simplicity and effectiveness, as well as its ability to achieve strong performance on various  computer vision  tasks, including  image classification  and  object recognition. The model’s architecture features a stack of  convolutional layers  followed by  max-pooling  layers, with progressively increasing depth.

### Why this Model

- As this model has achieved  92.7% top-5 test accuracy on the  ImageNet  dataset which contains 14 million images belonging to 1000 classes, It would be highly suitable for Food-101, maybe an overkill too.
![enter image description here](https://neurohive.io/wp-content/uploads/2018/11/vgg16-1-e1542731207177.png)

### Paper Replication to Make the Model

#### Source Research Paper

[*VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION*](https://arxiv.org/pdf/1409.1556v6)

- On page 3, Architecture of VGG-16 is mentioned under **ConvNet Configuration D with 16 weight layers**.

- This project involved a comprehensive analysis of the research paper "Very Deep Convolutional Networks for Large-Scale Image Recognition," coupled with a thorough examination of relevant images from diverse sources about the architecture. Based on these insights, I successfully implemented a VGG-16 model using PyTorch, replicating the exact architecture and parameter count as described in the original research.

## Coding Implementation

### **Data Acquisition and Preprocessing:**

- The Food101 dataset was sourced from the `torchvision.datasets`  library.
- To increase the visual understanding of the dataset, a random selection of images and their corresponding labels were plotted, facilitating visual inspection and analysis of the data.
- The raw data was transformed into dataloaders using the PyTorch DataLoader class, enabling efficient batching and shuffling of the data during training. A **batch size of 32** was chosen to balance computational efficiency and model performance.

### **Model Architecture:**

- A deep convolutional neural network (CNN) architecture was designed inspired by the VGG-16, incorporating **13 convolutional layers** with varying input and output sizes as per the research paper.
- **ReLU activation** functions were used between each layer to introduce non-linearity.
- **Max pooling** layers were strategically placed after each convolutional block.
- **3 fully connected layers** were included to process the extracted features and produce class probabilities.
- The **Softmax activation** function was applied to the final layer to normalize the **output probabilities**.

### **Model Summary and Testing:**

- The `torchinfo` library was utilized to obtain a detailed summary of the model's architecture. As the implementation was correct, The model came out with **134,674,341 parameters.**
- To **verify the model's functionality**, a single image was passed through the network, and the predicted output was inspected.
- **The model was found to be functioning as expected, producing predictions of label from input of image.**

### **Loss Function and Optimizer:**

- The **cross-entropy loss** function was selected as it is well-suited for multi-class classification tasks.
- The **RMSprop** optimizer was chosen for its effectiveness in training image classification models.

### **Training and Evaluation:**

- **Training Function**  to implement the standard training loop, with the following steps:
  - **Forward pass/Forward Propagation**
  - **Loss calculation**
  - **Optimizer reset/ Optimizer Zero Gradient**
  - **Backpropagation**
  - **Optimizer step**
- **Testing Function**  to evaluate the model's performance on the test set, calculating both loss and accuracy.
- **Combined Training and Testing function** was created to  iterate over multiple epochs and record performance metrics and **store them in a dictionary per epoch for plotting.**

## Due to the substantial computational resources required to train a model with *134,674,341 parameters* on the *101,000-images in Food101 dataset*, as indicated by the original research paper's training time of *5 weeks*, this project has reached its conclusion. Due to resource problems, Training the model is not possible. However, the results obtained would closely align with the research paper showing an *accuracy of 92.17%* on an even larger dataset
