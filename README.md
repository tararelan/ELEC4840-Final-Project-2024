## ELEC4840: Artificial Intelligence for Medical Imaging Analysis
### Final Project
# Skin Lesion Classification using Domain Generalisation and Contrastive Learning
### Tara Relan (20716928), Yudan Zhao (20761911)

### Table of Contents
- [Introduction](#introduction)
- [Project Requirements](#requirements)
- [Part 1: Domain Generalisation via Ensemble Learning](#part-1-domain-generalisation-via-ensemble-learning)
  - [Data](#part-1-data)
  - [Running the Code](#part-1-getting-started)
- [Part 2: Simple Contrastive learning(SimCLR)](#part-2-Simple-Contrastive-Learning)
  - [Step](#part-2-step)
  - [Running the Code](#part-2-getting-started)

### Introduction
Skin cancer is a significant health concern, with pigmented skin lesions like moles and nevi having the potential to be either benign or malignant. While most skin lesions are harmless such as nevi and benign keratosis, some can develop into various types of skin cancer, including squamous cell carcinoma (SCC), basal cell carcinoma (BCC), and the most dangerous form, melanoma. For example, in HK in 2021, there were 1200 new cases of skin cancer, with 1094 cases of non-melanoma and 106 cases of melanoma, and there seems to be a decreasing trend in the percentage of incidence rates per 100k people. Detecting and diagnosing these malignant lesions accurately is crucial for timely and effective treatment. However, the traditional diagnostic process is time-consuming and relies on human expertise, which can be limited by the intricate variability in the appearance of skin lesions. To address this challenge, there is a growing interest in developing automated systems using deep learning techniques that can detect lesion transformation and classify skin cancer types. These systems have the potential to assist medical professionals in expediting diagnoses and improving patient outcomes. However, deploying such models in real-world scenarios presents its own set of challenges, including generalization across different imaging domains and the scarcity of labeled data.

### Requirements
* If using Google Colab, use a GPU with high RAM.
```sh
from google.colab import drive
drive.mount('/content/drive')
```
* Regardless of if you're using Google Colab / running the code locally, change the directory to where your files are.
* To install the requirements for both parts, run the following in terminal:
```sh
pip install -r requirements.txt
```

### Part 1: Domain Generalisation via Ensemble Learning
Domain generalisation is a sub-field of transfer learning that aims to bridge the gap between two different domains in the absence of any knowledge about the target domain. Deep learning models for medical image analysis easily suffer from distribution shifts caused by dataset artifacts bias, camera variations, differences in the imaging station, etc., leading to unreliable diagnoses in real-world clinical settings. Domain generalization (DG) methods, which aim to train models on multiple domains to perform well on unseen domains, offer a promising direction to solve the problem. In this project, we built an ensemble model from base deep learning models that have been trained on different datasets from ISIC to introduce diversity into the model.

### Part 1: Data
2016
- The 2016 ISIC dataset ([accessed here][1]) has 900 training images and 379 testing images, from which we split it into 862 training images, and 322 validation images, and 379 testing images.
- There are two labels:
  - Melanoma (malignant)
  - Benign
- We trained the ResNet50 model on this dataset for 20 epochs, with an SGD optimiser that has a learning rate of 0.01, and a binary CE loss.
- We also applied the ensemble model on the entire 2016 dataset.

2017
- The 2017 ISIC dataset ([accessed here][2]) has 2000 training images, 150 validation images, and 600 testing images. There are two labels:
  – Melanoma (malignant)
  - Seborrheic keratosis (benign)
- We trained the AlexNet model on this dataset for 20 epochs, with an SGD optimiser that has a learning rate of 0.01, and a binary CE loss.

2018
- The 2018 ISIC dataset ([accessed here][3]) has 10015 training images, 193 validation images, and 1512 testing images.
- There are seven labels:
  - Melanoma (malignant)
  - Nevus (benign)
  - Basal cell carcinoma (malignant)
  - Actinic keratosis (malignant)
  - Benign keratosis (benign)
  - Dermatofibroma (benign)
  - Vascular lesion (which could be both)
- We trained the VGG19 model on this dataset for 20 epochs, with an SGD optimiser that has a learning rate of 0.01, and a binary CE loss.

2019
- The 2019 ISIC dataset ([accessed here][4]) has 20148 training images, from which we split it into 15198 training images, 5066 validation images, and 5067 testing images.
- There are nine labels:
  - Melanoma (malignant)
  - Nevus (benign)
  - Basal cell carcinoma (malignant)
  - Actinic keratosis (malignant)
  - Benign keratosis (benign)
  - Dermatofibroma (benign)
  - Vascular lesion (which could be both)
  - Squamous cell carcinoma (malignant)
  - None of the others (which could be both)
- We trained the ResNet50 model on this dataset for 20 epochs, with an SGD optimiser that has a learning rate of 0.01, and a binary CE loss.

### Part 1: Getting Started
- Download the relevant datasets from ISIC, and then sort the data into the following format:
```
root: train/val/test
    class_a
        a1.png
        a2.png
        ...
    class_b
        b1.png
        b2.png
        ...
```
- The code for this is already given in the Jupyter notebook.

[1]: https://challenge.isic-archive.com/landing/2016/
[2]: https://challenge.isic-archive.com/landing/2017/
[3]: https://challenge.isic-archive.com/landing/2018/
[4]: https://challenge.isic-archive.com/landing/2019/

### Part 2: Simple Contrastive learning(SimCLR)
Simple Contrastive Learning(SimCLR) is an unsupervised learning technique used in image analysis to learn representations of images. It maximizes the similarity between differently augmented views of the same image and minimizes the similarity between different images. SimCLR uses a contrastive loss function to encourage the model to learn similar representations for augmented views of the same image while pushing the representations of different images further apart. This technique is trained on large amounts of unlabeled data, allowing it to learn useful representations without the need for labeled data.
In this project, we aimed to improve the performance of our deep learning model in skin lesion classification. To introduce diversity into the model, we built an ensemble model from base deep learning models that have been trained on different datasets ranging from ISIC to other sources. By combining the predictions of multiple models, we were able to achieve better accuracy and reduce the risk of overfitting.

### Part 2: Steps
- Generate two different views of the same image using transforms such as cropping, flipping, and normalization. These versions are used for comparative learning.
- Utilize an encoder network to extract features from the image. Adapt the pre-trained ResNet-50 model by replacing the last fully connected layer with an identity mapping. Freeze the weight of the ResNet-50 architecture to focus on the most informative part of the image.
- Project the heads to map the features to a space where the comparison loss can be applied more effectively. Use a small neural network with linear layer and ReLU activation to refine features for comparison tasks.
- Apply the NT-Xent Loss to minimize the distance between representations of the same image and maximize the distance between different image representations. Calculate the loss using cosine similarity and the softmax function.
- Update the weights of the encoder and projection head using the Adam optimizer with a learning rate of 2e-5 to minimize the contrast loss. Each training iteration processes a pair of enhanced images, calculates the embedding, loss, and performs back propagation.

### Part 2: Getting Started
Please execute all the code in the file named "contrastive_learning.ipynb" in sequence. The code was written based on my assignment 2 question 1. I have added two paragraphs of code at the end of the file, but the original code remains unchanged.(But please run all of them in sequence, error will happened if skip the front code) The "contrastive learning" section contains code for rebuilding the model, while the last paragraph of code under "Test" is for testing the model. These codes will execute the results enhanced by contrastive learning. Please note that the results may differ slightly from those given in the report because the results may vary each time the code is executed.
