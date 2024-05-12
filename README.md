## ELEC4840: Artificial Intelligence for Medical Imaging Analysis
### Final Project
# Skin Lesion Classification using Domain Generalisation and Contrastive Learning
### Tara Relan (20716928), Yudao Zhang (20761911)

### Table of Contents
- [Introduction](#introduction)
- [Project Requirements](#requirements)
- [Part 1: Domain Generalisation via Ensemble Learning](#part-1-domain-generalisation-via-ensemble-learning)
  - [Data](#part-1-data)
  - [Running the Code](#part-1-getting-started)
- [Part 2: Contrastive Learning](#part-2-contrastive-learning)
  - [Data](#part-2-data)
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
