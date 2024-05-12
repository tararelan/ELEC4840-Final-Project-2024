from google.colab import drive
import os
import tempfile
import shutil
import pandas as pd
import numpy as np

from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch

def setup_directory(directory):

    if not os.path.exists("./train"):
        os.makedirs("./train")
        os.makedirs("./train/benign")
        os.makedirs("./train/malignant")

    if not os.path.exists("./val"):
        os.makedirs("./val")
        os.makedirs("./val/benign")
        os.makedirs("./val/malignant")

    if not os.path.exists("./test"):
        os.makedirs("./test")
        os.makedirs("./test/benign")
        os.makedirs("./test/malignant")
    
    print('Train folder exists:', os.path.exists('./train'))
    print('Val folder exists:', os.path.exists('./val'))
    print('Test folder exists:', os.path.exists('./test'))

def setup_data(directory, df_train, df_val, df_test):
    for i in range(len(df_train)):
        name, label_mel, label_nv, label_bcc, label_akiec, label_bkl, label_df, label_vasc = df_train["image"][i], df_train["MEL"][i], df_train["NV"][i], df_train["BCC"][i], df_train["AKIEC"][i], df_train["BKL"][i], df_train["df_train"][i], df_train["VASC"][i]

        label = "benign" if label_mel == 0 and label_bcc == 0 and label_akiec == 0 else "malignant"
        shutil.copy(directory+name+".jpg", "./train/"+label+"/"+name+".jpg")
    
    for i in range(len(df_val)):
        name, label_mel, label_nv, label_bcc, label_akiec, label_bkl, label_df, label_vasc = df_val["image"][i], df_val["MEL"][i], df_val["NV"][i], df_val["BCC"][i], df_val["AKIEC"][i], df_val["BKL"][i], df_val["df_val"][i], df_val["VASC"][i]

        label = "benign" if label_mel == 0 and label_bcc == 0 and label_akiec == 0 else "malignant"
        shutil.copy(directory+name+".jpg", "./val/"+label+"/"+name+".jpg")
    
    for i in range(len(df_test)):
        name, label_mel, label_nv, label_bcc, label_akiec, label_bkl, label_df, label_vasc = df_test["image"][i], df_test["MEL"][i], df_test["NV"][i], df_test["BCC"][i], df_test["AKIEC"][i], df_test["BKL"][i], df_test["df_test"][i], df_test["VASC"][i]

        label = "benign" if label_mel == 0 and label_bcc == 0 and label_akiec == 0 else "malignant"
        shutil.copy(directory+name+".jpg", "./test/"+label+"/"+name+".jpg")

def count_files(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        count += len(files)
    return count

def label_statistics(train_dataset):
    cls_count = np.zeros(2).astype(np.int64)

    for i, label in train_dataset:
        cls_count[label] += 1
    return cls_count

def label_weights_for_balance(train_dataset):
    cls_count = label_statistics(train_dataset)
    labels_weight_list = []
    for i, label in train_dataset:
        weight = 1 / cls_count[label]
        labels_weight_list.append(weight)
    return labels_weight_list

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=1, saturation=0.1, hue=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def get_train_test_set(root_train, root_val, root_test):
    train_dataset = ImageFolder(root_train, transform=train_transform)

    loader_train = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=16,
        shuffle=True
        )

    val_dataset = ImageFolder(root_val, transform=test_transform)
    loader_val = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=16,
        shuffle=False
        )

    test_dataset = ImageFolder(root_test, transform=test_transform)
    loader_test = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=16,
        shuffle=False
        )

    return loader_train, loader_val, loader_test