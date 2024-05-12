import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score

def train_model(model, loader_train, loader_val, loader_test, max_epoch=20, use_cuda=True):
    if use_cuda:
        model = model.cuda()

    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    loss_list, val_loss_list = [], []
    auc_train_list, acc_train_list = [], []
    auc_val_list, acc_val_list = [], []

    for epoch in range(max_epoch):
        print(" -- Epoch {}/{}".format(epoch + 1, max_epoch))

        model.train()
        running_loss = 0.0
        train_lbl = []
        train_pred = []
        for data in tqdm(loader_train):
            optimizer.zero_grad()
            images, labels = data
            labels = labels.float()
            if use_cuda:
                images = images.cuda()
                labels = labels.float().cuda()
            outputs = model(images)[:, 0]
            outputs = torch.sigmoid(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            y_scores = list(outputs.detach().cpu().numpy())
            y_true = list(labels.detach().cpu().numpy())
            train_lbl += y_true
            train_pred += y_scores

        loss = running_loss / len(loader_train)
        loss_list.append(loss)
        train_lbl, train_pred = np.array(train_lbl), np.array(train_pred)
        train_pred_lbl = np.around(train_pred)
        train_auc = roc_auc_score(train_lbl, train_pred)
        train_acc = accuracy_score(train_lbl, train_pred_lbl)
        auc_train_list.append(train_auc)
        acc_train_list.append(train_acc)

        model.eval()
        test_lbl, test_pred = [], []
        val_loss = 0.0
        for data in loader_val:
            test_images, test_labels = data
            test_labels = test_labels.float()
            if use_cuda:
                test_images = test_images.cuda()
                test_labels = test_labels.float().cuda()
            with torch.no_grad():
                test_outputs = model(test_images)[:, 0]
            test_outputs = torch.sigmoid(test_outputs)
            y_scores = list(test_outputs.detach().cpu().numpy())
            y_true = list(test_labels.detach().cpu().numpy())
            test_lbl += y_true
            test_pred += y_scores

            v_loss = criterion(test_outputs, test_labels)
            val_loss += v_loss.item()

        test_lbl, test_pred = np.array(test_lbl), np.array(test_pred)
        test_pred_lbl = np.around(test_pred)
        test_auc = roc_auc_score(test_lbl, test_pred)
        test_acc = accuracy_score(test_lbl, test_pred_lbl)
        auc_val_list.append(test_auc)
        acc_val_list.append(test_acc)
        val_loss = val_loss / len(loader_val)
        val_loss_list.append(val_loss)
        print(loss, val_loss, train_auc, test_auc)

    torch.save(model.state_dict(), 'ensemble_model.pth')

    model.eval()
    test_lbl, test_pred = [], []
    for data in loader_test:
        test_images, test_labels = data
        test_labels = test_labels.float()
        if use_cuda:
            test_images = test_images.cuda()
            test_labels = test_labels.float().cuda()
        test_outputs = model(test_images)[:, 0]
        test_outputs = torch.sigmoid(test_outputs)
        y_scores = list(test_outputs.detach().cpu().numpy())
        y_true = list(test_labels.detach().cpu().numpy())
        test_lbl += y_true
        test_pred += y_scores

    test_lbl, test_pred = np.array(test_lbl), np.array(test_pred)
    test_pred_lbl = np.around(test_pred)
    test_auc = roc_auc_score(test_lbl, test_pred)
    test_acc = accuracy_score(test_lbl, test_pred_lbl)
    print(test_auc, test_acc)

    return {
        'loss_list': loss_list,
        'val_loss_list': val_loss_list,
        'auc_train_list': auc_train_list,
        'acc_train_list': acc_train_list,
        'auc_val_list': auc_val_list,
        'acc_val_list': acc_val_list,
        'test_auc': test_auc,
        'test_acc': test_acc
    }