import matplotlib.pyplot as plt

def plot_curves(loss_list, val_loss_list, acc_train_list, acc_val_list):
    plt.figure(figsize=(8, 6))
    plt.plot(loss_list, label='Training Loss')
    plt.plot(val_loss_list, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(acc_train_list, label='Training Accuracy')
    plt.plot(acc_val_list, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy Curves')
    plt.legend()
    plt.show()