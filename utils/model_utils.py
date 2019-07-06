from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools


def plot_confusion_matrix(y_true, y_pred, title='Classification Report',
                          labels_=None, target_names=None, normalize=False):
    """
    Utility function for plotting confusion matrix for classification evaluation
    :param y_true: true labels array
    :param y_pred: predicted labels array
    :param title: Title of the confusion matrix plot
    :param labels_: list of unique labels (e.g. in classification with two classes it could be [0, 1])
    :param target_names: names list for unique labels (e.g. in two classes classification it can be ['male', 'female'])
    :param normalize: boolean, whether to print number in confusion matrix as percentage or not
    :return:
    """

    # print classification report
    print(classification_report(y_true, y_pred,
                                labels=labels_, target_names=target_names))

    # plot the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, f"{cm[i, j]:.4f}",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, f"{cm[i, j]}",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel(
        f'Predicted label\naccuracy={accuracy:.4f}; misclass={misclass:.4f}')
    plt.show()
