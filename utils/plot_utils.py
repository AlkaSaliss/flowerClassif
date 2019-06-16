from bokeh.layouts import gridplot
from bokeh.io import output_notebook, show
from bokeh.plotting import show, figure
from bokeh.models import ColumnDataSource
from bokeh.transform import factor_cmap
from bokeh.palettes import Spectral6
import os
import glob
import pathlib
import random
import numpy as np
import skimage
import matplotlib.pyplot as plt


def plot_class_distribution(root_folder="./", title=None):
    """Function to plot the distribution of number of samples in each class

    Keyword Arguments:
        root_folder {str} -- path to the directory containing subdirectories, one for each class (default: {"./"})
    """

    # get list of the paths
    list_paths = glob.glob(os.path.join(root_folder, '*[a-zA-Z]'))
    # get list of classes
    list_classes = [pathlib.Path(p).name for p in list_paths]
    # get counts and frequencies for each class
    counts = [len(glob.glob(os.path.join(root_folder, class_, '*.jpg')))
              for class_ in list_classes]
    freq = [np.round(item*100/sum(counts), 1) for item in counts]

    source = ColumnDataSource(data=dict(
        counts=counts,
        classes=list_classes,
        freq=freq
    ))
    TOOLTIPS = [
        ("class", "@classes"),
        ("count", "@counts"),
        ("freq", "@freq")
    ]
    if title is None:
        title = "Class distribution "
    title += " - Total images : {}".format(sum(counts))
    p = figure(x_range=list_classes, tooltips=TOOLTIPS,
               title=title, plot_width=500, plot_height=300)
    p.vbar(x='classes', top='counts', width=0.4, source=source,
           line_color='white', fill_color=factor_cmap('classes', palette=Spectral6, factors=list_classes))

    show(p)


def display_sample_images(im_path):
    """function to display some sample images in a notebook

    Arguments:
        im_path {str} -- path to the directory containing subdirectories, one for each class
    """

    list_paths = glob.glob(os.path.join(im_path, "*[a-zA-Z]"))
    list_classes = [pathlib.Path(p).name for p in list_paths]

    dict_images = {
        class_: random.sample(glob.glob(os.path.join(p, '*.jpg')), 5)
        for p, class_ in zip(list_paths, list_classes)
    }

    dict_images = {
        class_: [skimage.io.imread(p) for p in path]
        for class_, path in dict_images.items()
    }

    for cl in list_classes:
        fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(10, 12))
        ax = ax.ravel()
        fig.text(0.06, 0.5, str(cl), ha='center', va='center', color='blue', fontsize=15)
        for idx, im in enumerate(dict_images[cl]):
            ax[idx].imshow(skimage.transform.resize(im, (200, 200)))
            ax[idx].axis("off")

    plt.show()


def plot_image_dimensions(im_path):
    """function that plots the distribution of images dimension, i.e. height and width.

    Arguments:
        im_path {str} -- path to the folder containing subdirectories, each subdirectory contains images for a given class

    """

    def _get_dims(ps):
        h, w = [], []
        for p in ps:
            im = skimage.io.imread(p)
            h.append(im.shape[0])
            w.append(im.shape[1])
        return {'h': h, 'w': w}

    list_paths = glob.glob(os.path.join(im_path, "*[a-zA-Z]"))
    list_classes = [pathlib.Path(p).name for p in list_paths]

    dict_images = {
        class_: glob.glob(os.path.join(p, '*.jpg'))
        for p, class_ in zip(list_paths, list_classes)
    }

    dict_dims = {
        class_: _get_dims(dict_images[class_])
        for class_ in dict_images
    }

    fig_row1 = []
    fig_row2 = []

    for idx, cl in enumerate(list_classes):
        h, w = dict_dims[cl]['h'], dict_dims[cl]['w']

        hist, edges = np.histogram(h, bins=50)
        s = figure(width=250, plot_height=250,
                   title="Mean Height: {:.1f}".format(np.mean(h)))
        s.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
               fill_color="#6600ff", line_color="white", alpha=0.5)
        if idx == 0:
            s.yaxis.axis_label = 'Height'
        fig_row1.append(s)

        hist, edges = np.histogram(w, bins=50)
        s = figure(width=250, plot_height=250,
                   title="Mean Width: {:.1f}".format(np.mean(w)))
        s.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
               fill_color="#6600ff", line_color="white", alpha=0.5)
        s.xaxis.axis_label = cl
        if idx == 0:
            s.yaxis.axis_label = 'Width'
        fig_row2.append(s)

    p = gridplot([fig_row1, fig_row2])
    show(p)


def plot_image_dimensions_old(im_path):

    def _get_dims(ps):
        h, w = [], []
        for p in ps:
            im = skimage.io.imread(p)
            h.append(im.shape[0])
            w.append(im.shape[1])
        return {'h': h, 'w': w}

    list_paths = glob.glob(os.path.join(im_path, "*[a-zA-Z]"))
    list_classes = [pathlib.Path(p).name for p in list_paths]

    dict_images = {
        class_: glob.glob(os.path.join(p, '*.jpg'))
        for p, class_ in zip(list_paths, list_classes)
    }

    dict_dims = {
        class_: _get_dims(dict_images[class_])
        for class_ in dict_images
    }

    fig, ax = plt.subplots(nrows=2, ncols=len(list_classes), figsize=(20, 6))
    title = "Images height and width distribution per class"
    fig.suptitle(title, fontsize=20)

    for idx, cl in enumerate(list_classes):
        h, w = dict_dims[cl]['h'], dict_dims[cl]['w']
        ax[0, idx].hist(h, color="  # 6600ff")
        ax[0, idx].set_title("Mean Height: {:.1f}".format(np.mean(h)))
        if idx == 0:
            ax[0, idx].set_ylabel('Height')
            ax[1, idx].set_ylabel('Width')
        ax[1, idx].hist(w, color="#6600ff")
        ax[1, idx].set_title("Mean Width: {:.1f}".format(np.mean(w)))
        ax[1, idx].set_xlabel(str(cl))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
