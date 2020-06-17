# -*- coding: utf-8 -*-
# @Time    : 2020/6/17 2:28 下午
# @Author  : Yingke Ding
# @FileName: visualization.py
# @Software: PyCharm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from DBSCAN.DBSCAN_algorithm import DBSCAN
from k_means.k_means_algorithm import KMeans

CSV_FILE_PATH = "../dataset/Iris.csv"


def main(algorithm):
    if algorithm == 'KMeans':
        algorithm_object = KMeans(CSV_FILE_PATH, k=3)
        algorithm_object.algorithm()
        results = algorithm_object.get_results()
    else:
        algorithm_object = DBSCAN(CSV_FILE_PATH, radius=1.375, min_pts=5)
        algorithm_object.algorithm()
        results = algorithm_object.get_results()

    features = pd.DataFrame([result.get_point() for result in results])
    classes = pd.DataFrame([result.get_class_number() for result in results])

    feature_class_dicts = [
        {class_number: [feature for index, feature in enumerate(features.values.tolist())
                        if classes.values[index][0] == class_number]} for class_number in set(np.unique(classes.values))
    ]

    fig = plt.figure()
    ax = Axes3D(fig)
    colors = ['red', 'blue', 'green', 'black', 'white', 'yellow', 'lime', 'cyan', 'orange', 'gray']

    for index, class_features in enumerate(feature_class_dicts):
        all_features = list(class_features.values())[0]

        xs = [feature[0] for feature in all_features]
        ys = [feature[1] for feature in all_features]
        zs = [feature[2] for feature in all_features]

        if index >= len(colors):
            color = 'black'
        else:
            color = colors[index]
        ax.scatter(xs, ys, zs, c=color)
        # plt.scatter(xs, ys, c=color)

    fig.savefig('../figures/' + algorithm + '_classified.svg', format='svg')


if __name__ == '__main__':
    main("KMeans")
    main("DBSCAN")
