# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/6/3
# @Author  : Yingke Ding
# @File    : run_algorithms.py
# @Software: PyCharm
import csv

from DBSCAN.DBSCAN_algorithm import DBSCAN
from k_means.k_means_algorithm import KMeans

CSV_FILE_PATH = "../dataset/Iris.csv"


def _save_results(results, algorithm_name):
    features = [data_object.get_point() for data_object in results]
    numbers = [data_object.get_class_number() for data_object in results]
    rows = [
        (index, feature[0], feature[1], feature[2], feature[3], numbers[index]) for index, feature in
        enumerate(features)
    ]

    with open("../results/" + algorithm_name + "_results.csv", "w") as f:
        headers = ['INDEX', 'D1', 'D2', 'D3', 'D4', 'CLUSTER NUMBER']
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def _print_results(results):
    """
    Compare the results with original.
    :return:
    """
    for index, data_object in enumerate(results):
        print('INDEX       ' + str(index))
        print('FEATURES    ' + str(data_object.get_point()))
        print('CLUSTER NO. ' + str(data_object.get_class_number()))
        print('\n')


def run_k_means():
    algorithm_object = KMeans(CSV_FILE_PATH, k=3)
    algorithm_object.algorithm()

    results = algorithm_object.get_results()
    _print_results(results)
    _save_results(results, "KMeans")


def run_dbscan():
    algorithm_object = DBSCAN(csv_file_path=CSV_FILE_PATH, radius=1.375, min_pts=5)
    algorithm_object.algorithm()

    results = algorithm_object.get_results()
    _print_results(results)
    _save_results(results, "DBSCAN")


if __name__ == '__main__':
    run_k_means()
    run_dbscan()
