# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/6/3
# @Author  : Yingke Ding
# @File    : run.py
# @Software: PyCharm
from k_means import KMeans


def main():
    algorithm_object = KMeans(k=3, csv_file_path="./dataset/Iris.csv")

    algorithm_object.algorithm()
    algorithm_object.print_cluster_result()


if __name__ == '__main__':
    main()
