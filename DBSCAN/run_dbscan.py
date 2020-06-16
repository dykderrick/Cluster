# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/6/4
# @Author  : Yingke Ding
# @File    : run_dbscan.py
# @Software: PyCharm
from DBSCAN.DBSCAN_algorithm import DBSCAN


def main():
    algorithm_object = DBSCAN(csv_file_path="../dataset/Iris.csv", radius=0.8, min_pts=5)

    algorithm_object.algorithm()
    algorithm_object.print_cluster_result()


if __name__ == '__main__':
    main()
