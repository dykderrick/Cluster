# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/6/4
# @Author  : Yingke Ding
# @File    : run_dbscan.py
# @Software: PyCharm
from DBSCAN.DBSCAN_algorithm import DBSCAN


def main():
    DBSCAN(csv_file_path="../dataset/Iris.csv", radius=0.39, min_pts=4).algorithm()


if __name__ == '__main__':
    main()
