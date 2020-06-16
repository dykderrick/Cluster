# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/6/3
# @Author  : Yingke Ding
# @File    : k_means_algorithm.py
# @Software: PyCharm
import csv
import random

from k_means.data_object import DataObject


def _calculate_centroid(points, dimension):
    return [sum(point[i] for point in points) / len(points) for i in range(dimension)]


class KMeans:
    def __init__(self, k, csv_file_path="../dataset/Iris.csv"):
        self._k = k
        self._data_objects = []
        self._results = None
        self._scan_dataset(csv_file_path)
        self._original_data_objects = self._data_objects.copy()
        self._shuffle_data_objects()

    def _scan_dataset(self, csv_file_path):
        with open(csv_file_path, "r") as csv_file:
            reader = csv.reader(csv_file)

            for row in reader:
                self._data_objects.append(DataObject([float(i) for i in row[:-1]]))

    def _shuffle_data_objects(self):
        random.shuffle(self._data_objects)

    def algorithm(self):
        has_changed = True

        for index, data_object in enumerate(self._data_objects):
            data_object.set_class_number(index % self._k)

        while has_changed:
            centroids = []
            unexpected_bug = False
            for i in range(self._k):
                points = [point.get_point() for point in self._data_objects if point.get_class_number() == i]

                # 这个bug我真不知道错哪, 大概平均每10次运行会出现一次
                # 正常有bug都是每次运行都会有, 但是这个就非常神奇, 玄学
                # self._data_objects里面class_number不知道为什么被篡改
                # 导致丢失某个类别的所有标签, 后面算centroids会找不到某个类别的点
                # 换句话说, 本来k=3的算法被强行改成k=2了
                # 这简直就是降维打击
                # 如果您知道bug出自哪里, 欢迎issue或email: dykderrick@bupt.edu.cn
                # 这里我不得不重新来一遍算法
                if len(points) == 0:
                    unexpected_bug = True
                    break

                centroid = _calculate_centroid(points, 4)
                centroids.append(centroid)

            if unexpected_bug:
                break
            """
            centroids = [self._calculate_centroid(
                points=[point.get_point() for point in self._data_objects if point.get_class_number() == i],
                dimension=4) for i in range(self._k)]
            """

            change_class_number = False
            for data_object in self._data_objects:
                distances_to_centroids = [data_object.euclidean_distance(centroid) for centroid in centroids]
                closest_centroid_index = distances_to_centroids.index(min(distances_to_centroids))

                if data_object.get_class_number() != closest_centroid_index:
                    data_object.set_class_number(closest_centroid_index)
                    change_class_number = True

            has_changed = change_class_number

        if has_changed:  # 重来
            self.__init__(self._k)
            self.algorithm()
        else:
            self._results = self._data_objects

    def print_cluster_result(self):
        for index, data_object in enumerate(self._original_data_objects):
            print(self._results[self._results.index(data_object)].get_point())
            print(self._results[self._results.index(data_object)].get_class_number())
            print("\n")
