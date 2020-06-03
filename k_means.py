# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/6/3
# @Author  : Yingke Ding
# @File    : k_means.py
# @Software: PyCharm
import csv
import random

from data_object import DataObject


class KMeans:
    def __init__(self, k, csv_file_path):
        self._k = k
        self._data_objects = []
        self._results = None
        self._scan_dataset(csv_file_path)

    def _scan_dataset(self, csv_file_path):
        with open(csv_file_path, "r") as csv_file:
            reader = csv.reader(csv_file)

            for row in reader:
                self._data_objects.append(DataObject([float(i) for i in row[:-1]]))

    def get_data_objects(self):
        return self._data_objects

    def _calculate_centroid(self, points, dimension):
        return [sum(point[i] for point in points) / len(points) for i in range(dimension)]  # TODO: BUG division by zero

    def algorithm(self):
        has_changed = True
        shadow_data_objects = self._data_objects.copy()

        random.shuffle(shadow_data_objects)
        for index, data_object in enumerate(shadow_data_objects):
            data_object.set_class_number(index % self._k)

        while has_changed:
            centroids = [self._calculate_centroid(points=[point.get_point() for point in shadow_data_objects if point.get_class_number() == i], dimension=4) for i in range(self._k)]

            change_class_number = False
            for data_object in shadow_data_objects:
                distances_to_centroids = [data_object.euclidean_distance(centroid) for centroid in centroids]
                closest_centroid_index = distances_to_centroids.index(min(distances_to_centroids))

                if data_object.get_class_number() != closest_centroid_index:
                    data_object.set_class_number(closest_centroid_index)
                    change_class_number = True

            has_changed = change_class_number

        self._results = shadow_data_objects
        return shadow_data_objects

    def print_cluster_result(self):
        for index, data_object in enumerate(self._data_objects):
            print(self._results[self._results.index(data_object)].get_point())
            print(self._results[self._results.index(data_object)].get_class_number())
            print("\n")
