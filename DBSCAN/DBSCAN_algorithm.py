# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/6/4
# @Author  : Yingke Ding
# @File    : DBSCAN_algorithm.py
# @Software: PyCharm
import csv
import random

from DBSCAN.dbscan_data_object import DBSCANDataObject


class DBSCAN:
    """
    Object for DBSCAN algorithm,
    """
    def __init__(self, csv_file_path, radius, min_pts):
        self._radius = radius
        self._min_pts = min_pts
        self._data_objects = []
        self._results = None
        self._scan_dataset(csv_file_path)
        self._original_data_objects = self._data_objects.copy()

    def _scan_dataset(self, csv_file_path):
        with open(csv_file_path, "r") as csv_file:
            reader = csv.reader(csv_file)

            for row in reader:
                self._data_objects.append(DBSCANDataObject([float(i) for i in row[:-1]]))

    def _calc_radius_neighborhood(self, a_data_object):
        """
        Calculate a specific point's neighbor, with radius being specified at __init__.
        :param a_data_object:
        :return:
        """
        _neighbors = []

        for data_object in self._data_objects:
            if a_data_object.directly_density_reachable(data_object.get_point(), self._radius):
                _neighbors.append(data_object)

        return _neighbors

    def _get_unvisited_objects(self):
        """
        Find all unvisited data objects. DBSCAN will end if return empty.
        :return:
        """
        return [data_object for data_object in self._data_objects if data_object.get_visited() is False]

    def algorithm(self):
        """
        Core part for the algorithm.
        :return:
        """
        unvisited_data_objects = self._data_objects.copy()
        random.shuffle(unvisited_data_objects)

        current_class_number = -1
        while len(unvisited_data_objects) != 0:
            current_data_object = unvisited_data_objects[0]

            current_data_object.mark_visited()

            radius_neighborhood = self._calc_radius_neighborhood(current_data_object)
            if len(radius_neighborhood) >= self._min_pts:
                current_class_number += 1
                current_data_object.set_class_number(current_class_number)

                for neighbor in radius_neighborhood:
                    if neighbor.get_visited() is False:
                        neighbor.mark_visited()

                        neighbor_radius_neighborhood = self._calc_radius_neighborhood(neighbor)
                        if len(neighbor_radius_neighborhood) >= self._min_pts:
                            radius_neighborhood += neighbor_radius_neighborhood
                            radius_neighborhood = list(set(radius_neighborhood))

                    if neighbor.get_class_number() is None:
                        neighbor.set_class_number(current_class_number)
            else:
                # noise
                current_data_object.mark_noise()
                current_data_object.set_class_number(-1)  # -1 means noise

            unvisited_data_objects = self._get_unvisited_objects()
            random.shuffle(unvisited_data_objects)

        self._results = self._data_objects

    def print_cluster_result(self):
        for index, data_object in enumerate(self._original_data_objects):
            print(self._results[self._results.index(data_object)].get_point())
            print(self._results[self._results.index(data_object)].get_class_number())
            print("\n")
