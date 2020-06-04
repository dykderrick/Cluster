# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/6/3
# @Author  : Yingke Ding
# @File    : data_object.py
# @Software: PyCharm


class DataObject:
    def __init__(self, point):
        self._point = point
        self._class_number = None

    def set_class_number(self, class_number):
        self._class_number = class_number

    def get_class_number(self):
        return self._class_number

    def get_point(self):
        return self._point

    def euclidean_distance(self, centroid_point):
        if len(self._point) != len(centroid_point):
            raise Exception("ERROR DIMENSION")

        squares = [(p - q) ** 2 for p, q in zip(self._point, centroid_point)]
        return sum(squares) ** 0.5
