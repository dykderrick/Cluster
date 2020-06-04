# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/6/4
# @Author  : Yingke Ding
# @File    : dbscan_data_object.py
# @Software: PyCharm
from k_means.data_object import DataObject


class DBSCANDataObject(DataObject):
    def __init__(self, point):
        super().__init__(point)
        self._visited = False
        self._noise = False

    def get_visited(self):
        return self._visited

    def mark_visited(self):
        self._visited = True

    def mark_noise(self):
        self._noise = True

    def directly_density_reachable(self, another_point, radius):
        return self.euclidean_distance(another_point) <= radius
