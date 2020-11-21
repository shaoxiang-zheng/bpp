#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time: 2020/11/20 17:18
# Author: Zheng Shaoxiang
# @Email: zhengsx95@163.com
# Description:
from collections import namedtuple

Corner = namedtuple("Corner", "id x y width height")
Gap = namedtuple("Gap", "x y width")
TreeKey = namedtuple("TreeKey", "w h id")


class Segment:
    __slots__ = ['x', 'y', 'width']

    def __init__(self, x, y, width):
        self.x, self.y = x, y
        self.width = width

    def __repr__(self):
        return f"(x={self.x},y={self.y},width={self.width})"

    def __hash__(self):
        return hash((self.x, self.y, self.width))

    def __eq__(self, other):
        return self.y == other.y

    def __le__(self, other):
        return self.y <= other.y

    def __gt__(self, other):
        return not self.__le__(other)

    def __ge__(self, other):
        return self.y >= other.y

    def __lt__(self, other):
        return not self.__ge__(other)


class Results(list):
    def __init__(self):
        super().__init__()


def get_item(*attr):
    Item = namedtuple("Item", [at for at in attr])
    return Item


def get_1d_item(*attr):
    attrs = ['id', 'width'] + list(attr)
    return get_item(*attrs)


def get_2d_item(*attr):
    return get_1d_item('height', *attr)


if __name__ == '__main__':
    pass
