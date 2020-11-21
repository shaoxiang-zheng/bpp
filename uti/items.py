#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time: 2020/11/20 17:18
# Author: Zheng Shaoxiang
# @Email: zhengsx95@163.com
# Description:
from collections import namedtuple


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
