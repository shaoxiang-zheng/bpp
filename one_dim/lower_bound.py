#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time : 2020/11/22 15:04
# Author: Zheng Shaoxiang
# @Email : zhengsx95@163.com
# Description:
import math


class BinPackingBound:
    """
    一维装箱问题的下界:
    (1) 连续型下界 continuous bound O(n)
    工件尺寸之和与箱子尺寸比值 worst-case performance ratio 1/2

    (2) 列生成
    """
    def __init__(self, items, width):
        self.items = items
        self.width = width
        self.w = {item.id: item.width for item in self.items.values()}

    def continuous_bound(self):
        return math.ceil(sum(self.w.values()) / self.width)


if __name__ == '__main__':
    pass
