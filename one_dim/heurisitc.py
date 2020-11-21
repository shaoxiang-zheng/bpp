#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time : 2020/11/21 16:57
# Author: Zheng Shaoxiang
# @Email : zhengsx95@163.com
# Description:
"""
一维装箱问题的启发式算法：
一维装箱问题：
best fit first (BF)
first fit first (FF)
"""
from uti.params import INF


class FirstFit:
    """
    一维装箱问题的first fit
    思想: 将所有items按照宽度width 降序排序,每次选择序列中第一个item,将其放到第一个可以放得下
    的箱子中,如果当前所有箱子都放置不下,则打开新的箱子,直到所有items都放完为止

    输入:
    items = {item_id: Item(id width)} -> dict
    width -> int (float)
    调用:
    from uti.items import get_1d_item
    Item = get_1d_item()
    # items = {item_id: Item(id=item_id, width=width)}
    items = {1: Item(id=1, width=1), 2: Item(id=2, width=2)}
    width = 10
    bp = FirstFit(items, width)
    res = bp.solve()
    # res = {bin_id: [Item(id, width),...]}
    """
    def __init__(self, items, width, *args, **kwargs):
        self.items = items
        self.width = width
        self.preprocessing_for_items()

    def preprocessing_for_items(self):
        self.items = dict(sorted(self.items.items(), key=lambda x: x[1].width, reverse=True))

    def choose_bin_id(self, item, capacity, schedule):
        for bin_id, c in capacity.items():  # 按顺序遍历箱子
            if c + item.width <= self.width:  # 如果当前箱子可以放得下当前item
                schedule[bin_id].append(item)
                capacity[bin_id] += item.width
                return bin_id
        return -1

    def solve(self):
        schedule = {}
        k = 0  # 目前开启的箱子数
        capacity = {}  # 记录过程中每个箱子的容量
        for item in self.items.values():
            bin_id = self.choose_bin_id(item, capacity, schedule)  # 选择箱子并更新capacity和schedule

            if bin_id == -1:  # 如果遍历完所有箱子都放不下，则打开新的箱子
                k += 1
                schedule[k] = [item]
                capacity[k] = item.width
        return schedule


class BestFit(FirstFit):
    """
    一维装箱问题的best fit
    思想: 将所有items按照宽度width 降序排序,每次选择序列中第一个item,将其放到可以放得下且剩余空间最小
    的箱子中,如果当前所有箱子都放置不下,则打开新的箱子,直到所有items都放完为止

    输入:
    items = {item_id: Item(id width)} -> dict
    width -> int (float)
    调用:
    from uti.items import get_1d_item
    Item = get_1d_item()
    # items = {item_id: Item(id=item_id, width=width)}
    items = {1: Item(id=1, width=1), 2: Item(id=2, width=2)}
    width = 10
    bp = BestFit(items, width)
    res = bp.solve()
    # res = {bin_id: [Item(id, width),...]}
    """

    def __init__(self, items, width, *args, **kwargs):
        super().__init__(items, width, *args, **kwargs)

    def choose_bin_id(self, item, capacity, schedule):
        k, residual = -1, INF
        for bin_id, c in capacity.items():  # 按顺序遍历箱子
            if c + item.width <= self.width:  # 如果当前箱子可以放得下当前item
                if self.width - c - item.width < residual:
                    k, residual = bin_id, self.width - c - item.width
        if k != -1:
            schedule[k].append(item)
            capacity[k] += item.width

        return k


if __name__ == '__main__':
    from uti.items import get_1d_item

    Item = get_1d_item()
    # items = {item_id: Item(id=item_id, width=width)}
    items = {1: Item(id=1, width=6), 2: Item(id=2, width=5)}
    width = 10
    bp = BestFit(items, width)
    res = bp.solve()
    print(res)
