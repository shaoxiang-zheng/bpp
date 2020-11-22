#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time : 2020/11/22 15:05
# Author: Zheng Shaoxiang
# @Email : zhengsx95@163.com
# Description:
from one_dim.lower_bound import BinPackingBound as oneBpb
import math


class BinPackingBound(oneBpb):
    """
    二维装箱问题的下界:
    (1) 连续型下界 continuous bound O(n)
    工件尺寸之和与箱子尺寸比值 worst-case performance ratio 1/2
    (2) 不可旋转 划分型下界 O(n^2)
    Lw
    Jw = {j in J: wj > W/2}
    给定1 <= p <= H/2, J1 = {j in Jw: hj > H - p}, J2 = {j in Jw: H-p >= hj < H/2}
    Lw(p) = max(L_alpha^w(p), max(L_beta^w(p)))
    L = max(Lw(p), L(h)p)
    reference: [1]Silvano Martello, Danilel Vigo. Exact Solution of the Two-Dimensional Finite Bin Packing Problem.
                  Management Science.
    """
    def __init__(self, items, width, height):
        super().__init__(items, width)
        self.height = height
        self.h = {item.id: item.height for item in self.items.values()}

    def continuous_bound(self):
        return math.ceil(sum(w * h for w, h in zip(self.w.values(), self.h.values())) / self.width * self.height)

    def get_height(self, job_set):
        return sum(self.h[j] for j in job_set)

    def get_width(self, job_set):
        return sum(self.w[j] for j in job_set)

    def partition_bound_without_rotation(self):
        j_w = {j for j, w in self.w.items() if w > self.width / 2}
        j_h = {j for j, h in self.h.items() if h > self.height / 2}

        def get_lw(p):
            j_w_1 = {j for j in j_w if self.h[j] > self.height - p}
            j_w_2 = {j for j in j_w if self.height - p >= self.h[j] > self.height / 2}
            j_w_3 = {j for j in j_w if self.height / 2 >= self.h[j] >= p}
            l_alpha = len(j_w_1) + len(j_w_2) + max(0, math.ceil(
                (self.get_height(j_w_3) - (len(j_w_2) * self.height - self.get_height(j_w_2))) / self.height))
            l_beta = len(j_w_1) + len(j_w_2) + max(0, math.ceil((len(j_w_3) - sum(
                math.floor((self.height - self.h[j]) / p) for j in j_w_2)) / math.floor(self.height / p)))
            return max(l_alpha, l_beta)

        def get_lh(p):
            j_h_1 = {j for j in j_h if self.w[j] > self.width - p}
            j_h_2 = {j for j in j_h if self.width - p >= self.w[j] > self.width / 2}
            j_h_3 = {j for j in j_h if self.width / 2 >= self.w[j] >= p}
            l_alpha = len(j_h_1) + len(j_h_2) + max(0, math.ceil(
                (self.get_width(j_h_3) - (len(j_h_2) * self.width - self.get_width(j_h_2))) / self.width))
            l_beta = len(j_h_1) + len(j_h_2) + max(0, math.ceil((len(j_h_3) - sum(
                math.floor((self.width - self.w[j]) / p) for j in j_h_2)) / math.floor(self.width / p)))
            return max(l_alpha, l_beta)

        lw = max(get_lw(p) for p in range(1, int(self.height / 2) + 1))
        lh = max(get_lh(p) for p in range(1, int(self.width / 2) + 1))

        return max(lw, lh)


if __name__ == '__main__':
    pass
