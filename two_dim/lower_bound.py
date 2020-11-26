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
    L_1^w(p) = max(L_alpha^w(p), max(L_beta^w(p)))
    L_1^w = max_{1 <= p <= H/2} {L_1^w(p)}
    L_1 = max(L_1^w, L_1^w)
    reference: [1]Silvano Martello, Danilel Vigo. Exact Solution of the Two-Dimensional Finite Bin Packing Problem.
                  Management Science.
    (3) 不可旋转 同时考虑宽度和高度的划分 O(n^3)

    reference: [1]Silvano Martello, Danilel Vigo. Exact Solution of the Two-Dimensional Finite Bin Packing Problem.
                  Management Science.
    (4)
    (5) 可旋转
    reference:[2] Mauro Dell' Amico, Silvano Martello, Daniele Vigo. A lower bound for the non-oriented
                  two-dimensional bin packing problem
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

    def get_lw_and_lh(self):
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
        return lw, lh

    def partition_bound_without_rotation1(self):
        lw, lh = self.get_lw_and_lh()
        return max(lw, lh)

    def partition_bound_without_rotation2(self):
        J = {j for j in self.items}
        lw, lh = self.get_lw_and_lh()

        def get_lw(q):
            k1 = {j for j in J if self.w[j] > self.width - q}
            k2 = {j for j in J if self.width - q >= self.w[j] > self.width / 2}
            k3 = {j for j in J if self.width / 2 >= self.w[j] >= q}
            return lw + max(0, math.ceil((sum(self.h[j] * self.w[j] for j in k2 | k3) - (
                    self.height * lw - sum(self.h[j] for j in k1)) * self.width) / (self.width * self.height)))

        def get_lh(q):
            k1 = {j for j in J if self.h[j] > self.height - q}
            k2 = {j for j in J if self.height - q >= self.h[j] > self.height / 2}
            k3 = {j for j in J if self.height / 2 >= self.h[j] >= q}
            return lw + max(0, math.ceil((sum(self.w[j] * self.h[j] for j in k2 | k3) - (
                    self.width * lh - sum(self.w[j] for j in k1)) * self.height) / (self.width * self.height)))

        lw2 = max(get_lw(q) for q in range(1, int(self.width / 2) + 1))
        lh2 = max(get_lh(q) for q in range(1, int(self.height / 2) + 1))
        return max(lw2, lh2)

    def partition_bound_without_rotation3(self):
        J = {j for j in self.items}

        def m(j, p, q):
            cur = math.floor(self.height / p) * math.floor((self.width - self.w[j]) / q) + math.floor(self.width / q) \
                  * math.floor((self.height - self.h[j]) / p) - math.floor((self.height - self.h[j]) / p) * math.floor(
                (self.width - self.w[j]) / q)
            return cur

        def get_lower(p, q):
            I1 = {j for j in J if self.h[j] > self.height - p and self.w[j] > self.width - q}
            I2 = {j for j in J if j not in I1 and self.h[j] > self.height / 2 and self.w[j] > self.width / 2}
            I3 = {j for j in J if self.height / 2 >= self.h[j] >= p and self.width / 2 >= self.w[j] >= q}
            low = len(I1 | I2) + max(0, math.ceil((len(I3) - sum(m(j, p, q) for j in I2)) / (
                    math.floor(self.height / p) * math.floor(self.width / q))))
            return low

        return max(get_lower(p, q) for p in range(1, math.floor(self.height / 2 + 1))
                   for q in range(1, math.floor(self.width / 2 + 1)))

    def get_lower_bound_without_rotation(self, J):
        return max(self.partition_bound_without_rotation2(J), self.partition_bound_without_rotation3(J))

    def cut_squares(self):
        squares = {}
        i = 1
        for j in self.items:
            s = {}
            w, h = max(self.w[j], self.h[j]), min(self.w[j], self.h[j])

            while h > 1:  # 不产生1X1的item因此在后面用不到
                k = math.floor(w / h)
                for num in range(k):
                    s[i + num] = h
                i += k
                w -= k * h
                w, h = h, w
            squares.update(s)
        return squares

    def get_lower_bound_with_rotation(self):
        squares = self.cut_squares()  # {id: l}
        if self.height > self.width:
            self.width, self.height = self.height, self.width

        def get_lower(q):
            S1 = {j for j in squares if squares[j] > self.width - q}
            S2 = {j for j in squares if self.width - q >= squares[j] > self.width / 2}
            S3 = {j for j in squares if self.width / 2 >= squares[j] > self.height / 2}
            S4 = {j for j in squares if self.height / 2 >= squares[j] >= q}

            if self.width == self.height:
                low_q = len(S1 | S2) + max(0, math.ceil(sum(squares[j] ** 2 for j in S2 | S4) /
                                                        self.width ** 2 - len(S2)))
                return low_q

            S_3 = set()
            # 按边长降序
            _S2, _S3 = sorted(list(S2), key=lambda x: squares[x], reverse=True), sorted(
                list(S3), key=lambda x: squares[x], reverse=True)
            q = 0
            for i in _S2:
                for _j in range(q, len(_S3)):
                    j = _S3[_j]
                    if squares[i] + squares[j] <= self.width:
                        S_3.add(j)
                        q = _j + 1
                        break
                else:  # 如果一个都不存在
                    break

            low = len(S2) + max(math.ceil(sum(squares[j] for j in S3 if j not in S_3) / self.width), math.ceil(
                (len(S3) - len(S_3)) / math.floor(self.width / math.floor(1 + self.height / 2))))
            S23 = {j for j in S2 | S3 if squares[j] > self.height - q}
            low_q = len(S1) + low + max(0, math.ceil((sum(squares[j] ** 2 for j in S2 | S3 | S4) - (
                    self.width * self.height * low - sum(squares[j] * (self.height - squares[j]) for j in S23))) / (
                    self.width * self.height)))
            return low_q
        return max(get_lower(q) for q in range(math.floor(self.height / 2) + 1))


if __name__ == '__main__':
    from uti.items import get_2d_item
    import random
    # random.seed(0)
    Item = get_2d_item()
    n = 10
    items = {item_id: Item(id=item_id, width=random.randint(1, 10),
                           height=random.randint(1, 10)) for item_id in range(1, n + 1)}

    width = 10
    height = 10
    bl = BinPackingBound(items, width, height=height)
    r = bl.get_lower_bound_with_rotation()
    print(f"{r=}")
