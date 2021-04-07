#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time: 2020/11/20 20:36
# Author: Zheng Shaoxiang
# @Email: zhengsx95@163.com
# Description:
"""
二维相关问题：
二维正交问题 (two-dimensional Orthogonal packing problem)
二维装箱问题 (two-dimensional bin packing problem)
二维条带问题 (two-dimensional strip packing problem)
二维背包问题 (two-dimensional knapsack problem)
二维批调度问题 (two-dimensional single batch machine scheduling problem)
"""
from one_dim.model import BinPacking, Cutting, Knapsack, SingleBatch
from utility.basicmodel import GRB, tuplelist, quicksum


class OrthogonalPacking(Knapsack):
    """
    二维正交问题：定义一系列具有width属性和height属性的 items和一个长width宽height固定的箱子
    目标: 判断当前所有items是否可以放入该箱子
    复杂度: NP-completeness
    约束:
    (1)任意两个矩形不相互重叠;
    (2)任意矩形不超过箱子的边;
    (3)任意矩形的边平行于箱子的边(Orthogonal)

    输入:
    items = {item_id: Item(id width height)} -> dict
    width -> int (float)
    height -> int (float)
    调用:
    from utility.items import get_2d_item
    Item = get_2d_item()
    # items = {item_id: Item(id=item_id, width=width, height=height)}
    items = {1: Item(id=1, width=1, height=3), 2: Item(id=2, width=2, height=3)}
    width, height = 10, 10
    bp = OrthogonalPacking(items, width, height)
    bp.solve()

    """
    def __init__(self, items, width, height, *args, **kwargs):
        self.height = height
        super().__init__(items, width, *args, **kwargs)

    def init_params(self):
        self.bid_index = tuplelist([(i, j) for i in self.J for j in self.J if i != j])
        self.w = {item.id: item.width for item in self.items.values()}
        self.h = {item.id: item.height for item in self.items.values()}

    def construct_variables(self):
        self.i = self.addVars(self.J, vtype=GRB.BINARY, name="i")  # item i is in the bin or not
        self.o = self.addVars(self.J, vtype=GRB.BINARY, name="o")
        self.l = self.addVars(self.bid_index, vtype=GRB.BINARY, name="l")
        self.b = self.addVars(self.bid_index, vtype=GRB.BINARY, name="b")
        self.x = self.addVars(self.J, vtype=GRB.CONTINUOUS, name="x")
        self.y = self.addVars(self.J, vtype=GRB.CONTINUOUS, name="y")

    def construct_constraints(self):
        self.addConstrs((self.x[j] + self.w[j] * self.o[j] + self.h[j] * (1 - self.o[j]) <= self.width for j in self.J),
                        name="x_not_exceed")
        self.addConstrs((self.y[j] + self.h[j] * self.o[j] + self.w[j] * (1 - self.o[j]) <= self.height for j in self.J),
                        name="y_not_exceed")
        self.addConstrs((self.x[j] + self.w[j] * self.o[j] + self.h[j] * (1 - self.o[j]) <= self.x[j] +
                         self.width * (1 - self.l[i, j]) for i, j in self.bid_index), name="x_not_overlap")
        self.addConstrs((self.y[j] + self.h[j] * self.o[j] + self.w[j] * (1 - self.o[j]) <= self.y[j] +
                         self.height * (1 - self.b[i, j]) for i, j in self.bid_index), name="y_not_overlap")
        self.addConstrs((self.l[i, j] + self.l[j, i] + self.b[i, j] + self.b[j, i] >= self.i[i] + self.i[j] - 1
                        for i, j in self.bid_index if i < j), name="relative_position")

    def construct_objective(self):
        self.setObjective(self.i.sum(), GRB.MAXIMIZE)


class OrthogonalKnapsack(OrthogonalPacking):
    """
    二维背包问题：定义一系列具有width属性,height属性和profit属性的 items和一个长width宽height固定的箱子
    目标: 使得合法放入箱子中的profits之和最大
    复杂度: NP-hard
    约束:
    (1)任意两个矩形不相互重叠;
    (2)任意矩形不超过箱子的边;
    (3)任意矩形的边平行于箱子的边(Orthogonal)

    输入:
    items = {item_id: Item(id width height, profit)} -> dict
    width -> int (float)
    height -> int (float)
    调用:
    from utility.items import get_2d_item
    Item = get_2d_item("profit")
    # items = {item_id: Item(id=item_id, width=width, height=height, profit=profit)}
    items = {1: Item(id=1, width=1, height=3, profit=2), 2: Item(id=2, width=2, height=3, profit=2)}
    width, height = 10, 10
    bp = KnapsackPacking(items, width, height)
    bp.solve()

    """
    def __init__(self, items, width, height, *args, **kwargs):
        super().__init__(items, width, height, *args, **kwargs)
        self.p = {item.id: item.profit for item in self.items.values()}

    def construct_objective(self):
        self.setObjective(self.i.prod(self.p), GRB.MAXIMIZE)


class OrthogonalBinPacking(BinPacking):
    """
    二维装箱问题：定义一系列具有width属性,height的 items和无限个长width宽height固定的箱子
    目标: 使得合使用的箱子数最少
    复杂度: NP-hard
    约束:
    (1)任意两个矩形不相互重叠;
    (2)任意矩形不超过箱子的边;
    (3)任意矩形的边平行于箱子的边(Orthogonal)

    输入:
    items = {item_id: Item(id width height)} -> dict
    width -> int (float)
    height -> int (float)
    调用:
    from utility.items import get_2d_item
    Item = get_2d_item()
    # items = {item_id: Item(id=item_id, width=width, height=height)}
    items = {1: Item(id=1, width=1, height=3), 2: Item(id=2, width=2, height=3)}
    width, height = 10, 10
    bp = OrthogonalBinPacking(items, width, height)
    bp.solve()

    """
    def __init__(self, items, width, height, *args, **kwargs):
        super().__init__(items, width, *args, **kwargs)
        self.height = height
        self.bid_index = tuplelist([(i, j) for i in self.J for j in self.J if i != j])
        self.h = {item.id: item.height for item in self.items.values()}

    def _construct_variables(self):
        self.X = self.addVars(self.x_index, vtype=GRB.BINARY, name="X")
        self.o = self.addVars(self.J, vtype=GRB.BINARY, name="o")
        self.x = self.addVars(self.J, vtype=GRB.CONTINUOUS, name="x")
        self.y = self.addVars(self.J, vtype=GRB.CONTINUOUS, name="y")
        self.l = self.addVars(self.bid_index, vtype=GRB.BINARY, name="l")
        self.b = self.addVars(self.bid_index, vtype=GRB.BINARY, name="b")

    def construct_variables(self):
        self._construct_variables()
        self.Y = self.addVars(self.B, vtype=GRB.BINARY, name="Y")

    def _construct_constraints(self):
        self.addConstrs((self.x[j] + self.w[j] * self.o[j] + self.h[j] * (1 - self.o[j]) <= self.width for j in self.J),
                        name="x_not_exceed")
        self.addConstrs(
            (self.y[j] + self.h[j] * self.o[j] + self.w[j] * (1 - self.o[j]) <= self.height for j in self.J),
            name="y_not_exceed")
        self.addConstrs((self.X.sum(j, '*') == 1 for j in self.J), name="exact_one")

        self.addConstrs((self.x[j] + self.w[j] * self.o[j] + self.h[j] * (1 - self.o[j]) <= self.x[j] +
                         self.width * (1 - self.l[i, j]) for i, j in self.bid_index), name="x_not_overlap")
        self.addConstrs((self.y[j] + self.h[j] * self.o[j] + self.w[j] * (1 - self.o[j]) <= self.y[j] +
                         self.height * (1 - self.b[i, j]) for i, j in self.bid_index), name="y_not_overlap")
        self.addConstrs((self.l[i, j] + self.l[j, i] + self.b[i, j] + self.b[j, i] >= self.X[i, k] + self.X[j, k] - 1
                         for k in self.B for i, j in self.bid_index if i < j), name="relative_position")

    def construct_constraints(self):
        self._construct_constraints()
        self.addConstrs((quicksum(self.w[j] * self.h[j] * self.X[j, k] for j in self.J) <=
                         self.Y[k] * self.width * self.height for k in self.B), name="bin_formed")

    def construct_objective(self):
        self.setObjective(self.Y.sum(), GRB.MINIMIZE)


class OrthogonalStripPacking:
    pass


class OrthogonalSingleBatch(OrthogonalBinPacking):
    """
    二维批调度问题：定义一系列具有width,height,processing_time属性的 items和无限个长width宽height固定的箱子
    目标: 最小化makespan
    复杂度: NP-hard
    约束:
    (1)任意两个矩形不相互重叠;
    (2)任意矩形不超过箱子的边;
    (3)任意矩形的边平行于箱子的边(Orthogonal)
    (4)一个箱子的加工时间不小于箱子里任意工件的加工时间

    输入:
    items = {item_id: Item(id width height, processing_time)} -> dict
    width -> int (float)
    height -> int (float)
    调用:
    from utility.items import get_2d_item
    Item = get_2d_item("processing_time")
    # items = {item_id: Item(id=item_id, width=width, height=height, processing_time=processing_time)}
    items = {1: Item(id=1, width=1, height=3, processing_time=3), 2: Item(id=2, width=2, height=3, processing_time=3)}
    width, height = 10, 10
    bp = OrthogonalSingleBatch(items, width, height)
    bp.solve()

    """

    def __init__(self, items, width, height, *args, **kwargs):
        super().__init__(items, width, height *args, **kwargs)
        self.p = {item.id: item.processing_time for item in self.items.values()}

    def construct_variables(self):
        self._construct_variables()
        self.P = self.addVars(self.B, vtype=GRB.CONTINUOUS, name="P")

    def construct_constraints(self):
        self._construct_constraints()
        self.addConstrs((self.P[k] >= self.X[j, k] * self.p[j] for j, k in self.x_index), name="bin_processing_time")

    def construct_objective(self):
        self.setObjective(self.P.sum(), GRB.MINIMIZE)


if __name__ == '__main__':
    from utility.items import get_2d_item

    Item = get_2d_item()
    # items = {item_id: Item(id=item_id, width=width, height=height)}
    items = {1: Item(id=1, width=1, height=3), 2: Item(id=2, width=2, height=3)}
    width, height = 10, 10
    bp = OrthogonalBinPacking(items, width, height)
    bp.solve()
    print(bp.obj_val)
