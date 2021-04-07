#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time: 2020/11/20 17:03
# Author: Zheng Shaoxiang
# @Email: zhengsx95@163.com
# Description:
"""
一维相关问题：
一维装箱问题 (one-dimensional bin packing problem)
一维cutting问题 (one-dimensional cutting and stock problem)
一维背包问题 (one-dimensional knapsack problem)
一维批调度问题 (one-dimensional single batch machine problem)
"""
from utility.basicmodel import BasicModel
from utility.basicmodel import GRB, tuplelist, quicksum


class BinPacking(BasicModel):
    """
    一维装箱问题：定义一系列具有一定长度的 items和无限多相同长度的bins
    目标: 将所有items放入bins使得使用bins的数量最小

    输入:
    items = {item_id: Item(id width)} -> dict
    width -> int (float)
    调用:
    from utility.items import get_1d_item
    Item = get_1d_item()
    # items = {item_id: Item(id=item_id, width=width)}
    items = {1: Item(id=1, width=1), 2: Item(id=2, width=2)}
    width = 10
    bp = BinPacking(items, width)
    bp.solve()

    """
    def __init__(self, items, width, *args, **kwargs) -> None:
        name = self.__class__.__name__
        super().__init__(name, *args, **kwargs)
        self.items = items  # {item_id: Item(id=item_id, width=width)}
        self.width = width
        self.init()  # 初始化相关索引和参数

    def init(self):
        self.init_indices()
        self.init_params()

    def init_indices(self):
        # Indices and set
        self.n = len(self.items)  # number of jobs
        self.J = tuple(j for j in self.items)  # job indices
        self.B = tuple(range(1, self.get_batches_num() + 1))  # batch indices

    def init_params(self):
        # parameters
        self.x_index = tuplelist([(j, k) for j in self.J for k in self.B])
        self.y_index = self.B
        self.w = {item.id: item.width for item in self.items.values()}

    def get_batches_num(self):
        return len(self.items)

    def _construct_variables(self):
        self.x = self.addVars(self.x_index, vtype=GRB.BINARY, name="x")
        self.y = self.addVars(self.y_index, vtype=GRB.BINARY, name="y")

    def construct_variables(self):
        self._construct_variables()

    def _construct_constraints(self):
        self.addConstrs((self.x.sum(j, '*') == 1 for j in self.J), name="exact_one")
        self.addConstrs((quicksum(self.x[j, k] * self.w[j] for j in self.J) <=
                         self.width * self.y[k] for k in self.B), name="knapsack_constraints")

    def construct_constraints(self):
        self._construct_constraints()

    def construct_objective(self):
        self.model.setObjective(self.y.sum(), GRB.MINIMIZE)

    def build_model(self):
        # variables
        self.construct_variables()

        # constraints
        self.construct_constraints()

        # objective
        self.construct_objective()


class Cutting(BinPacking):
    """
    一维下料问题：定义一系列具有一定长度的 items, 每个item有一定的需求d_j 和无限多相同长度的stocks
    目标: 从数量最小的stocks中切割出满足需求的items

    输入:
    items = {item_id: Item(id width demand)} -> dict
    width -> int (float)
    调用:
    from utility.items import get_1d_item
    Item = get_1d_item("demand")
    # items = {item_id: Item(id=item_id, width=width, demand=demand)}
    items = {1: Item(id=1, width=1, demand=3), 2: Item(id=2, width=2, demand=3)}
    width = 10
    bp = Cutting(items, width)
    bp.solve()

    """
    def __init__(self, items, width, *args, **kwargs) -> None:
        super().__init__(items, width, *args, **kwargs)
        self.model.modelName = self.__class__.__name__
        self.d = {item.id: item.demand for item in self.items.values()}  # 需求

    def get_batches_num(self):
        return sum(item.width * item.demand for item in self.items.values())

    def construct_variables(self):
        self.x = self.addVars(self.x_index, vtype=GRB.INTEGER, name="x")
        self.y = self.addVars(self.y_index, vtype=GRB.BINARY, name="y")

    def construct_constraints(self):
        self.addConstrs((self.x.sum(j, '*') >= self.d[j] for j in self.J), name="demand_fulfill")
        self.addConstrs((quicksum(self.x[j, k] * self.w[j] for j in self.J) <=
                         self.width * self.y[k] for k in self.B), name="knapsack_constraints")


class Knapsack(BinPacking):
    """
    一维背包问题：定义一系列具有width属性和profit属性的 items和一个容量固定width的背包
    目标: 使得放入背包的items的profits之和最大

    输入:
    items = {item_id: Item(id width profit)} -> dict
    width -> int (float)
    调用:
    from utility.items import get_1d_item
    Item = get_1d_item("profit")
    # items = {item_id: Item(id=item_id, width=width, profit=profit)}
    items = {1: Item(id=1, width=1, profit=3), 2: Item(id=2, width=2, profit=3)}
    width = 10
    bp = Knapsack(items, width)
    bp.solve()

    """
    def __init__(self, items, width, *args, **kwargs) -> None:
        super().__init__(items, width, *args, **kwargs)
        self.model.modelName = self.__class__.__name__

    def init_indices(self):
        self.n = len(self.items)
        self.J = tuple(j for j in self.items)

    def init_params(self):
        self.x_index = self.J
        self.w = {item.id: item.width for item in self.items.values()}  # profit
        self.p = {item.id: item.profit for item in self.items.values()}  # profit

    def construct_variables(self):
        self.x = self.addVars(self.x_index, vtype=GRB.BINARY, name="x")

    def construct_constraints(self):
        self.addConstr(self.x.prod(self.w), GRB.LESS_EQUAL, self.width, name="knapsack_constraint")

    def construct_objective(self):
        self.setObjective(self.x.prod(self.p), GRB.MAXIMIZE)


class SingleBatch(BinPacking):
    """
    一维批调度：定义一系列具有width属性和processing_time属性的 items和无限个固定容量的批
    目标: 使得所有items放入批中且最大完工时间C_{max}最小
    约束:  (1)每个工件能且仅能放入一个批次中;
          (2)每个批中工件不能超过其容量;
          (3)每个批的加工时间是该批中加工时间最长的工件;
    输入:
    items = {item_id: Item(id width processing_time)} -> dict
    width -> int (float)
    调用:
    from utility.items import get_1d_item
    Item = get_1d_item("processing_time")
    # items = {item_id: Item(id=item_id, width=width, processing_time=processing_time)}
    items = {1: Item(id=1, width=1, processing_time=3), 2: Item(id=2, width=2, processing_time=3)}
    width = 10
    bp = SingleBatch(items, width)
    bp.solve()

    """

    def __init__(self, items, width, *args, **kwargs) -> None:
        super().__init__(items, width, *args, **kwargs)
        self.model.modelName = self.__class__.__name__
        self.p = {item.id: item.processing_time for item in self.items.values()}

    def construct_variables(self):
        self.x = self.addVars(self.x_index, vtype=GRB.BINARY, name="x")
        self.P = self.addVars(self.B, vtype=GRB.CONTINUOUS, name="P")

    def construct_constraints(self):
        self.addConstrs((self.x.sum(j, '*') == 1 for j in self.J), name="exact_one")
        self.addConstrs((quicksum(self.x[j, k] * self.w[j] for j in self.J) <=
                         self.width for k in self.B), name="knapsack_constraints")
        self.addConstrs((self.P[k] >= self.p[j] * self.x[j, k] for j in self.J for k in self.B),
                        name="batch_processing_time")

    def construct_objective(self):
        self.setObjective(self.P.sum(), GRB.MINIMIZE)


if __name__ == '__main__':
    from utility.items import get_1d_item

    Item = get_1d_item("processing_time")
    items = {1: Item(id=1, width=7, processing_time=2), 2: Item(id=2, width=4, processing_time=3)}
    width = 10
    bp = SingleBatch(items, width)
    bp.solve()
    print(bp.obj_val)
