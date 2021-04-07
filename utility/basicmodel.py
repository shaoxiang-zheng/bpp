#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time: 2020/11/20 16:01
# Author: Zheng Shaoxiang
# @Email: zhengsx95@163.com
# Description: 定义了一个针对一般混合整数规划模型的求解框架
from gurobipy import *
import psutil
from utility.params import EPS

from abc import ABC, abstractmethod


class BasicModel(ABC):
    """
    定义基础模型求解模板
    """
    @abstractmethod  # 子类中必须重写的方法
    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.model = Model(name)

        # 模型求解前参数
        self.output_flag = kwargs.get("output_flag", False)   # 是否输出求解过程，默认False
        self.time_limit = kwargs.get("time_limit", 3600)  # 默认设置3600s
        self.threads = kwargs.get("threads", psutil.cpu_count())  # 设置使用进程数

        # 模型加速有关的参数
        self.init_sol_flag = kwargs.get("init_sol_flag", True)  # 为模型注入初始解
        self.bound = kwargs.get("bound", None)  # 最小值问题则是下界设置，最大值问题是上界

        # 模型求解后记录的运行时间，目标值以及求解状态
        self.runtime = 0
        self.obj_val = None  # objective value
        self.status = None  # the optimized status of the model

        # 求解后处理的相关参数
        self.print_variable_flag = kwargs.get("print_variable_flag", False)

    def set_params(self):
        # self.model.Params.Threads = self.threads
        self.model.Params.OutputFlag = self.output_flag
        self.model.Params.TimeLimit = self.time_limit

    def addVar(self, vtype, lb=0.0, ub=GRB.INFINITY, obj=0.0, name="", column=None):
        return self.model.addVar(lb, ub, obj, vtype, name, column)

    def addVars(self, *indexes, lb=0.0, ub=GRB.INFINITY, obj=0.0, vtype=None, name=""):
        return self.model.addVars(*indexes, lb=lb, ub=ub, obj=obj, vtype=vtype, name=name)

    def addConstr(self, lhs, sense, rhs, name):
        return self.model.addConstr(lhs, sense, rhs, name)

    def addConstrs(self, constrs, name=""):
        return self.model.addConstrs(constrs, name)

    def setObjective(self, expression, sense=None):
        return self.model.setObjective(expression, sense)

    def set_bound(self, bound):
        self.bound = bound

    def get_bound(self):
        pass

    def get_init_sol(self):
        pass

    def print_variables(self):
        pass

    def add_init_sol(self):
        pass

    def do_something(self):
        pass

    @abstractmethod  # 建立问题模型
    def build_model(self):
        pass

    # 求解模型
    def solve(self):
        bound = self.bound

        def callback(model, where):  # simple callback
            # 定义callback函数，当新发现的可行整数解和当前设定的界相同，即返回最优解
            if where == GRB.Callback.MIPSOL:  # 找到新的目标值
                if abs(model.cbGet(GRB.Callback.MIPSOL_OBJBST) - bound) <= EPS:
                    model.terminate()

        # 调用构建模型的方法
        if self.model.NumConstrs > 0:  # 表示已经构建了模型
            self.build_model()  # build the model
            self.model.update()

        if self.init_sol_flag:  # if the flag is True, generate initial solutions and use them.
            self.add_init_sol()
            self.model.update()

        self.set_params()

        if self.bound is None:  # when the lower bound is generated, use it for callback
            self.model.optimize()
        else:
            self.model.optimize(callback)

        self.status = self.model.Status
        self.runtime = self.model.Runtime

        try:
            self.obj_val = self.model.objVal
        except AttributeError:  # 表示在有限时间内没有求解出结果
            pass

        if self.print_variable_flag:  # 允许打印变量
            self.print_variables()

        self.do_something()  # 处理后处理
        
        return self.model  # 返回求解后的模型


class SetCovering:
    def __init__(self):
        self.rmp = None
        self.solve()

    def init_rmp(self):
        pass

    def pricing(self, dual):
        """
        :param dual
        :return
        """
        return 0, []

    def add_column(self, coe):
        pass

    def solve(self):
        self.init_rmp()  # 初始化限制主问题

        while True:
            self.rmp.optimize()  # 求解限制主问题

            dual = self.rmp.getAttr(GRB.Attr.Pi)  # 获取各约束对偶变量

            reduced_cost, coe = self.pricing(dual)  # 获取reduced cost

            if reduced_cost + EPS >= 0:
                break
            self.add_column(coe)  # 为模型添加列


if __name__ == '__main__':
    pass
