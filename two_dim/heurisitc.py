#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time : 2020/11/21 17:19
# Author: Zheng Shaoxiang
# @Email : zhengsx95@163.com
# Description:
"""
二维正交问题的启发式算法：
二维strip packing问题：
Bottom-left (BL)
Best-fit (BF)
Efficient-best-fit (EBF)

二维bin packing问题：
Bottom-left (BL)
Best-fit (BF)
Efficient-best-fit (EBF)
"""
from uti.params import INF
from uti.items import Corner, Gap, Results, Segment, TreeKey
from uti.tree import BalancedBinaryTree
import numpy as np
from uti.heap import Heap
from uti.node import Node
from uti.linked_list import DoubleLinkList
import copy


class BottomLeftStrip:
    """
    思想：按照输入的顺序从右上角开始，轮流向下和向左，直到item不能移动为止

    输入:
    items = {item_id: Item(id width height)} -> dict
    width -> int (float)
    height -> int (float)

    输出: :return:
          bl.results = [Corner(id= x= y= width= height=),...] list[<class Corner>]
          bl.residual = [Item(id width height)]

    调用:
    from uti.items import get_2d_item
    Item = get_2d_item()
    # items = {item_id: Item(id=item_id, width=width, height=height)}
    items = {1: Item(id=1, width=1, height=3), 2: Item(id=2, width=2, height=3)}
    width = 10
    bl = BottomLeftStrip(items, width)
    bl.packing()

    """
    def __init__(self, items, width, height=INF, *args, **kwargs):
        self.items = items
        self.width, self.height = width, height
        self.results = []  # 存储最终结果

    def packing(self):
        raise NotImplementedError("This method has not been implemented!")


class BestFitStrip(BottomLeftStrip):
    """
    思想：先旋转所有工件使得width >= height再按照宽度降序排序(如果宽度相同则高度降序)

    输入:
    items = {item_id: Item(id width height)} -> dict
    width -> int (float)
    height -> int (float)

    输出: :return:
          bl.results = [Corner(id= x= y= width= height=),...] list[<class Corner>]
          bl.residual = [Item(id width height)]

    调用:
    from uti.items import get_2d_item
    Item = get_2d_item()
    # items = {item_id: Item(id=item_id, width=width, height=height)}
    items = {1: Item(id=1, width=1, height=3), 2: Item(id=2, width=2, height=3)}
    width = 10
    bl = BestFitStrip(items, width)
    bl.packing()

    """
    def __init__(self, items, width, height=INF, *args, **kwargs):
        super().__init__(items, width, height, *args, **kwargs)
        self._init()
        self.is_further_optimize = kwargs.get("is_further_optimize", False)
        self.items_by_policy = None

    def _init(self):
        self.skyline = np.zeros(self.width, dtype=int)
        self.gap = Gap(x=0, y=0, width=width)

    def preprocessing_for_items(self) -> list:
        items = []
        for item in self.items.values():
            if item.width < item.height:
                item = item._replace(width=item.height, height=item.width)
            items.append(item)
        # 按照宽度降序和高度降序的规则排序
        items.sort(key=lambda x: (x.width, x.height), reverse=True)
        return items

    @ property
    def is_terminate(self, *args, **kwargs):
        if not self.items_by_policy:
            return True
        if self.gap is None:
            return True
        return False

    def find_best_fit_item(self, *args, **kwargs):
        """
        在当前gap和items下选择最好的item
        items: [Item(id width height)]  按照宽度和高度降序
        """
        items = kwargs.get("items")
        best_item = None
        index = -1
        for index, item in enumerate(items):
            if item.width <= self.gap.width:
                # 如果width放得下，有三种情况可以跳出:(1)这是第一个放得下的，直接跳出;(2)这不是第一个
                # 放得下的item,但是这个width更好(3)这不是第一个但后面也找不到更好的了
                # 为什么不和之前的width比较而是和height?width要是能放下还轮得到现在?
                if best_item is None:
                    best_item = item
                    break
                elif item.width > best_item.height and self.gap.y + item.height <= self.height:
                    best_item = item
                    break
                elif item.width <= best_item.height:
                    break
            elif item.height <= self.gap.width:
                # 只有当某个item width放不下时才进行这一步，因为如果第一个width放得下，则直接结束
                if best_item is None:
                    best_item = item
                elif item.height > best_item.height and self.gap.y + item.width <= self.height:
                    best_item = item

        if best_item is not None:
            if best_item.width > self.gap.width:
                best_item = best_item._replace(width=best_item.height, height=best_item.width)
        return best_item, index

    def pack_a_item(self, item, *args, **kwargs):
        """
        在gap上放置item
        :param item Item(id width height)
        """
        policy = kwargs.get("policy")
        left = False  # 放在gap的左边或者右边 True表示靠左放
        if policy == "leftmost":
            left = True
        else:
            if self.gap.x == 0 and self.gap.x + self.gap.width == self.width:
                left = True
            elif self.gap.x == 0:  # 如果gap在最左边则放在左边
                if policy == "tallest":
                    left = True
            elif self.gap.x + self.gap.width == self.width:  # 在最右边
                if policy == "shortest":
                    left = True
            elif self.skyline[self.gap.x - 1] >= self.skyline[self.gap.x + self.gap.width]:
                if policy == "tallest":
                    left = True
            else:
                if policy == "shortest":
                    left = True
        if left:
            cor = Corner(id=item.id, x=self.gap.x, y=self.gap.y, width=item.width, height=item.height)
            self.skyline[self.gap.x: self.gap.x + item.width] += item.height
        else:
            x = self.gap.x + self.gap.width - item.width
            cor = Corner(id=item.id, x=x, y=self.gap.y, width=item.width, height=item.height)
            self.skyline[x: x + item.width] += item.height
        return cor

    def set_gap(self):
        y = min(self.skyline)
        if y == self.height:
            self.gap = None
            return
        collection = []
        for i, s in enumerate(self.skyline):
            if s == y:
                collection.append(i)
                continue
            if collection:
                break
        self.gap = Gap(x=collection[0], y=y, width=collection[-1] - collection[0] + 1)

    def raise_gap(self):
        """
        将该最低gap升到两边最近的gap上
        """
        if self.gap.x == 0:
            left = self.height
        else:
            left = self.skyline[self.gap.x - 1]
        if self.gap.x + self.gap.width == self.width:
            right = self.height
        else:
            right = self.skyline[self.gap.x + self.gap.width]

        self.skyline[self.gap.x: self.gap.x + self.gap.width] = min(left, right)
        self.set_gap()

    def further_optimize(self):
        raise NotImplementedError("This method has not been implemented!")

    def _packing(self, **kwargs):
        self.items_by_policy = kwargs.get("items")
        policy = kwargs.get("policy")
        # 1. 初始化
        results = Results()
        results.h = 0  # 初始高度为0

        # 2.循环放置直到满足终止条件
        while not self.is_terminate:
            # 2.1 选择当前最低gap下能放置的item
            best_fit_item, index = self.find_best_fit_item(items=self.items_by_policy)

            # 2.2 放置这个item(if any)并更新相应布局
            if best_fit_item is not None:
                # 得到放置后的占角动作(同时更新self.skyline)
                corner = self.pack_a_item(best_fit_item, policy=policy)
                results.append(corner)
                self.set_gap()  # 获得当前最低gap
                self.items_by_policy.pop(index)  # 删除best_fit_item
            else:
                self.raise_gap()  # raise gap并返回当前最低gap

        if self.is_further_optimize:
            self.further_optimize()

        return results

    def packing(self):
        items = self.preprocessing_for_items()
        res = []
        for policy in ['leftmost', 'tallest', 'smallest']:
            _items = items.copy()
            self.skyline = np.zeros(self.width, dtype=int)
            self.gap = Gap(x=0, y=0, width=self.width)
            cur = self._packing(items=_items, policy=policy)
            res.append(cur)
        bl.results = min(res, key=lambda x: x.h)
        return bl.results


class EfficientBestFitStrip(BestFitStrip):
    """
    思想：在best-fit的基础上
    (1)每个gap由(x y width)表示，且由最小堆(以y排序)和双向链表(以x排序)来存储所有gap
    (2)每个item以及副本(如果item.width != item.height)由平衡二叉树存储

    输入:
    items = {item_id: Item(id width height)} -> dict
    width -> int (float)
    height -> int (float)

    输出: :return:
          bl.results = [Corner(id= x= y= width= height=),...] list[<class Corner>]
          bl.residual = [Item(id width height)]

    调用:
    from uti.items import get_2d_item
    Item = get_2d_item()
    # items = {item_id: Item(id=item_id, width=width, height=height)}
    items = {1: Item(id=1, width=1, height=3), 2: Item(id=2, width=2, height=3)}
    width = 10
    bl = EfficientBestFitStrip(items, width)
    bl.packing()

    """
    def __init__(self, items, width, height=INF, *args, **kwargs):
        self.bbs = None
        self.linked_list, self.heap = None, None
        self.params = []
        super().__init__(items, width, height, *args, **kwargs)

    def _init(self):
        linked_list, heap = self._init_segments()

        bbs = self._init_items()
        self.params = [linked_list, heap, bbs]

    def _init_segments(self):
        # 初始化存储segment的最小堆和双向链表
        node = Node(key=Segment(x=0, y=0, width=self.width))
        heap = Heap()  # 初始化最小堆
        heap.push(node)

        linked_list = DoubleLinkList()  # 双向链表
        linked_list.insert(0, node)  # 插入初始结点
        linked_list.head.key = Segment(x=0, y=self.height, width=0)
        linked_list.tail.key = Segment(x=self.width, y=self.height, width=0)
        return linked_list, heap

    def _init_items(self):
        bbs = BalancedBinaryTree()
        for item in self.items.values():
            if item.width <= self.width:
                bbs.insert(key=TreeKey(w=item.width, h=item.height, id=item.id), value=item)
            if hasattr(item, "oriented") and getattr(item, "oriented") is True or item.height == item.width:
                pass
            else:
                if item.height <= self.width:
                    _item = item._replace(width=item.height, height=item.width)
                    bbs.insert(key=TreeKey(w=item.height, h=item.width, id=item.id),
                               value=_item)
        return bbs

    @property
    def is_terminate(self):
        """
        达到终止条件当:
        (1) 所有item都放置完
        (2) height是有限的且当前item不可放置
        :return:
        """
        if self.bbs.root is None or not self.heap:  # 如果树空了终止
            return True
        return False

    def find_best_fit_item(self, *args, **kwargs):
        """
        find the best fit item with largest width being less than the seg,
        breaking the tie with larger height
        :return:
        """
        seg = kwargs.get("seg")
        item = [0, 0, None]
        w, h = seg.width, self.height - seg.y

        def search(node):
            if node is None:
                return

            if node.key.w > w:
                search(node.left)
            elif node.key.h > h:
                search(node.left)
                search(node.right)
            else:
                if [node.key.w, node.key.h] > item:
                    item[:] = node.key.w, node.key.h, node.key.id
                search(node.right)

        tree_node = self.bbs.root  # 平衡二叉搜索树根节点
        search(tree_node)

        if item[2] is None:
            return None
        return tree_node.value._replace(id=item[2], width=item[0], height=item[1])

    def raise_gap(self):
        """

        :return:
        """

        node = self.heap.top()
        seg = node.seg
        dummy_item = self.bbs.root.value._replace(id=0, width=seg.w, height=min(
            node.prior.seg.y, node.next.seg.y) - node.seg.y)
        self.pack_a_item(dummy_item, x=seg.x)

    def pack_a_item(self, item, *args, **kwargs):
        x = kwargs.get("x")
        node = self.heap.top()  # segment with smallest y without popping it from heap
        new_node = Node(key=Segment(x=x, y=node.key.y + item.height, width=item.width))
        if new_node.key.y == self.height:
            self.heap.delete(node)
            self.linked_list.pop(node)
        else:
            self.heap.replace(node, new_node)
            self.linked_list.replace(node, new_node)
        assert self.heap.size == self.linked_list.size
        left = right = False
        if x > node.key.x:  # 左侧有空隙
            low_node = Node(key=Segment(x=node.key.x, y=node.key.y, width=x - node.key.x))
            self.heap.push(low_node)
            self.linked_list.insert_before(node.next, low_node)
            right = True
        elif x + item.width < node.key.x + node.key.width:  # 右侧有空隙
            low_node = Node(key=Segment(x=new_node.key.x + new_node.key.width, y=node.key.y,
                                        width=node.key.x + node.key.width - new_node.key.x - new_node.key.width))
            self.heap.push(low_node)
            self.linked_list.insert_after(node.prior, low_node)
            left = True
        else:
            left = right = True
        if left:
            if new_node.key.y == node.prior.key.y != self.height:
                merge_node = Node(key=Segment(x=new_node.prior.seg.x, y=new_node.prior.seg.y,
                                              width=new_node.prior.seg.w + new_node.key.w))
                self.heap.replace(new_node.prior, merge_node)
                self.linked_list.replace(new_node.prior, merge_node)
                self.heap.delete(new_node)
                self.linked_list.pop(new_node)
                new_node = merge_node

        if right:
            if new_node.key.y == node.next.key.y != self.height:
                merge_node = Node(key=Segment(x=new_node.key.x, y=new_node.key.y,
                                              width=new_node.key.w + new_node.next.key.w))
                self.heap.replace(new_node.next, merge_node)
                self.linked_list.replace(new_node.next, merge_node)
                self.heap.delete(new_node)
                self.linked_list.pop(new_node)
                new_node = merge_node
        return new_node

    def pack_best_fit_item(self, item, policy):
        """

        :param item:
        :param policy:
        :return:
        """
        node = self.heap.top()  # 最小的gap
        seg = node.key
        left = True if node.prior.key.y >= node.next.key.y else False  # 左边更高
        if policy == "leftmost":
            x = seg.x
        elif policy == "tallest":
            x = seg.x if left else seg.x + seg.width - item.width
        else:
            x = seg.x if not left else seg.x + seg.width - item.width

        self.pack_a_item(item, x=x)
        return x, seg.y

    def update_items_info(self, item):
        """

        :param item:
        :return:
        """
        self.bbs.delete(TreeKey(w=item.width, h=item.height, id=item.id))
        if hasattr(item, "oriented") and getattr(item, "oriented") is True or item.height == item.width:
            pass
        else:
            self.bbs.delete(TreeKey(w=item.height, h=item.width, id=item.id))

    def _packing(self, **kwargs):
        policy = kwargs.get("policy")
        # 初始化
        res = Results()
        self.linked_list, self.heap, self.bbs = copy.deepcopy(self.params)
        res.h = 0

        # 循环放置直到满足终止条件
        while not self.is_terminate:
            # print("ha")
            node = self.heap.top()  # segment with smallest y without popping it from heap
            # find the best-fit item
            item = self.find_best_fit_item(seg=node.key)
            if item is None:  # cannot find the best-fit item, raise the gap
                self.raise_gap()
            else:
                x, y = self.pack_best_fit_item(item, policy)  # 放置并更新最小堆和双向链表
                self.update_items_info(item)  # 更新二叉搜索树

                res.append(Corner(id=item.id, x=x, y=y, width=item.width, height=item.height))
                res.h = max(res.h, y + item.height)

        if self.is_further_optimize:
            self.further_optimize()
        return res

    def packing(self):
        res = []
        for policy in ['leftmost', 'tallest', 'smallest']:

            cur = self._packing(policy=policy)
            res.append(cur)
        bl.results = min(res, key=lambda x: x.h)
        return bl.results


if __name__ == '__main__':
    from uti.items import get_2d_item

    Item = get_2d_item()
    # items = {item_id: Item(id=item_id, width=width, height=height)}
    items = {1: Item(id=1, width=1, height=3), 2: Item(id=2, width=2, height=3)}
    width = 10
    bl = EfficientBestFitStrip(items, width)
    bl.packing()
    print(f"{bl.results=}")
