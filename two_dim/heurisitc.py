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
from utility.params import INF
from utility.items import Corner, Gap, Results, Segment, TreeKey
from utility.tree import BalancedBinaryTree, IntervalTree
import numpy as np
from utility.heap import Heap
from utility.node import Node
from utility.linked_list import DoubleLinkList, Deque
from collections import defaultdict


class BottomLeftStrip:
    """
    思想：按照输入的顺序从右上角开始，轮流向下和向左，直到item不能移动为止

    输入:
    items = {item_id: Item(id width height)} -> dict
    width -> int (float)
    height -> int (float)

    输出: :return:
          self.results = [Corner(id= x= y= width= height=),...] list[<class Corner>]

    调用:
    from utility.items import get_2d_item
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
    每次在最低的gap上选择best fit的item

    输入:
    items = {item_id: Item(id width height)} -> dict
    width -> int (float)
    height -> int (float)

    输出: :return:
          self.results = [Corner(id= x= y= width= height=),...] list[<class Corner>]

    调用:
    from utility.items import get_2d_item
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
        pass

    def pack_for_a_bin(self, **kwargs):
        """
        给定当前items 和一个箱子,装到箱子装完或箱子装不下为止
        :return 返回布局和剩余工件
        """
        items = kwargs.get("items")
        policy = kwargs.get("policy")
        # 每个箱子需要初始化skyline和gap
        self.skyline = np.zeros(self.width, dtype=int)
        self.gap = Gap(x=0, y=0, width=self.width)
        cur = Results()
        cur.h = 0

        while not self.is_terminate:
            # 2.1 选择当前最低gap下能放置的item
            best_fit_item, index = self.find_best_fit_item(items=items)

            # 2.2 放置这个item(if any)并更新相应布局
            if best_fit_item is not None:
                # 得到放置后的占角动作(同时更新self.skyline)
                corner = self.pack_a_item(best_fit_item, policy=policy)
                cur.append(corner)
                self.set_gap()  # 获得当前最低gap
                items.pop(index)  # 删除best_fit_item
                cur.h = max(cur.h, corner.y + corner.height)
            else:
                self.raise_gap()  # raise gap并返回当前最低gap

        if self.is_further_optimize:
            self.further_optimize()
        return cur

    def _packing(self, **kwargs):
        self.items_by_policy = kwargs.get("items")
        policy = kwargs.get("policy")

        # 注：self.items_by_policy已相应更新为剩余零件
        cur = self.pack_for_a_bin(items=self.items_by_policy, policy=policy)

        return cur

    def select_schedule_by_policy(self, res):
        return min(res, key=lambda x: x.h)

    def packing(self):
        items = self.preprocessing_for_items()
        res = []
        for policy in ['leftmost', 'tallest', 'smallest']:
            _items = items.copy()
            cur = self._packing(items=_items, policy=policy)
            res.append(cur)
        self.results = self.select_schedule_by_policy(res)

        return self.results


class BestFitBin(BestFitStrip):
    """
    思想：先旋转所有工件使得width >= height再按照宽度降序排序(如果宽度相同则高度降序)
        在best-fit-strip的基础上设置高度height,若放置item的高度超过,则不放置
    输入:
    items = {item_id: Item(id width height)} -> dict
    width -> int (float)
    height -> int (float)

    输出: :return:
          self.results = [Corner(id= x= y= width= height=),...] list[<class Corner>]

    调用:
    from utility.items import get_2d_item
    Item = get_2d_item()
    # items = {item_id: Item(id=item_id, width=width, height=height)}
    items = {1: Item(id=1, width=1, height=3), 2: Item(id=2, width=2, height=3)}
    width = 10
    height = 10
    bl = BestFitBin(items, height)
    bl.packing()

    """
    def __init__(self, items, width, height, *args, **kwargs):
        super().__init__(items, width, height, *args, **kwargs)

    def _packing(self, **kwargs):
        self.items_by_policy = kwargs.get("items")
        policy = kwargs.get("policy")

        schedule = {}
        k = 0  # 箱子编号
        while self.items_by_policy:
            k += 1
            cur = self.pack_for_a_bin(items=self.items_by_policy, policy=policy)
            schedule[k] = cur
        return schedule

    def select_schedule_by_policy(self, schedule):
        return min(schedule, key=lambda x: len(x))


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

    调用:
    from utility.items import get_2d_item
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
        super().__init__(items, width, height, *args, **kwargs)

    def _init(self):
        pass

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

    def further_optimize(self):
        pass

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
        seg = node.key
        dummy_item = self.bbs.root.value._replace(id=0, width=seg.width, height=min(
            node.prior.key.y, node.next.key.y) - node.key.y)
        self.pack_a_item(dummy_item, x=seg.x)

    def pack_a_item(self, item, *args, **kwargs):
        x = kwargs.get("x")
        # print(f"{item=}")
        node = self.heap.top()  # segment with smallest y without popping it from heap
        new_node = Node(key=Segment(x=x, y=node.key.y + item.height, width=item.width))
        if new_node.key.y == self.height:
            self.heap.delete(node)
            self.linked_list.pop_node(node)
        else:
            self.heap.replace(node, new_node)
            self.linked_list.replace(node, new_node)
        assert self.heap.size == self.linked_list.size
        left = right = True
        if x > node.key.x:  # 左侧有空隙
            low_node = Node(key=Segment(x=node.key.x, y=node.key.y, width=x - node.key.x))
            self.heap.push(low_node)
            self.linked_list.insert_after(node.prior, low_node)
            left = False
        elif x + item.width < node.key.x + node.key.width:  # 右侧有空隙
            low_node = Node(key=Segment(x=new_node.key.x + new_node.key.width, y=node.key.y,
                                        width=node.key.x + node.key.width - new_node.key.x - new_node.key.width))
            self.heap.push(low_node)
            self.linked_list.insert_before(node.next, low_node)
            right = False
        else:
            left = right = True
        if left:  # leftmost
            if new_node.key.y == node.prior.key.y != self.height and new_node.key.x == node.prior.key.x + node.prior.key.width:
                merge_node = Node(key=Segment(x=new_node.prior.key.x, y=new_node.prior.key.y,
                                              width=new_node.prior.key.width + new_node.key.width))
                self.heap.replace(new_node.prior, merge_node)
                self.linked_list.replace(new_node.prior, merge_node)
                self.heap.delete(new_node)
                self.linked_list.pop_node(new_node)
                new_node = merge_node

        if right:
            if new_node.key.y == node.next.key.y != self.height and new_node.key.x + new_node.key.width == node.next.key.x:
                merge_node = Node(key=Segment(x=new_node.key.x, y=new_node.key.y,
                                              width=new_node.key.width + new_node.next.key.width))
                self.heap.replace(new_node.next, merge_node)
                self.linked_list.replace(new_node.next, merge_node)
                self.heap.delete(new_node)
                self.linked_list.pop_node(new_node)
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

    def pack_for_a_bin(self, **kwargs):
        """
        在当前bbs(表示items)下装箱
        """
        policy = kwargs.get("policy")
        # 对每个箱子，linked_list和heap需要初始化，bbs则不用
        self.linked_list, self.heap = self._init_segments()
        cur = Results()
        cur.h = 0

        # 循环放置直到满足终止条件
        while not self.is_terminate:
            node = self.heap.top()  # segment with smallest y without popping it from heap
            # find the best-fit item
            item = self.find_best_fit_item(seg=node.key)
            if item is None:  # cannot find the best-fit item, raise the gap
                self.raise_gap()
            else:
                x, y = self.pack_best_fit_item(item, policy)  # 放置并更新最小堆和双向链表
                self.update_items_info(item)  # 更新二叉搜索树

                cur.append(Corner(id=item.id, x=x, y=y, width=item.width, height=item.height))
                cur.h = max(cur.h, y + item.height)

        if self.is_further_optimize:
            self.further_optimize()
        return cur

    def _packing(self, **kwargs):
        policy = kwargs.get("policy")

        # 每次装完箱子bbs会自动更新,
        cur = self.pack_for_a_bin(policy=policy)
        return cur

    def select_schedule_by_policy(self, res):
        return min(res, key=lambda x: x.h)

    def packing(self):
        res = []
        for policy in ['leftmost', 'tallest', 'smallest']:
            self.bbs = self._init_items()
            cur = self._packing(policy=policy)
            res.append(cur)
        self.results = self.select_schedule_by_policy(res)
        return self.results


class EfficientBestFitBin(EfficientBestFitStrip):
    """
    思想：在efficient best-fit的基础上
    (1)每个gap由(x y width)表示，且由最小堆(以y排序)和双向链表(以x排序)来存储所有gap
    (2)每个item以及副本(如果item.width != item.height)由平衡二叉树存储

    输入:
    items = {item_id: Item(id width height)} -> dict
    width -> int (float)
    height -> int (float)

    输出: :return:
          self.results = [Corner(id= x= y= width= height=),...] list[<class Corner>]

    调用:
    from utility.items import get_2d_item
    Item = get_2d_item()
    # items = {item_id: Item(id=item_id, width=width, height=height)}
    items = {1: Item(id=1, width=1, height=3), 2: Item(id=2, width=2, height=3)}
    width = 10
    height = 10
    bl = EfficientBestFitBin(items, width, height)
    bl.packing()

    """
    def __init__(self, items, width, height, *args, **kwargs):
        super().__init__(items, width, height, *args, **kwargs)

    def _packing(self, **kwargs):
        policy = kwargs.get("policy")
        schedule = {}
        k = 0
        while self.bbs.root is not None:  # 还有工件
            k += 1
            cur = self.pack_for_a_bin(policy=policy)
            schedule[k] = cur
        return schedule

    def select_schedule_by_policy(self, res):
        return min(res, key=lambda x: len(x))


class BestFitPackStrip(EfficientBestFitStrip):
    """
    思想:
    (1)每个gap由(x y width)表示，且由最小堆(以y排序)和双向链表(以x排序)来存储所有gap
    (2)每个item在现在最低的gap上如果放进去则与当前gap的完美贴边数称之为fitness
       如果存在多个fitness则选择原输入序列中最靠前的一个

    输入:
    items = {item_id: Item(id width height)} -> dict
    width -> int (float)
    height -> int (float)

    输出: :return:
          self.results = [Corner(id= x= y= width= height=),...] list[<class Corner>]

    调用:
    from utility.items import get_2d_item
    Item = get_2d_item()
    # items = {item_id: Item(id=item_id, width=width, height=height)}
    items = {1: Item(id=1, width=1, height=3), 2: Item(id=2, width=2, height=3)}
    width = 10
    bl = BestFitPackStrip(items, width)
    bl.packing()

    """
    def __init__(self, items, width, height=INF, *args, **kwargs):
        super().__init__(items, width, height, *args, **kwargs)
        self.packed_id = set()  # 记录当前已经放置的item id

    def _init(self):
        # 初始化

        self._init_items()

    def _init_items(self):
        # s1 {(width, height): [index] } 记录特定item在原序列中的位置
        # s1 以item (width, height)作为字典key值，以便在需要查找特定item时可以直接查找
        self.s1 = defaultdict(Deque)  # 双向链表添加和删除第一个元素的时间复杂度都为O(1)

        # {index: [item, _item]}  原序列中的位置对应的两个item
        self.all_items = {}  # 该列表存储所有item及其旋转90°后的item(如果两个item的宽度和高度不等)

        self.id_to_pos = {}  # item id以及其在原序列中的位置
        for index, item in enumerate(self.items.values()):
            self.all_items[index] = []
            if item.width <= self.width and item.height <= self.height:
                self.all_items[index].append(item)
                self.s1[item.width, item.height].append(index)
            if item.width <= self.height and item.width <= self.width and item.width != item.height:
                self.all_items[index].append(item._replace(width=item.height, height=item.width))
                self.s1[item.height, item.width].append(index)
            self.id_to_pos[item.id] = index

        assert len(self.id_to_pos) == len(self.items), "输入items的id不应该相同"

        _items = [item for items in self.all_items.values() for item in items]
        # s2按照item宽度升序排序(相同宽度按高度升序)
        # tree2记录给定任意给定两个s2中的区间位置[c, d]之间的最小索引值(当找到多个得分相同的item
        # 最小的索引即在原序列中位置最靠前的)
        # id_to_pos2表示指定id和方向的item在s2中的位置(方便在O(1)时间内找到s2中的位置以便更新tree2)
        self.s2 = sorted(_items, key=lambda x: (x.width, x.height))
        self.tree2 = IntervalTree([self.id_to_pos[item.id] for item in self.s2])
        # print(f"{self.tree2.sequence}")
        # {(id, width): pos}  根据item id和width 定位到 其在s2中的索引
        self.id_to_pos2 = {(item.id, item.width): index for index, item in enumerate(self.s2)}
        # s3按照item高度升序排序(相同高度按宽度升序)
        self.s3 = sorted(_items, key=lambda x: (x.height, x.width))
        self.tree3 = IntervalTree([self.id_to_pos[item.id] for item in self.s3])
        self.id_to_pos3 = {(item.id, item.width): index for index, item in enumerate(self.s3)}

    @ property
    def is_terminate(self):
        if self.heap.empty():  # 没有放置的位置
            # print("no place")
            return True
        if not self.s1:
            # print("no item")
            return True
        return False

    def pop_item_from_s1(self, w, h):
        self.s1[w, h].popleft()
        if self.s1[w, h].empty():
            self.s1.pop((w, h))

    def update_s1(self, w, h):
        # 删除item = (w, h)及其旋转副本(如果存在)在链表中的位置
        self.pop_item_from_s1(w, h)
        if w != h and (h, w) in self.s1:
            self.pop_item_from_s1(h, w)

    @staticmethod
    def satisfy1(sequence, pos, w, h):
        if sequence[pos].width == w:
            return True
        return False

    @staticmethod
    def to_left1(sequence, pos, w, h):
        if sequence[pos].width > w:
            return True
        return False

    @staticmethod
    def satisfy2(sequence, pos, w, h):
        if sequence[pos].width == w and sequence[pos].height <= h:
            return True
        return False

    @staticmethod
    def to_left2(sequence, pos, w, h):
        if sequence[pos].width > w or (sequence[pos].width == w and sequence[pos].height > h):
            return True
        return False

    @staticmethod
    def satisfy3(sequence, pos, w, h):
        if sequence[pos].height == h and sequence[pos].width <= w:
            return True
        return False

    @staticmethod
    def to_left3(sequence, pos, w, h):
        if sequence[pos].height > h or (sequence[pos].height == h and sequence[pos].width > w):
            return True
        return False

    @staticmethod
    def search_index_in_sequence(sequence, w, h, to_left, satisfy):
        """
        搜索sequence中满足函数satisfy要求的元素的起止位置，若不存在返回(-1, -1)
        满足要求的元素一定是连续的
        :param sequence:
        :param w:
        :param h:
        :param to_left: to_left(sequence, pos, w, h) 判断满足要求的元素在当前元素pos左侧(不包括当前元素)
        :param satisfy:
        :return:
        """
        def search(left=True):
            lo, hi = 0, len(sequence)
            while lo < hi:
                mid = lo + (hi - lo) // 2
                if to_left(sequence, mid, w, h) or (left and satisfy(sequence, mid, w, h)):
                    hi = mid
                else:
                    lo = mid + 1
            return lo

        left_index = search()
        if left_index == len(sequence) or not satisfy(sequence, left_index, w, h):
            return -1, -1
        right_index = search(left=False) - 1
        return left_index, right_index

    def get_item_by_width(self, index, w):
        # 通过在原序列中的索引与其与箱子width平行的边的长度寻找对应的item
        if self.all_items[index][0].width == w:
            return self.all_items[index][0]
        return self.all_items[index][1]

    def get_item_by_height(self, index, h):
        # 通过在原序列中的索引与其与箱子height平行的边的长度寻找对应的item
        if self.all_items[index][0].height == h:
            return self.all_items[index][0]
        return self.all_items[index][1]

    def find_specific_item(self, w, h):
        if (w, h) in self.s1:
            index = self.s1[w, h].get_head()  # 返回对应item在原序列中的索引
            item = self.get_item_by_width(index, w)
            return item
        return None

    def find_best_item(self, node, seg, x):
        if node.prior.key.y == node.next.key.y:  # case 1: s.lh = s.rh
            # 此类情况fitness只可能取值3, 1, 0
            # 是否存在fitness == 3的item，同时更新s1
            w, h = seg.width, node.prior.key.y - seg.y
            item = self.find_specific_item(w, h)
            if item is not None:
                return item, x
        else:  # case 2: s.lh ！= s.rh
            # fitness = 2, 1, 0
            # 是否存在fitness=2的item，返回找到的第一个
            w, lh, rh = seg.width, node.prior.key.y - seg.y, node.next.key.y - seg.y
            l_item = self.find_specific_item(w, lh)
            r_item = self.find_specific_item(w, rh)
            if l_item is not None and r_item is not None:
                if self.id_to_pos[l_item.id] <= self.id_to_pos[r_item.id]:
                    return l_item, x
                else:
                    return r_item, x
            elif l_item is not None:
                return l_item, x
            elif r_item is not None:
                return r_item, x
        return None, x

    def get_item_with_width(self, w, **kwargs):
        """
        在self.s2中找到width=w的item中index最靠前的一个
        """
        min_pos, max_pos = self.search_index_in_sequence(
            self.s2, w, INF, self.to_left1, self.satisfy1)
        r_index = self.tree2.find_min(min_pos, max_pos)  # 找到元素中最小的id
        return r_index

    def search_s4_item(self, w, h):
        """
        搜索self.s2中最后一个满足item.width <= w的item_id:
        :param w:
        :param h:
        :return:
        """
        # print(f"{self.s2=}")
        index = -1
        if self.s2[-1].width <= w:
            index = len(self.s2) - 1
        if index == -1:
            left, right = 0, len(self.s2) - 1
            while left < right:
                mid = left + (right - left) // 2
                if not self.s2[mid].width <= w:
                    right = mid
                else:
                    left = mid + 1
            index = left - 1
        # print(f"{index=}")
        if index == -1:  # 不存在
            return None
        # print(f"{self.tree2.sequence}")
        r_index = self.tree2.find_min(0, index)
        # print(f"{r_index=}")
        if r_index == INF:
            return None
        if self.all_items[r_index][0].width <= w:
            return self.all_items[r_index][0]
        return self.all_items[r_index][1]

    def find_second_item(self, node, seg, x):
        # case 3: 存在fitness = 1的item 取最小的id
        item_index, item = INF, None
        # case 3.1 寻找是否存在item.w = seg.w且item.h <= self.h - seg.y的item,若存在,找到id最小的一个
        r_index = self.get_item_with_width(seg.width, h=self.height - seg.y)
        if r_index < item_index:
            item_index, item = r_index, self.get_item_by_width(r_index, seg.width)

        # case 3.2 寻找是否存在item.h = node.prior.key.y and item.w <= seg.w的item
        # case 3.3 寻找是否存在item.h = node.right.key.y and item.w <= seg.w的item
        for h in [node.prior.key.y - seg.y, node.next.key.y - seg.y]:
            min_pos, max_pos = self.search_index_in_sequence(self.s3, seg.width, h, self.to_left3, self.satisfy3)
            r_index = self.tree3.find_min(min_pos, max_pos)
            if r_index < item_index:
                item_id, item = r_index, self.get_item_by_height(r_index, h)
        if item_index < INF:
            if item.height == node.next.key.y:
                x = seg.x + seg.width - item.width
            return item, x
        return None, x

    def find_third_item(self, node, seg, x):
        # print(f"{seg=}")
        # case 4: 找到fitness = 0的item
        item = self.search_s4_item(seg.width, self.height - seg.y)
        # print(f"{item=}")
        if item is not None:
            if node.prior.key.y < node.next.key.y:
                x = seg.x + seg.width - item.width
        return item, x

    def find_best_fit_item(self, *args, **kwargs):
        node = kwargs.get("node")
        seg = node.key
        x = seg.x
        # print(f"all items = {self.s1}")

        # step 1: 判断是否存在fitness = 3 or 2的item
        item, x = self.find_best_item(node, seg, x)
        if item is not None:
            return item, x
        # print(f"best = {item}")
        # step 2: 判断是否存在fitness = 1的item
        item, x = self.find_second_item(node, seg, x)
        if item is not None:
            return item, x
        # print(f"second = {item}")
        # step 3: 判断是否存在fitness = 0的item
        item, x = self.find_third_item(node, seg, x)
        if item is not None:
            return item, x
        # print(f"third = {item}")
        return None, x

    def raise_gap(self):
        node = self.heap.top()
        seg = node.key
        if node.prior.key.x + node.prior.key.width == node.key.x:
            height = node.prior.key.y
        else:
            height = self.height

        if node.next.key.x == node.key.x + node.key.width:
            height = min(height, node.next.key.y)
        else:
            height = min(height, self.height)
        dummy_item = self.s2[0]._replace(id=0, width=seg.width, height=height - node.key.y)
        # print(f"{dummy_item=}\n{seg=}\n{node=}")
        self.pack_a_item(dummy_item, x=seg.x)

    def update_items_info(self, item):
        """
        更新s1 s2 s3
        :param item:
        :return:
        """
        self.packed_id.add(item.id)

        # update s1
        # 在存储元素的s1中删除item及其旋转后的副本(如果存在的话),s2和s3不需要更新
        # 如果双向链表为空，还需删除对应的字典key值
        w, h = item.width, item.height

        self.update_s1(w, h)

        # s2, s3无需更新

        # update tree2, tree3
        # 主要更新s2中对应区间位置[c, d]内id的最小值(先将已经放置的id位置置为inf,若item存在旋转后的副本,则需要删除两个位置)
        pos = self.id_to_pos2[item.id, item.width]
        self.tree2.delete(pos)
        if item.width != item.height:
            pos = self.id_to_pos2[item.id, item.height]
            self.tree2.delete(pos)

        pos = self.id_to_pos3[item.id, item.width]
        self.tree3.delete(pos)
        if item.width != item.height:
            pos = self.id_to_pos3[item.id, item.height]
            self.tree3.delete(pos)

    def pack_for_a_bin(self, **kwargs):
        cur = Results()
        cur.h = 0
        self.linked_list, self.heap = self._init_segments()
        while not self.is_terminate:
            node = self.heap.top()
            # print(f"lowest segment = {node.key}")
            y = node.key.y  # 最低gap的高度
            # 返回寻找到的最好的item(得分最高，多个得分相同则在原序列中最靠前的一个)及其放置的x位置
            item, x = self.find_best_fit_item(node=node)
            # print(f"{x=}")
            # print(f"best fit {item=}\n")
            if item is None:
                self.raise_gap()
            else:
                self.pack_a_item(item, x=x)
                self.update_items_info(item)

                cur.append(Corner(id=item.id, x=x, y=y, width=item.width, height=item.height))
                cur.h = max(cur.h, y + item.height)
            # print(f"{self.heap=}\n{self.linked_list=}")

        if self.is_further_optimize:
            self.further_optimize()
        # print(f"{cur=}")
        return cur

    def _packing(self, **kwargs):
        cur = self.pack_for_a_bin()
        return cur

    def packing(self, **kwargs):
        batch_limit = kwargs.get("batch_limit", INF)
        self.results = self._packing(batch_limit=batch_limit)
        return self.results


class BestFitPackBin(BestFitPackStrip):
    """
    思想:
    (1)每个gap由(x y width)表示，且由最小堆(以y排序)和双向链表(以x排序)来存储所有gap
    (2)每个item在现在最低的gap上如果放进去则与当前gap的完美贴边数称之为fitness
       如果存在多个fitness则选择原输入序列中最靠前的一个

    输入:
    items = {item_id: Item(id width height)} -> dict
    width -> int (float)
    height -> int (float)

    输出: :return:
          self.results = [Corner(id= x= y= width= height=),...] list[<class Corner>]

    调用:
    from utility.items import get_2d_item
    Item = get_2d_item()
    # items = {item_id: Item(id=item_id, width=width, height=height)}
    items = {1: Item(id=1, width=1, height=3), 2: Item(id=2, width=2, height=3)}
    width = 10
    height = 10
    bl = BestFitPackBin(items, width, height)
    bl.packing()

    """
    def __init__(self, items, width, height, *args, **kwargs):
        super().__init__(items, width, height, *args, **kwargs)

    def search_s4_item(self, w, h):
        """
        搜索self.all_items中第一个满足item.width <= w and item.height <= h的item_id:
        :param w:
        :param h:
        :return:
        """
        for items in self.all_items.values():
            for item in items:
                if item.width <= w and item.height <= h and item.id not in self.packed_id:
                    return item
        return None

    def get_item_with_width(self, w, **kwargs):
        """
        在self.s2中找到width=w且height<=h的item中index最靠前的一个
        """
        h = kwargs.get("h")
        min_pos, max_pos = self.search_index_in_sequence(self.s2, w, h, self.to_left2, self.satisfy2)
        r_index = self.tree2.find_min(min_pos, max_pos)  # 找到元素中最小的id
        return r_index

    def _packing(self, **kwargs):
        batch_limit = kwargs.get("batch_limit", INF)
        assert batch_limit >= 1
        schedule = {}
        k = 0
        while self.s1 and k < batch_limit:  # 当还存在items未排
            k += 1
            schedule[k] = self.pack_for_a_bin()
        return schedule


class COA:
    pass


if __name__ == '__main__':
    from utility.items import get_2d_item
    import random
    from utility.draw import draw_2d_pattern
    Item = get_2d_item()
    # items = {item_id: Item(id=item_id, width=width, height=height)}
    n = 10
    items = {item_id: Item(id=item_id, width=random.randint(1, 10),
                           height=random.randint(1, 10)) for item_id in range(1, n + 1)}

    width = 10
    height = INF
    bl = BestFitPackStrip(items, width, height=height)
    bl.packing()
    draw_2d_pattern(bl.results, width, height)
    print(f"{bl.results=}")
