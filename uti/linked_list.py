#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time : 2020/11/21 22:10
# Author: Zheng Shaoxiang
# @Email : zhengsx95@163.com
# Description:
from uti.node import Node
from functools import reduce


class Deque:
    def __init__(self):
        self.head, self.tail = Node(), Node()  # 头结点和尾结点
        self.head.next = self.tail
        self.tail.prior = self.head
        self.size = 0

    def __repr__(self):
        s = "["
        p = self.head
        while p.next.next is not None:
            p = p.next
            s += f"{p.key},"
        s += "]"
        return s

    def empty(self):
        return self.size == 0

    def append(self, key):
        self.size += 1
        node = Node(key=key)
        node.next = self.tail
        node.prior = self.tail.prior
        self.tail.prior.next = node
        self.tail.prior = node

    def appendleft(self, key):
        self.size += 1
        node = Node(key=key)
        node.prior = self.head
        node.next = self.head.next
        self.head.next.prior = node
        self.head.next = node

    def pop(self):
        assert not self.empty()
        self.size -= 1
        node = self.tail.prior
        self.tail.prior = self.tail.prior.prior
        self.tail.prior.next = self.tail
        return node.key

    def popleft(self):
        assert not self.empty()
        self.size -= 1
        node = self.head.next
        self.head.next = self.head.next.next
        self.head.next.prior = self.head
        return node.key

    def get_head(self):
        assert not self.empty()
        return self.head.next.key

    def get_tail(self):
        assert not self.empty()
        return self.tail.prior.key


class DoubleLinkList(Deque):
    def __init__(self):
        super().__init__()

    # def __repr__(self):
    #     if self.size == 0:
    #         return "head -> tail"
    #     return reduce(lambda x, y: str(x) + " " + str(y), self)

    def __str__(self):
        return str(self.__repr__())

    def __iter__(self):
        _p = self.head
        while _p.next.next is not None:  # 当p的下一个节点不为尾结点
            yield _p.next
            _p = _p.next

    @staticmethod
    def replace(node, new_node):
        new_node.next = node.next
        new_node.prior = node.prior
        node.prior.next = new_node
        node.next.prior = new_node

    def insert(self, index, node):
        """
        将node添加到双向链表中的第index个位置，index从0开始，且头结点和尾结点不作为计数
        :param index: int size >= index >= 0
        :param node: Node()
        :return:
        """
        assert 0 <= index <= self.size
        p = self.head
        for _ in range(index):
            p = p.next

        node.prior = p
        node.next = p.next

        p.next.prior = node
        p.next = node
        self.size += 1

    def insert_before(self, node, new_node):
        self.insert_after(node.prior, new_node)

    def insert_after(self, node, new_node):
        new_node.next = node.next
        node.next.prior = new_node
        node.next = new_node
        new_node.prior = node
        self.size += 1

    def pop_node(self, node):
        node.prior.next = node.next
        node.next.prior = node.prior
        self.size -= 1


if __name__ == '__main__':
    pass
