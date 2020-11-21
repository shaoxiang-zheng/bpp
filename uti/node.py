#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time : 2020/11/21 20:06
# Author: Zheng Shaoxiang
# @Email : zhengsx95@163.com
# Description:

class TreeNode:
    def __init__(self, key, value=None):
        self.key = key
        self.value = value
        self.left, self.right = None, None
        self.height = 0


class Node:
    def __init__(self, key=None):
        self.key = key
        self.prior, self.next = None, None
        
    def __eq__(self, other):
        return self.key == other.key

    def __le__(self, other):
        return self.key <= other.key

    def __gt__(self, other):
        return not self.__le__(other)

    def __ge__(self, other):
        return self.key >= other.key

    def __lt__(self, other):
        return not self.__ge__(other)

    def __str__(self):
        return str(self.key)

    def __repr__(self):
        return str(self.key)


if __name__ == '__main__':
    pass
