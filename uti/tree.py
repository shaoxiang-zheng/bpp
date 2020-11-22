#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time : 2020/11/21 20:07
# Author: Zheng Shaoxiang
# @Email : zhengsx95@163.com
# Description:
from uti.node import TreeNode
from uti.params import INF


class BalancedBinaryTree:
    """
    平衡二叉树：
    (1)对某个节点，左子树所有节点的值小于该节点的值，且右子树所有节点的值大于该节点的值
    (2)对某个节点，左右子树的高度差的绝对值不超过1

    bbt = BalanceBinaryTree()  # 初始化

    bbt.insert(key, value)  # 插入节点
    bbt.delete(key)  # 删除节点

    m = bbt.find_min() # 找到最小的节点
    M = bbt.find_max() # 找到最大的节点

    """
    def __init__(self):
        self.root = None

    @property
    def is_valid(self):
        def _valid(root, _min, _max):
            if root is None:
                return True
            if _min is not None and root.key <= _min.key:
                return False
            if _max is not None and root.key >= _max.key:
                return False
            return _valid(root.left, _min, root) and _valid(root.right, root, _max)

        def height(root):
            if root is None:
                return 0
            return max(height(root.left), height(root.right)) + 1

        def is_balanced(root):
            if root is None:
                return True
            return abs(height(root.left) - height(root.right)) <= 1 and is_balanced(
                root.left) and is_balanced(root.right)

        return _valid(self.root, None, None) and is_balanced(self.root)

    def breadth_first(self):
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            if node is None:
                break
            assert node.key[-1] == node.value.id
            if node.left is not None:
                queue.append(node.left)

            if node.right is not None:
                queue.append(node.right)

    def preorder(self):
        """
        前序遍历
        :return:
        """

        def preorder(node):
            if node is None:
                return
            print(node)
            if node.left is not None:
                preorder(node.left)

            if node.right is not None:
                preorder(node.right)

        preorder(self.root)

    def inorder(self):
        """
        中序遍历
        :return:
        """

        def inorder(node):
            if node is None:
                return
            if node.left is not None:
                inorder(node.left)

            if node.right is not None:
                inorder(node.right)

        inorder(self.root)

    def find_min(self):
        if self.root is None:
            return None
        return self._find_min(self.root)

    def find_max(self):
        if self.root is None:
            return None
        return self._find_max(self.root)

    def _find_min(self, node):
        if node.left:
            return self._find_min(node.left)
        else:
            return node

    def _find_max(self, node):
        if node.right:
            return self._find_max(node.right)
        else:
            return node

    @staticmethod
    def height(node):
        if node is None:
            return -1
        return node.height

    def single_left_rotate(self, node):
        k = node.left
        node.left = k.right
        k.right = node
        node.height = max(self.height(node.right), self.height(node.left)) + 1
        k.height = max(self.height(k.left), node.height) + 1
        return k

    def single_right_rotate(self, node):
        k = node.right
        node.right = k.left
        k.left = node
        node.height = max(self.height(node.right), self.height(node.left)) + 1
        k.height = max(self.height(k.right), node.height) + 1
        return k

    def double_left_rotate(self, node):
        node.left = self.single_right_rotate(node.left)
        return self.single_left_rotate(node)

    def double_right_rotate(self, node):
        node.right = self.single_left_rotate(node.right)
        return self.single_right_rotate(node)

    def insert(self, key, value):
        if self.root is None:  # 空树直接插入
            self.root = TreeNode(key, value)
        else:
            self.root = self._insert(key, value, self.root)

    def _insert(self, key, value, node):
        if node is None:
            node = TreeNode(key, value)
        elif key < node.key:
            node.left = self._insert(key, value, node.left)
            if self.height(node.left) - self.height(node.right) == 2:
                if key < node.left.key:
                    node = self.single_left_rotate(node)
                elif key > node.left.key:
                    node = self.double_left_rotate(node)
                else:
                    raise ValueError("Can't insert a same node into the Tree")
        elif key > node.key:
            node.right = self._insert(key, value, node.right)
            if self.height(node.right) - self.height(node.left) == 2:
                if key > node.right.key:
                    node = self.single_right_rotate(node)
                elif key < node.right.key:
                    node = self.double_right_rotate(node)
                else:
                    raise ValueError("Can't insert a same node into the Tree")
        node.height = max(self.height(node.right), self.height(node.left)) + 1
        return node

    def delete(self, key):
        if self.root is None:
            pass
        else:
            self.root = self._delete(key, self.root)

    def _delete(self, key, node):
        if node is None:
            raise KeyError("Cannot delete a node from an empty tree")
        elif key < node.key:
            node.left = self._delete(key, node.left)
            if self.height(node.right) - self.height(node.left) == 2:
                if self.height(node.right.right) >= self.height(node.right.left):
                    node = self.single_right_rotate(node)
                else:
                    node = self.double_right_rotate(node)
            node.height = max(self.height(node.left), self.height(node.right)) + 1
        elif key > node.key:
            node.right = self._delete(key, node.right)
            if self.height(node.left) - self.height(node.right) == 2:
                if self.height(node.left.left) >= self.height(node.left.right):
                    node = self.single_left_rotate(node)
                else:
                    node = self.double_left_rotate(node)
            node.height = max(self.height(node.left), self.height(node.right)) + 1
        elif node.left and node.right:
            if node.left.height <= node.right.height:
                min_node = self._find_min(node.right)
                node.key = min_node.key
                node.value = min_node.value
                node.right = self._delete(node.key, node.right)
            else:
                max_node = self._find_max(node.left)
                node.key = max_node.key
                node.value = max_node.value
                node.left = self._delete(node.key, node.left)
            node.height = max(self.height(node.left), self.height(node.right)) + 1
        else:
            if node.right:
                node = node.right
            else:
                node = node.left
        return node


class IntervalTreeNode(TreeNode):
    def __init__(self, a, b, key=None, value=None):
        super().__init__(key, value)
        self.a, self.b = a, b
        self.parent = None


class IntervalTree:
    """
    给定一个长度为n的序列sequence,构建快速查询任意区间[c, d]内最小值的区间数

    调用:
    sequence = [2, 1, 5, 6, 4, 2, 8]
    it = IntervalTree(sequence)

    it.find_min(1, 5)
    Output: 1

    it.delete(1)
    # sequence = [2, inf, 5, 6, 4, 2, 8]

    it.find_min(1, 5)
    Output: 2
    """
    def __init__(self, sequence):
        self.sequence = sequence
        self.root = IntervalTreeNode(0, len(sequence) - 1)
        self.hash = {(0, len(sequence) - 1): self.root}  # {(a, b): TreeNode()}
        self._construction_tree()  # 构建树

    def _construction_tree(self):
        def recur(node):
            start, end = node.a, node.b
            if start == end:
                node.value = self.sequence[start]
                return node.value

            mid = int((start + end) / 2)
            left_node = IntervalTreeNode(start, mid)
            right_node = IntervalTreeNode(mid + 1, end)
            node.left = left_node
            node.right = right_node
            left_node.parent, right_node.parent = node, node
            self.hash[start, mid] = left_node
            self.hash[mid + 1, end] = right_node
            node.value = min(recur(node.left), recur(node.right))
            return node.value

        recur(self.root)

    def delete(self, pos):
        """
        删除位置pos上的元素
        :param pos:
        :return:
        """
        self.sequence[pos] = INF

        # method 1: 从叶子节点开始更新，直到子代节点不影响父节点为止
        node = self.hash[pos, pos]
        node.value = INF
        while node.parent is not None:
            node = node.parent
            val = node.value
            node.value = min(node.left.value, node.right.value)
            if val == node.value:
                break

        # method 2: 递归更新，从根节点开始，类似数的构建
        # def recur(node):
        #     if node.a == node.b:
        #         if node.a == pos:
        #             node.value = INF
        #         return node.value
        #
        #     if node.left.a <= pos <= node.left.b:
        #         node.value = min(recur(node.left), node.right.value)
        #     elif node.right.a <= pos <= node.right.b:
        #         node.value = min(node.left.value, recur(node.right))
        #     return node.value
        #
        # recur(self.root)

    def find_min(self, c, d):
        """
        查找在索引位置[c, d]之间的最小值
        :param c:
        :param d:
        :return:
        """
        if not self.root.a <= c <= d <= self.root.b:
            return INF

        def recur(node, sp, ep):
            # 如果起止节点相同(叶子节点)或区间与当前节点重复，直接返回
            if sp == ep:
                return self.sequence[sp]
            if (node.a, node.b) == (sp, ep):  # 查询区间与当前节点区间重复
                return node.value

            if node.left.a <= sp <= ep <= node.left.b:
                return recur(node.left, sp, ep)
            if node.right.a <= sp <= ep <= node.right.b:
                return recur(node.right, sp, ep)
            # 递归更新其左右节点
            lm, rm = INF, INF
            if sp <= node.left.b:
                lm = recur(node.left, sp, node.left.b)
            if ep >= node.right.a:
                rm = recur(node.right, node.right.a, ep)
            return min(lm, rm)

        m = recur(self.root, c, d)
        return m


if __name__ == '__main__':
    pass
