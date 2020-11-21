#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time : 2020/11/21 20:07
# Author: Zheng Shaoxiang
# @Email : zhengsx95@163.com
# Description:
from uti.node import TreeNode


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


if __name__ == '__main__':
    pass
