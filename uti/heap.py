#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time : 2020/11/10 21:55
# Author: Zheng Shaoxiang
# @Email : zhengsx95@163.com
# Description:


class Heap:
    def __init__(self, data=None):
        if data is not None:
            assert isinstance(data, list), "inputs should be list"
        self.info = {}  # (x, y, w)->index
        self.size = 0
        if data is None:
            self.data = []
        else:
            self.data = data
            raise NotImplementedError()

    def __repr__(self):
        return str(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __len__(self):
        return len(self.data)

    def empty(self):
        return self.size == 0

    def top(self):
        return self.data[0]

    def push(self, item):
        self.data.append(item)
        self.size += 1
        self.info[item.key] = self.size - 1
        self._siftdown(0, self.size - 1)

    def pop(self):
        last_item = self.data.pop()
        self.info.pop(last_item.key)
        self.size -= 1
        if self.data:
            return_item = self.data[0]
            self.data[0] = last_item
            self.info.pop(return_item.key)

            self._siftup(0)
            return return_item
        return last_item

    def delete(self, item):
        pos = self.info[item.key]  # 获取该元素位置
        last_item = self.data.pop()
        self.info.pop(last_item.key)
        self.size -= 1
        if pos == self.size:
            return

        if self.data:
            self.info.pop(item.key)
            self.data[pos] = last_item
            self._siftup(pos)
            self._siftdown(0, pos)  # 如果被删除元素在最下一层需要考虑向上

    def replace(self, item, new_item):
        pos = self.info[item.key]
        self.info.pop(item.key)
        self.info[new_item.key] = pos
        end_pos = self.size
        if pos == 0:
            self.data[0] = new_item
            self._siftup(0)
            return

        parent_pos = (pos - 1) >> 1

        if new_item == self.data[parent_pos]:
            self.data[pos] = new_item
        elif new_item < self.data[parent_pos]:
            self._siftdown(0, pos)
        else:
            child_pos = 2 * pos + 1
            if child_pos >= end_pos:
                self.data[pos] = new_item
                return
            right_pos = child_pos + 1
            if right_pos < end_pos and self.data[child_pos] >= self.data[right_pos]:
                child_pos = right_pos
            if new_item > self.data[child_pos]:
                self._siftup(pos)
            else:
                self.data[pos] = new_item

    def _siftup(self, pos):
        end_pos = len(self.data)
        start_pos = pos
        new_item = self.data[pos]
        child_pos = 2 * pos + 1
        while child_pos < end_pos:
            right_pos = child_pos + 1
            if right_pos < end_pos:
                if not self.data[child_pos] < self.data[right_pos]:
                    child_pos = right_pos

            self.data[pos] = self.data[child_pos]
            self.info[self.data[child_pos].key] = pos
            pos = child_pos
            child_pos = 2 * pos + 1
        self.data[pos] = new_item
        self.info[new_item.key] = pos
        self._siftdown(start_pos, pos)

    def _siftdown(self, start_pos, pos):
        new_item = self.data[pos]
        while pos > start_pos:
            parent_pos = (pos - 1) >> 1
            parent = self.data[parent_pos]
            if new_item < parent:
                self.data[pos] = parent

                self.info[parent.key] = pos
                pos = parent_pos
                continue
            break
        self.data[pos] = new_item
        self.info[new_item.key] = pos


if __name__ == '__main__':
    pass
