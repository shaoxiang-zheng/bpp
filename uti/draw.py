#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time : 2020/11/22 17:42
# Author: Zheng Shaoxiang
# @Email : zhengsx95@163.com
# Description:
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
from matplotlib.patches import Rectangle


def draw_2d_pattern(layouts, w, h):
    matplotlib.matplotlib_fname()

    def draw_layout(layout):
        plt.plot([0, w, w, 0, 0], [0, 0, h, h, 0], color='black')
        for rec in layout:
            rect = Rectangle((rec.x, rec.y), rec.width, rec.height, color='#666666')
            ax = plt.gca()
            ax.add_patch(rect)
            plt.plot([rec.x, rec.x + rec.width, rec.x + rec.width, rec.x, rec.x],
                     [rec.y, rec.y, rec.y + rec.height, rec.y + rec.height, rec.y], color='black')
            plt.text(rec.x + 0.1, rec.y + 0.1, str(rec.width) + 'X' + str(rec.height), fontsize=12)
        plt.axis('off')
        # plt.axis([0, w, 0, h])
        plt.show()
    if isinstance(layouts, list):
        draw_layout(layouts)
    elif isinstance(layouts, dict):
        for layout in layouts.values():
            draw_layout(layout)
    else:
        raise TypeError(f"The type {layouts.__class__.__name__} is not support!")


if __name__ == '__main__':
    pass
