import numpy as np

"""
0 : background
1 : cone
2 : granary
3 : bridge
4 : tractor
5 : corn
6 : pig
7 : crosswalk
8 : bump
"""


class ResultProcess:
    def __init__(self):
        self.memory = [[0 for i in range(9)]
                       for j in range(5)]  # 用于存储最近 5 帧的识别结果
        self.memory_index = 0  # 当前存储的帧数据的索引
        self.result = [0 for i in range(9)]  # 用于存储最终的识别结果

    def update(self, detection):
        self.memory_index = (self.memory_index + 1) % 5  # 更新索引
        self.result -= self.memory[self.memory_index]  # 将当前帧的识别结果从最终结果中减去
        self.memory[self.memory_index] = [0 for i in range(9)]  # 将当前帧的识别结果清零
        for det in detection:
            # 当前帧的识别结果保存到 memory 中
            self.memory[self.memory_index][det.type] += 1
        self.result += self.memory[self.memory_index]  # memory 同步至 result

    def getResult(self):
        for i in range(9):
            if self.result[i] > 3:  # 如果某个识别结果出现次数超过 3 次，则认为合理
                return i
        return -1
