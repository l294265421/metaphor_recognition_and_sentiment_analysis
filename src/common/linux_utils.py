#!/usr/bin/env python
# coding=utf-8
def head(file_path, n):
    result = []
    with open(file_path, encoding='utf-8') as in_file:
        count = 0
        for line in in_file:
            if count < n:
                result.append(line)
                count += 1
            else:
                break
    return result

import numpy as np
def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

if __name__ == '__main__':
    lines = head(r'D:\公司\中文隐喻识别与情感分析\data\sgns.baidubaike.bigram-char\sgns.baidubaike.bigram-char', 5000000)

    for line in lines:
        try:
            get_coefs(*line.split())
        except:
            print(line)
