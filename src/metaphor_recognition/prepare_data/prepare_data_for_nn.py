# !/usr/bin/env python
# coding=utf-8

import logging

import jieba

import src.common.file_utils as file_utils
import src.original_data.original_data as data
import src.original_data.data_path as data_path

if __name__ == '__main__':
    base_dir = data_path.metaphor_data_base_dir
    in_filename = '隐喻动词_train.xml'
    out_filename_train = 'metaphor_recognition.nn.train'
    out_filename_validation = 'metaphor_recognition.nn.validation'
    result = []
    for metephor in data.metaphor_recognition_data:
        label = metephor[2]
        sentence = metephor[1]
        words = [word for word in jieba.cut(sentence)]
        result.append(label + ' ' + ' '.join(words))
    split_index = int(len(result) / 10 * 7)
    file_utils.write_lines(result[:split_index], base_dir + out_filename_train)
    file_utils.write_lines(result[split_index:], base_dir + out_filename_validation)




