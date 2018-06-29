# !/usr/bin/env python
# coding=utf-8

import logging

import jieba

import src.common.file_utils as file_utils
import src.original_data.original_data as data
import src.original_data.data_path as data_path

if __name__ == '__main__':
    base_dir = data_path.metaphor_data_base_dir
    in_filename = '隐喻情感_train.xml'
    out_filename_train = 'sentiment_analysis.fasttext.train'
    out_filename_validation = 'sentiment_analysis.fasttext.validation'
    label_prefix = '__label__'
    result = []
    for metephor in data.sentiment_analysis_data:
        emo_class = metephor[2]
        sentence = metephor[1]
        words = [word for word in jieba.cut(sentence)]
        logging.warning('emo_class: %s sentence: %s' %(label_prefix + emo_class, ' '.join(words)))
        result.append('\t'.join(words) + '\t' + label_prefix + emo_class )
    split_index = int(len(result) / 10 * 7)
    file_utils.write_lines(result[:split_index], base_dir + out_filename_train)
    file_utils.write_lines(result[split_index:], base_dir + out_filename_validation)




