# !/usr/bin/env python
# coding=utf-8

import collections

import src.original_data.integer_label_mapping as integer_label_mapping
import src.original_data.original_data as original_data
import src.original_data.data_path as data_path

if __name__ == '__main__':
    base_dir = data_path.metaphor_data_base_dir
    result = collections.defaultdict(int)
    for metephor in original_data.metaphor_recognition_data:
        label = metephor[2]
        result[integer_label_mapping.integer_label_mapping_metaphor_recognition[label]] += 1
    for k, v in result.items():
        print("key: %s value: %d" % (k, v))

