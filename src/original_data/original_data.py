# coding=utf-8

import logging
import xml.dom.minidom as minidom

import src.original_data.integer_label_mapping as integer_labal_mapping
import src.original_data.data_path as data_path


def __load_sentiment_analysis_data():
    base_dir = data_path.metaphor_data_base_dir
    in_filename = '隐喻情感_train.xml'
    dom_tree = minidom.parse(open(base_dir + in_filename, encoding='utf-8'))
    metaphors = dom_tree.documentElement
    metaphors = metaphors.getElementsByTagName('metaphor')
    result = []
    for metephor in metaphors:
        id = metephor.getElementsByTagName('ID')[0].childNodes[0].data
        sentence = metephor.getElementsByTagName('Sentence')[0].childNodes[0].data
        emo_class = metephor.getElementsByTagName('Emo_Class')[0].childNodes[0].data
        result.append([id, sentence, emo_class, integer_labal_mapping.integer_label_mapping_sentiment_analysis[emo_class]])
    return result


def __load_metaphor_recognition_data():
    base_dir = data_path.metaphor_data_base_dir
    in_filename = '隐喻动词_train.xml'
    dom_tree = minidom.parse(open(base_dir + in_filename, encoding='utf-8'))
    metaphors = dom_tree.documentElement
    metaphors = metaphors.getElementsByTagName('metaphor')
    result = []
    for metephor in metaphors:
        id = metephor.getElementsByTagName('ID')[0].childNodes[0].data
        sentence = metephor.getElementsByTagName('Sentence')[0].childNodes[0].data
        label = metephor.getElementsByTagName('Label')[0].childNodes[0].data
        result.append([id, sentence, label, integer_labal_mapping.integer_label_mapping_metaphor_recognition[label]])
    return result

# 元素：[id, sentence, class, lavel]
sentiment_analysis_data = __load_sentiment_analysis_data()

metaphor_recognition_data = __load_metaphor_recognition_data()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    for item in sentiment_analysis_data:
        logging.info(item)