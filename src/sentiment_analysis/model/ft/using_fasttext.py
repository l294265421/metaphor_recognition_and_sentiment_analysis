#!/usr/bin/env python
# coding=utf-8

import os
import logging

import fasttext

import src.original_data.data_path as data_path

logging.basicConfig(level=logging.INFO)

def __predict(model_path, test_data_path):
    classifier = fasttext.load_model(model_path, label_prefix='__label__')
    labels_right = []
    texts = []
    with open(test_data_path, encoding='utf-8') as fr:
        for line in fr:
            labels_right.append(line.split("\t")[-1].replace("__label__", "").strip())
            text = ' '.join(line.split("\t")[:-1])
            texts.append(text)
    # break
    y_pred = classifier.predict(texts)
    labels_predict = [e[0] for e in classifier.predict(texts)]  # 预测输出结果为二维形式

    text_labels = list(set(labels_right))
    text_predict_labels = list(set(labels_predict))

    A = dict.fromkeys(text_labels, 0)  # 预测正确的各个类的数目
    B = dict.fromkeys(text_labels, 0)  # 测试数据集中各个类的数目
    C = dict.fromkeys(text_predict_labels, 0)  # 预测结果中各个类的数目
    for i in range(0, len(labels_right)):
        B[labels_right[i]] += 1
        C[labels_predict[i]] += 1
        if labels_right[i] == labels_predict[i]:
            A[labels_right[i]] += 1

    # 计算准确率，召回率，F值
    for key in B:
        try:
            r = float(A[key]) / float(B[key])
            p = float(A[key]) / float(C[key])
            f = p * r * 2 / (p + r)
            logging.info("%s:\t p:%f\t r:%f\t f:%f" % (key, p, r, f))
        except:
             logging.error("error:", key, "right:", A.get(key, 0), "real:", B.get(key, 0), "predict:", C.get(key, 0))

if __name__ == "__main__":
    base_dir = data_path.metaphor_data_base_dir
    filename_train = 'sentiment_analysis.fasttext.train'
    filename_validation = 'sentiment_analysis.fasttext.validation'

    # base_dir = r'/home/liyuncong/program/fasttext/data/'
    # filename_train = 'news_fasttext_train.txt'
    # filename_validation = 'news_fasttext_test.txt'

    filename_model = 'sentiment_analysis.fasttext'

    train_data = os.path.join(base_dir, filename_train)
    valid_data = os.path.join(base_dir, filename_validation)

    # train_supervised uses the same arguments and defaults as the fastText cli
    model = fasttext.supervised(train_data, base_dir + filename_model)

    test_result = model.test(valid_data)
    print(test_result.precision)
    print(test_result.recall)
    print(test_result.nexamples)

    __predict(base_dir + filename_model + ".bin", valid_data)
