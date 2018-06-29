import numpy as np

if __name__ == '__main__':
    base_dir = r'D:\公司\中文隐喻识别与情感分析\data\sgns.baidubaike.bigram-char\\'
    with open(base_dir + 'sgns.baidubaike.bigram-char', encoding='utf-8') as in_file, open(base_dir + 'sgns.baidubaike.bigram-char.correct', encoding='utf-8', mode='w') as out_file:
        for line in in_file:
            elements = line.strip().split()
            vector = elements[-300:]
            word = elements[:-300]
            new_line = ''.join(word) + ' ' + ' '.join(vector) + '\n'
            out_file.write(new_line)