#!/usr/bin/env python
# encoding: utf-8
import os
import pickle
import codecs
from collections import Counter
import random
import numpy as np
from tflearn.data_utils import pad_sequences


PAD_ID = 0
UNK_ID = 1
_PAD = "_PAD"
_UNK = "UNK"


def create_vocabulary(training_data_path, vocab_size, name_scope='cnn'):
    """create vocabulary"""
    cache_vocabulary_label_pik = '../checkpoint/cache'+"_"+name_scope
    if not os.path.isdir(cache_vocabulary_label_pik):
        os.makedirs(cache_vocabulary_label_pik)

    # if cache exists. load it; otherwise create it.
    cache_path = cache_vocabulary_label_pik+"/"+'vocab_label.pik'
    print("cache_path:", cache_path, "file_exists:", os.path.exists(cache_path))
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as data_f:
            return pickle.load(data_f)
    else:
        vocabulary_word2index = {}
        vocabulary_index2word = {}
        vocabulary_word2index[_PAD] = PAD_ID
        vocabulary_index2word[PAD_ID] = _PAD
        vocabulary_word2index[_UNK] = UNK_ID
        vocabulary_index2word[UNK_ID] = _UNK

        vocabulary_label2index = {}
        vocabulary_index2label = {}

        #1.load raw data
        file_object = codecs.open(training_data_path, mode='r', encoding='utf-8')
        lines = file_object.readlines()
        #2.loop each line,put to counter
        c_inputs = Counter()
        c_labels = Counter()
        for line in lines:
            raw_list = line.strip().split("__label__")

            input_list = raw_list[0].strip().split(" ")
            input_list = [x.strip().replace(" ", "") for x in input_list if x != '']
            label_list = [l.strip().replace(" ", "") for l in raw_list[1:] if l!='']
            c_inputs.update(input_list)
            c_labels.update(label_list)
        #return most frequency words
        vocab_list=c_inputs.most_common(vocab_size)
        label_list=c_labels.most_common()
        #put those words to dict
        for i,tuplee in enumerate(vocab_list):
            word,_ = tuplee
            vocabulary_word2index[word] = i+2 # 前两个是PAD和UNK
            vocabulary_index2word[i+2] = word

        for i,tuplee in enumerate(label_list):
            label,_ = tuplee
            label = str(label)
            vocabulary_label2index[label] = i
            vocabulary_index2label[i] = label

        #save to file system if vocabulary of words not exists.
        if not os.path.exists(cache_path):
            with open(cache_path, 'ab') as data_f: # 二进制追加模式
                pickle.dump((vocabulary_word2index,vocabulary_index2word,vocabulary_label2index,vocabulary_index2label), data_f)
    return vocabulary_word2index,vocabulary_index2word,vocabulary_label2index,vocabulary_index2label


def load_data_multilabel(traning_data_path, vocab_word2index, vocab_label2index, training_portion=0.95):
    """
    convert data as indexes using word2index dicts.
    """
    file_object = codecs.open(traning_data_path, mode='r', encoding='utf-8')
    lines = file_object.readlines()
    random.shuffle(lines)
    label_size = len(vocab_label2index)
    X = []
    Y = []
    max_senlen = 0 # max_senlen最终还需要传给TextCNN
    for i,line in enumerate(lines):
        raw_list = line.strip().split("__label__")
        input_list = raw_list[0].strip().split(" ")
        input_list = [x.strip().replace(" ", "") for x in input_list if x != '']
        x = [vocab_word2index.get(x, UNK_ID) for x in input_list]
        if len(x) > max_senlen:
            max_senlen = len(x)
        label_list = raw_list[1:]
        label_list = [l.strip().replace(" ", "") for l in label_list if l != '']
        label_list = [vocab_label2index[label] for label in label_list]
        y = transform_multilabel_as_multihot(label_list, label_size)
        X.append(x)
        Y.append(y)
    X = pad_sequences(X, maxlen=max_senlen+1, value=0.)  # padding to max length 可能会发生截断
    number_examples = len(lines)
    training_number = int(training_portion * number_examples)
    train = (X[0:training_number], Y[0:training_number])
    valid_number = min(1000, number_examples-training_number) # 可以稍微减少样本不均衡的问题
    test = (X[training_number + 1:training_number+valid_number+1], Y[training_number + 1:training_number+valid_number+1])
    return train, test, max_senlen


def transform_multilabel_as_multihot(label_list,label_size):
    """
    convert to multi-hot style
    :param label_list: e.g.[0,1,4], here 4 means in the 4th position it is true value(as indicate by'1')
    :param label_size: e.g.199
    :return:e.g.[1,1,0,1,0,0,........]
    """
    result = np.zeros(label_size)
    #set those location as 1, all else place as 0.
    result[label_list] = 1
    return result