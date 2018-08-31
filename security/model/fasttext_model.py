#!/usr/bin/env python
# encoding: utf-8
import fasttext
from sys import path
path.append("../static")
import tools

def train(train_corpus="../tmp/train.txt", model_save_path="demo_clf"):
    # 训练保存模型
    clf = fasttext.supervised(train_corpus, model_save_path,
                              epoch=120, word_ngrams=3, bucket=8000,
                              dim=500, loss="ns", lr=0.05)


def evaluate(model_path="demo_clf.bin"):
    # 加载模型
    model_clf = fasttext.load_model(model_path)

    # evaluate
    result0 = model_clf.test("../tmp/0_eval.txt")
    print("label_0 P@1: " + str(result0.precision))
    print("label_0 R@1: " + str(result0.recall))
    print("")
    result1 = model_clf.test("../tmp/1_eval.txt")
    print("label_1 P@1: " + str(result1.precision))
    print("label_1 R@1: " + str(result1.recall))
    print("")
    result2 = model_clf.test("../tmp/2_eval.txt")
    print("label_2 P@1: " + str(result2.precision))
    print("label_2 R@1: " + str(result2.recall))
    print("")
    result3 = model_clf.test("../tmp/3_eval.txt")
    print("label_3 P@1: " + str(result3.precision))
    print("label_3 R@1: " + str(result3.recall))
    print("")
    result4 = model_clf.test("../tmp/4_eval.txt")
    print("label_4 P@1: " + str(result4.precision))
    print("label_4 R@1: " + str(result4.recall))
    print("")
    result5 = model_clf.test("../tmp/5_eval.txt")
    print("label_5 P@1: " + str(result5.precision))
    print("label_5 R@1: " + str(result5.recall))
    print("")


def log_loss(eval_file="../tmp/eval.txt", model_path="demo_clf.bin"):
    """计算log_loss"""
    model_clf = fasttext.load_model(model_path)
    test_corpus = []
    ground_list = []
    with open(eval_file, 'r') as e:
        for line in e.readlines():
            test_corpus.append(line.strip('\n')[:-10])
            ground_label = int(line.strip('\n')[-1])
            ground_list.append(ground_label)
    labels_prob = model_clf.predict_proba(test_corpus, 6)
    log_loss_score = tools.get_log_loss(labels_prob, ground_list)
    # log_loss_score = tools.get_filter_log_loss(labels_prob, ground_list)
    return log_loss_score


def predict(test_file, model_path="demo_clf.bin"):
    # 加载模型
    model_clf = fasttext.load_model(model_path)
    test_corpus = []
    with open(test_file, 'r') as t:
        for line in t.readlines():
            test_corpus.append(line.strip('\n'))
    labels = model_clf.predict(test_corpus)
    # print(labels)
    labels_prob = model_clf.predict_proba(test_corpus, 6)
    return labels_prob


if __name__ == "__main__":
    train(train_corpus="../tmp/new_2_train.txt", model_save_path="ngram3_120epoch_0.05lr_new2")
    evaluate(model_path="ngram3_120epoch_0.05lr_new2.bin")

    score = log_loss(model_path="ngram3_120epoch_0.05lr_new2.bin")
    print("log loss: " + str(score))

    # labels_prob = predict("../tmp/test_small.txt")
    # tools.gen_submit(labels_prob, "../submit/test_submit.csv")
