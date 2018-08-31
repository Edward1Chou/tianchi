#!/usr/bin/env python
# encoding: utf-8
import os
import json
import itertools


def api_index(raw, output):
    api_list = []
    with open(raw, 'r') as r:
        for line in r.readlines()[1:]:
            sp = line.split(',')
            api = sp[2]
            api_list.append(api)
    api_dis = list(set(api_list))
    with open(output, 'w') as o:
        for i, api_name in enumerate(api_dis):
            o.write(api_name + '\t' + str(i) + '\n')


def each_class_own_api(raw, output):
    """
    每个类调用哪些api，落地成json文件
    """
    e_class_api_dict = {}
    with open(raw, 'r') as r:
        for line in r.readlines()[1:]:
            sp = line.split(',')
            label = sp[1]
            api = sp[2]
            e_class_api_dict.setdefault(label, []).append(api)

    # 去重
    for key in e_class_api_dict.keys():
        e_class_api_dict[key] = list(set(e_class_api_dict[key]))
    with open(output, 'w') as o:
        json.dump(e_class_api_dict, o, indent=1)
        o.write('\n')


def static_spec_api_for_class(api_class_json, output):
    """统计每类中独有的api"""
    ret_dict = {}
    with open(api_class_json, 'r') as a:
        e_class_api_dict = json.load(a)
    for key in e_class_api_dict.keys():
        combine_apis = []
        o_keys = list(e_class_api_dict.keys())
        o_keys.remove(key)
        for o_key in o_keys:
            combine_apis.extend(e_class_api_dict[o_key])
        class_apis = e_class_api_dict[key]
        intersect = list(set(class_apis).union(set(combine_apis))^(set(class_apis)^set(combine_apis)))
        spec = [x for x in class_apis if x not in intersect]
        ret_dict[key] = spec
    with open(output, 'w') as o:
        json.dump(ret_dict, o, indent=1)
        o.write('\n')


def call_api_num(raw):
    """
    每类文件调用api的最大值，最小值，平均值
    """
    pass


def e_class_file_num(raw, output):
    """各类训练文件数，落地json文件"""
    tmp_list = []
    ret_dict = {}
    mirror_list = ['0', '1', '2', '3', '4', '5']
    with open(raw, 'r') as r:
        for line in r.readlines()[1:]:
            sp = line.split(',')
            file_no = sp[0]
            label = sp[1]
            flag = file_no + "_" + label
            if flag in tmp_list:
                continue
            else:
                tmp_list.append(flag)
    for item in mirror_list:
        ret_dict[item] = 0
        for tuple_m in tmp_list:
            sp_item = tuple_m.split("_")
            if sp_item[1] == item:
                ret_dict[item] += 1
    with open(output, 'w') as o:
        json.dump(ret_dict, o, indent=1)
        o.write('\n')


def mean_call_api(raw, output, file_num):
    """某类调用各api平均数"""
    tmp_dict = {}
    ret_dict = {}
    with open(raw, 'r') as r:
        for line in r.readlines()[1:]:
            if line == "":
                break
            sp = line.split(',')
            label = sp[1]
            api = sp[2]
            if api+"^"+label in tmp_dict.keys():
                tmp_dict[api+"^"+label] += 1
            else:
                tmp_dict[api + "^" + label] = 1
    # print(tmp_dict)
    with open(file_num, 'r') as f:
        file_num_dict = json.load(f)
    print(tmp_dict)
    api_num_0 = {}; api_num_1 = {}; api_num_2 = {}
    api_num_3 = {}; api_num_4 = {}; api_num_5 = {}
    for key in tmp_dict.keys():
        k_label = key.split('^')[1]
        # print("label:" + k_label)
        k_file_num = file_num_dict[k_label]
        if k_file_num != 0:
            tmp_dict[key] = tmp_dict[key]/k_file_num
        if k_label == "0":
            api_num_0[key.split('^')[0]] = tmp_dict[key]
        if k_label == "1":
            api_num_1[key.split('^')[0]] = tmp_dict[key]
        if k_label == "2":
            api_num_2[key.split('^')[0]] = tmp_dict[key]
        if k_label == "3":
            api_num_3[key.split('^')[0]] = tmp_dict[key]
        if k_label == "4":
            api_num_4[key.split('^')[0]] = tmp_dict[key]
        if k_label == "5":
            api_num_5[key.split('^')[0]] = tmp_dict[key]
        # api_num = {key.split('^')[0]: tmp_dict[key]}
        # ret_dict.setdefault(k_label, []).append(api_num)
    ret_dict_0 = dict(sorted(api_num_0.items(), key=lambda x: x[1], reverse=True))
    ret_dict_1 = dict(sorted(api_num_1.items(), key=lambda x: x[1], reverse=True))
    ret_dict_2 = dict(sorted(api_num_2.items(), key=lambda x: x[1], reverse=True))
    ret_dict_3 = dict(sorted(api_num_3.items(), key=lambda x: x[1], reverse=True))
    ret_dict_4 = dict(sorted(api_num_4.items(), key=lambda x: x[1], reverse=True))
    ret_dict_5 = dict(sorted(api_num_5.items(), key=lambda x: x[1], reverse=True))
    ret_dict = {"0":ret_dict_0, "1":ret_dict_1, "2":ret_dict_2, "3":ret_dict_3,
                "4":ret_dict_4, "5":ret_dict_5}
    with open(output, 'w') as o:
        json.dump(ret_dict, o, indent=1)
        o.write('\n')


def continus_api(raw, output):
    """某类调用各api最大连续数"""
    pass


def all_label0_submit(output):
    """所有文件label标记1"""
    with open(output, 'w') as o:
        o.write("file_id,prob0,prob1,prob2,prob3,prob4,prob5\n")
        for i in range(53093):
            o.write(str(i)+",1,0,0,0,0,0\n")


def data_augment_1(input_file, output):
    """增加训练数据量少的label
    1.逆序生成相应的文本
    2.单词拆分，生成新的训练数据
    3.单词连续数大于10，减少一半，生成新数据
    """
    with open(output, 'w') as o:
        with open(input_file, 'r') as i:
            for line in i.readlines():
                label = line.strip('\n')[-1]
                if label == "0":
                    o.write(line)
                else:
                    o.write(line)
                    sentence = line.strip('\n')[:-10].split(" ")
                    sentence.reverse()
                    new_sen = " ".join(sentence) + " __label__" + label + "\n"
                    o.write(new_sen)


def data_augment_2():
    """单词拆分"""
    pass


def data_augment_3(input_file, output, threshold=10):
    """单词连续数大于10，减少一半，生成新数据"""
    with open(output, 'w') as o:
        with open(input_file, 'r') as i:
            for line in i.readlines():
                label = line.strip('\n')[-1]
                if label == "0":
                    o.write(line)
                else:
                    o.write(line)
                    sentence = line.strip('\n')[:-10].split(" ")
                    word_num_list = [len(list(v)) for k, v in itertools.groupby(sentence)]
                    single_word_list = [list(v)[0] for k, v in itertools.groupby(sentence)]
                    new_sen_list = []
                    for i, j in zip(word_num_list, single_word_list):
                        # print(i, j)
                        if i > threshold:
                            new_sen_list.extend([j] * int(i / 2))
                            # new_sen_list.extend([j])
                        else:
                            new_sen_list.extend([j] * i)
                    new_sen = " ".join(new_sen_list) + " __label__" + label + "\n"
                    o.write(new_sen)


def data_augment_4(input_file, output, threshold=10):
    """生成ngram特征统计目标文件"""
    with open(output, 'w') as o:
        with open(input_file, 'r') as i:
            for line in i.readlines():
                # label = line.strip('\n')[-1]
                # file_no = line.strip('\n')[:-11].split(" ")[-1]
                # sentence = line.strip('\n')[:-11].split(" ")[:-1]
                file_no = line.strip('\n').split(" ")[-1]
                sentence = line.strip('\n').split(" ")[:-1]
                word_num_list = [len(list(v)) for k, v in itertools.groupby(sentence)]
                single_word_list = [list(v)[0] for k, v in itertools.groupby(sentence)]
                new_sen_list = []
                for i, j in zip(word_num_list, single_word_list):
                    # print(i, j)
                    if i > threshold:
                        # new_sen_list.extend([j] * int(i / 2))
                        new_sen_list.extend([j])
                    else:
                        new_sen_list.extend([j] * i)
                new_sen = " ".join(new_sen_list) + " " + file_no + "\n"
                o.write(new_sen)


def data_cluster():
    """减少训练数据量特别多的label0数量
    通过聚类，异常值分析，训练的时候排除掉这些文本
    """
    pass


if __name__ == "__main__":
    # output = "../index/class_own_api.json"
    # raw = "../data/train.csv"
    # each_class_own_api(raw, output)

    # json_file = "../index/class_own_api.json"
    # output = "../index/class_spec_api.json"
    # static_spec_api_for_class(json_file, output)

    # raw = "../data/train.csv"
    # output = "../index/file_num.json"
    # e_class_file_num(raw, output)

    # raw = "../data/train.csv"
    # output = "../index/label_api_num.json"
    # file_num = "../index/file_num.json"
    # mean_call_api(raw, output, file_num)

    # output = "../submit/all_label0.csv"
    # all_label0_submit(output)

    # input_file = "../tmp/train.txt"
    # output = "../tmp/new_train.txt"
    # data_augment_1(input_file, output)

    input_file = "../tmp/api_test.txt"
    output = "../tmp/simple_test.txt"
    data_augment_4(input_file, output, threshold=1)






