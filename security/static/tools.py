#!/usr/bin/env python
# encoding: utf-8
import sys
from sklearn.metrics import log_loss
import numpy as np
from collections import Counter
import json


def divide_dataset(raw, out_prefix):
    """划分数据集"""
    with open(out_prefix+"0.csv", 'w') as t0,open(out_prefix+"1.csv", 'w') as t1,open(out_prefix+"2.csv", 'w') as t2,\
            open(out_prefix + "3.csv", 'w') as t3,open(out_prefix+"4.csv", 'w') as t4,open(out_prefix+"5.csv", 'w') as t5:
        with open(raw, 'r') as r:
            for line in r.readlines()[1:]:
                label = line.split(',')[1]
                if label == "0":
                    t0.write(line)
                if label == "1":
                    t1.write(line)
                if label == "2":
                    t2.write(line)
                if label == "3":
                    t3.write(line)
                if label == "4":
                    t4.write(line)
                if label == "5":
                    t5.write(line)


def gen_corpus(label_file, corpus):
    """生成文本"""
    with open(corpus, 'w') as c:
        with open(label_file, 'r') as a:
            flag = a.readlines()[0].split(',')[0]
            # print(flag)
            document = ""
        with open(label_file, 'r') as b:
            tag = b.readlines()[-1]
        with open(label_file, 'r') as i:
            for line in i.readlines():
                # print(line + "tag")
                sp = line.split(',')
                label = sp[1]
                file_no = sp[0]
                # print(file_no)
                if flag != file_no or line == tag:
                    document += flag + " __label__" + label + "\n"
                    flag = file_no
                    c.write(document)
                    document = ""
                api = sp[2]
                ret_num = sp[4]
                # api_name = api + "_" + ret_num
                api_name = api
                document += api_name + " "


def gen_test_corpus(input_file, output):
    """生成待测试的文本文件"""
    with open(output, 'w') as o:
        with open(input_file, 'r') as a:
            flag = a.readlines()[1].split(',')[0] # file_no
            document = ""
        with open(input_file, 'r') as b:
            tag = b.readlines()[-1]
        with open(input_file, 'r') as i:
            for line in i.readlines()[1:]:
                sp = line.split(',')
                file_no = sp[0]
                api = sp[1]
                ret_num = sp[3]
                # api_name = api + "_" + ret_num
                api_name = api
                if flag != file_no:
                    document += flag + "\n"
                    flag = file_no
                    o.write(document)
                    document = ""
                if line == tag:
                    flag = file_no
                    document += api_name + " " + file_no + "\n"
                    o.write(document)
                    document = ""
                document += api_name + " "


def get_log_loss(proba_list, ground_list):
    """计算log loss"""
    prob_array = []
    for i in range(len(proba_list)):
        prob_dict = {}
        prob_list = []
        for j in range(6):
            prob_dict[proba_list[i][j][0]] = proba_list[i][j][1]
        prob_list.append(prob_dict["__label__0"])
        prob_list.append(prob_dict["__label__1"])
        prob_list.append(prob_dict["__label__2"])
        prob_list.append(prob_dict["__label__3"])
        prob_list.append(prob_dict["__label__4"])
        prob_list.append(prob_dict["__label__5"])
        prob_array.append(prob_list)
    clf_prob = np.array(prob_array)
    y_test = np.array(ground_list)
    score = log_loss(y_test, clf_prob)
    return score


def get_filter_log_loss(proba_list, ground_list):
    """计算log loss 将最大概率的赋为1，其余为0"""
    prob_array = []
    for i in range(len(proba_list)):
        prob_dict = {}
        prob_list = []
        max_value = -1
        max_index = -1
        for j in range(6):
            prob_dict[proba_list[i][j][0]] = proba_list[i][j][1]
            if proba_list[i][j][1] >= max_value:
                max_value = proba_list[i][j][1]
                max_index = proba_list[i][j][0]
        for j in range(6):
            if proba_list[i][j][0] == max_index:
                prob_dict[proba_list[i][j][0]] = 1
            else:
                prob_dict[proba_list[i][j][0]] = 0
        prob_list.append(prob_dict["__label__0"])
        prob_list.append(prob_dict["__label__1"])
        prob_list.append(prob_dict["__label__2"])
        prob_list.append(prob_dict["__label__3"])
        prob_list.append(prob_dict["__label__4"])
        prob_list.append(prob_dict["__label__5"])
        prob_array.append(prob_list)
    clf_prob = np.array(prob_array)
    y_test = np.array(ground_list)
    score = log_loss(y_test, clf_prob)
    return score


def gen_submit(proba_list, output):
    """生成提交文件"""
    # if len(proba_list) != 53093:
    #     print("error:not match list!!!")
    # else:
    with open(output, 'w') as o:
        o.write("file_id,prob0,prob1,prob2,prob3,prob4,prob5\n")
        for i in range(len(proba_list)):
            prob_dict = {}
            for j in range(6):
                prob_dict[proba_list[i][j][0]] = proba_list[i][j][1]
            line = str(i) + "," + str(prob_dict["__label__0"]) + "," + str(prob_dict["__label__1"]) + \
                "," + str(prob_dict["__label__2"]) + "," + str(prob_dict["__label__3"]) + "," + \
                str(prob_dict["__label__4"]) + "," + str(prob_dict["__label__5"]) + "\n"
            o.write(line)


# 将一句话转化为(uigram,bigram,trigram)后的字符串
def process_one_sentence_to_get_ui_bi_tri_gram(sentence,n_gram=3):
    """
    :param sentence: string. example:'w17314 w5521 w7729 w767 w10147 w111'
    :param n_gram:
    :return:string. example:'w17314 w17314w5521 w17314w5521w7729 w5521 w5521w7729 w5521w7729w767 w7729 w7729w767 w7729w767w10147 w767 w767w10147 w767w10147w111 w10147 w10147w111 w111'
    """
    result = []
    word_list = sentence.split(" ") #[sentence[i] for i in range(len(sentence))]
    unigram='';bigram='';trigram='';fourgram=''
    length_sentence=len(word_list)
    for i, word in enumerate(word_list):
        # unigram = word                           #ui-gram
        # word_i = unigram
        word_i = ''
        if n_gram >= 2 and i+2 <= length_sentence: #bi-gram
            bigram = "".join(word_list[i:i+2])
            word_i = bigram
        if n_gram >= 3 and i+3 <= length_sentence: #tri-gram
            trigram = "".join(word_list[i:i+3])
            word_i = word_i + ' ' + trigram
        if n_gram >= 4 and i+4 <= length_sentence: #four-gram
            fourgram = "".join(word_list[i:i+4])
            word_i = word_i + ' ' + fourgram
        if n_gram >= 5 and i+5 <= length_sentence: #five-gram
            fivegram = "".join(word_list[i:i+5])
            word_i = word_i + ' ' + fivegram
        result.append(word_i)
    result = " ".join(result)
    return result


def count_each_label_ngram(input_file, label_num, n_gram=5, topk=20):
    """计算每个标签ngram的排序"""
    label_dict = {}
    new_list = []
    with open(input_file, 'r') as i:
        for line in i.readlines():
            label = line.strip('\n')[-1]
            if label == str(label_num):
                sentence = line.strip('\n')[:-12]
                result = process_one_sentence_to_get_ui_bi_tri_gram(sentence, n_gram)
                tmp_list = result.split(" ")
                new_list.extend(tmp_list)
    label_dict = Counter(new_list)
    output = "../index/ngram_label" + str(label_num) + ".json"
    items = label_dict.items()
    backitems = [[v[1], v[0]] for v in items]
    backitems.sort(reverse=True)
    select_dict = dict([(backitems[i][1], label_dict[backitems[i][1]]) for i in range(0, len(backitems))][:topk])
    with open(output, 'w') as o:
        json.dump(select_dict, o, indent=1)
        o.write('\n')


def gen_ngram_feature(input_file, label_num, n_gram=5):
    """生成topk 2-5gram特征数量与ratio"""
    label_dict = {}
    # feature_list = get_topk_combine_ngram()
    # feature_list = ['LoadResourceFindResourceExWLoadResource', 'RegOpenKeyExARegQueryValueExA', 'RegQueryValueExWRegCloseKey', 'NtCreateSectionNtMapViewOfSection', 'RegCloseKeyRegQueryValueExW', 'RegCloseKeyRegOpenKeyExWRegQueryValueExWRegCloseKey', 'NtQuerySystemInformationNtQueryInformationFile', 'NtQueryInformationFileGetNativeSystemInfo', 'LdrLoadDllLdrGetProcedureAddressLdrLoadDll', 'NtCreateFileNtCreateSection', 'LdrGetDllHandleLdrGetProcedureAddress', 'RegEnumKeyExARegOpenKeyExA', 'NtReadFileSetFilePointerEx', 'GetNativeSystemInfoNtQuerySystemInformationNtQueryInformationFileGetNativeSystemInfoNtQuerySystemInformation', 'NtMapViewOfSectionNtClose', 'NtQueryInformationFileGetNativeSystemInfoNtQuerySystemInformation', 'NtReadFileNtWriteFileNtReadFileNtWriteFile', 'RegQueryValueExWRegCloseKeyRegOpenKeyExW', 'NtReadFileNtWriteFile', 'NtQueryDirectoryFileNtCloseFindFirstFileExWNtQueryDirectoryFile', 'NtCloseNtOpenKey', 'NtDelayExecutionNtOpenMutant', 'NtCreateFileNtCreateSectionNtMapViewOfSection', 'GetKeyStateGetSystemMetrics', 'RegQueryValueExARegCloseKey', 'NtQueryDirectoryFileNtClose', 'RegOpenKeyExWRegQueryValueExWRegCloseKeyRegOpenKeyExWRegQueryValueExW', 'FindFirstFileExWNtQueryDirectoryFileNtClose', 'RegOpenKeyExWRegCloseKey', 'LdrGetProcedureAddressLdrLoadDll', 'LdrLoadDllLdrGetProcedureAddressLdrLoadDllLdrGetProcedureAddress', 'NtCloseFindFirstFileExWNtQueryDirectoryFileNtClose', 'NtCloseNtDelayExecution', 'RegCloseKeyRegOpenKeyExWRegQueryValueExWRegCloseKeyRegOpenKeyExW', 'NtWriteFileNtReadFile', 'NtQueryDirectoryFileNtCloseFindFirstFileExW', 'NtQuerySystemInformationNtQueryInformationFileGetNativeSystemInfoNtQuerySystemInformationNtQueryInformationFile', 'NtQueryInformationFileGetNativeSystemInfoNtQuerySystemInformationNtQueryInformationFileGetNativeSystemInfo', 'RegQueryValueExWRegCloseKeyRegOpenKeyExWRegQueryValueExW', 'LdrLoadDllLdrGetProcedureAddress', 'NtWriteFileNtReadFileNtWriteFile', 'FindFirstFileExWNtQueryDirectoryFile', 'NtOpenKeyNtQueryValueKey', 'LdrGetProcedureAddressLdrLoadDllLdrGetProcedureAddress', 'FindFirstFileExWNtQueryDirectoryFileNtCloseFindFirstFileExWNtQueryDirectoryFile', 'NtReadFileNtWriteFileNtReadFile', 'NtQuerySystemInformationNtQueryInformationFileGetNativeSystemInfo', 'GetNativeSystemInfoNtQuerySystemInformationNtQueryInformationFile', 'NtCreateSectionNtMapViewOfSectionNtClose', 'RegQueryValueExWRegCloseKeyRegOpenKeyExWRegQueryValueExWRegCloseKey', 'NtOpenMutantNtClose', 'NtQueryInformationFileGetNativeSystemInfoNtQuerySystemInformationNtQueryInformationFile', 'NtQueryValueKeyNtClose', 'RegOpenKeyExWRegQueryValueExWRegCloseKeyRegOpenKeyExW', 'NtCreateFileNtCreateSectionNtMapViewOfSectionNtClose', 'NtOpenMutantNtCloseNtDelayExecution', 'FindResourceExWLoadResource', 'LoadResourceFindResourceExW', 'FindResourceExWLoadResourceFindResourceExW', 'NtCloseNtUnmapViewOfSection', 'NtWriteFileNtReadFileNtWriteFileNtReadFileNtWriteFile', 'NtOpenKeyNtQueryValueKeyNtClose', 'RegCloseKeyRegOpenKeyExW', 'RegOpenKeyExWRegQueryValueExW', 'RegCloseKeyRegOpenKeyExWRegQueryValueExW', 'SetFilePointerExNtReadFile', 'RegOpenKeyExARegQueryValueExARegCloseKey', 'NtWriteFileNtReadFileNtWriteFileNtReadFile', 'GetNativeSystemInfoNtQuerySystemInformation', 'SetFilePointerExNtWriteFile', 'SetFilePointerExNtReadFileSetFilePointerEx', 'NtQuerySystemInformationNtQueryInformationFileGetNativeSystemInfoNtQuerySystemInformation', 'NtCloseFindFirstFileExW', 'RegCloseKeyRegOpenKeyExA', 'NtDelayExecutionNtOpenMutantNtClose', 'NtCloseFindFirstFileExWNtQueryDirectoryFile', 'FindFirstFileExWNtQueryDirectoryFileNtCloseFindFirstFileExW', 'GetNativeSystemInfoNtQuerySystemInformationNtQueryInformationFileGetNativeSystemInfo', 'RegOpenKeyExWRegCloseKeyRegQueryValueExW', 'RegOpenKeyExWRegQueryValueExWRegCloseKey', 'RegQueryValueExWRegOpenKeyExW']
    feature_list = ['RegQueryInfoKeyARegEnumKeyExARegOpenKeyExARegQueryValueExA', 'RegOpenKeyExWRegCloseKeyRegOpenKeyExW', 'RegCloseKeyRegOpenKeyExARegQueryInfoKeyARegEnumKeyExA', 'FindResourceExWLoadResourceFindResourceExW', 'LdrGetDllHandleLdrGetProcedureAddressLdrGetDllHandle', 'NtQueryValueKeyNtCloseNtOpenKey', 'RegOpenKeyExWRegCloseKeyRegQueryValueExW', 'NtWriteFileNtReadFileNtWriteFile', 'RegOpenKeyExARegQueryValueExARegCloseKeyRegOpenKeyExARegQueryInfoKeyA', 'SetFilePointerExNtReadFileSetFilePointerEx', 'NtWriteFileNtReadFileNtWriteFileNtReadFileNtWriteFile', 'RegQueryValueExWRegCloseKeyRegOpenKeyExWRegQueryValueExWRegCloseKey', 'NtWriteFileSetFilePointerExNtReadFile', 'NtCloseNtUnmapViewOfSection', 'NtQueryInformationFileGetNativeSystemInfoNtQuerySystemInformation', 'NtQueryDirectoryFileNtCloseFindFirstFileExWNtQueryDirectoryFileNtClose', 'SetFilePointerExNtWriteFileSetFilePointerEx', 'RegCloseKeyRegOpenKeyExWRegQueryValueExWRegCloseKey', 'FindFirstFileExWNtQueryDirectoryFileNtCloseFindFirstFileExW', 'NtDelayExecutionNtOpenMutantNtClose', 'NtCloseNtDelayExecutionNtOpenMutantNtCloseNtDelayExecution', 'RegOpenKeyExARegQueryInfoKeyARegEnumKeyExARegOpenKeyExARegQueryValueExA', 'RegEnumKeyExARegOpenKeyExARegQueryValueExARegCloseKeyRegOpenKeyExA', 'LdrLoadDllLdrGetProcedureAddressNtWriteFile', 'SetFilePointerExNtWriteFile', 'FindFirstFileExWNtQueryDirectoryFile', 'NtCloseNtDelayExecution', 'RegCloseKeyRegOpenKeyExWRegQueryValueExW', 'NtQuerySystemInformationNtQueryInformationFileGetNativeSystemInfo', 'RegOpenKeyExARegQueryValueExA', 'RegEnumKeyExARegOpenKeyExARegQueryValueExARegCloseKey', 'NtCreateFileNtCreateSectionNtMapViewOfSection', 'NtCloseFindFirstFileExW', 'LdrLoadDllLdrGetProcedureAddressLdrLoadDllLdrGetProcedureAddressLdrLoadDll', 'RegQueryValueExWRegCloseKey', 'FindResourceExWLoadResourceFindResourceExWLoadResource', 'LdrGetProcedureAddressLdrLoadDll', 'NtQueryDirectoryFileNtClose', 'LdrGetProcedureAddressLdrLoadDllLdrGetProcedureAddress', 'SetFilePointerExNtReadFileSetFilePointerExNtWriteFile', 'RegCloseKeyRegOpenKeyExWRegQueryValueExWRegCloseKeyRegOpenKeyExW', 'NtOpenKeyNtQueryValueKey', 'NtQueryValueKeyNtClose', 'NtWriteFileSetFilePointerEx', 'GetNativeSystemInfoNtQuerySystemInformationNtQueryInformationFileGetNativeSystemInfoNtQuerySystemInformation', 'NtDelayExecutionaccept', 'NtWriteFileLdrLoadDll', 'NtReadFileSetFilePointerExNtWriteFileSetFilePointerExNtReadFile', 'RegEnumKeyExARegOpenKeyExA', 'NtWriteFileSetFilePointerExNtReadFileSetFilePointerEx', 'NtReadFileNtWriteFileNtReadFileNtWriteFileNtReadFile', 'NtCreateFileNtCreateSection', 'NtCloseFindFirstFileExWNtQueryDirectoryFileNtCloseFindFirstFileExW', 'NtOpenMutantNtClose', 'FindFirstFileExWNtQueryDirectoryFileNtClose', 'LdrGetProcedureAddressLdrLoadDllLdrGetProcedureAddressLdrLoadDllLdrGetProcedureAddress', 'RegOpenKeyExARegQueryInfoKeyARegEnumKeyExA', 'RegCloseKeyRegOpenKeyExWRegCloseKeyRegOpenKeyExW', 'NtDelayExecutionGetSystemTimeAsFileTime', 'NtQueryDirectoryFileNtCloseFindFirstFileExWNtQueryDirectoryFile', 'NtCloseFindFirstFileExWNtQueryDirectoryFile', 'GetNativeSystemInfoNtQuerySystemInformationNtQueryInformationFileGetNativeSystemInfo', 'NtQuerySystemInformationNtQueryInformationFile', 'GetSystemMetricsGetKeyState', 'SetFilePointerNtReadFile', 'RegCloseKeyRegOpenKeyExWRegCloseKey', 'NtWriteFileNtReadFile', 'NtCreateSectionNtMapViewOfSection', 'SetFilePointerExNtReadFileSetFilePointerExNtWriteFileSetFilePointerEx', 'GetKeyStateGetSystemMetrics', 'RegOpenKeyExARegQueryInfoKeyA', 'RegOpenKeyExWRegQueryValueExWRegOpenKeyExW', 'RegQueryInfoKeyARegEnumKeyExARegOpenKeyExA', 'FindFirstFileExWNtQueryDirectoryFileNtCloseFindFirstFileExWNtQueryDirectoryFile', 'NtDelayExecutionNtOpenMutantNtCloseNtDelayExecutionNtOpenMutant', 'NtReadFileSetFilePointerExNtWriteFileSetFilePointerEx', 'LdrGetDllHandleLdrGetProcedureAddress', 'NtQueryInformationFileGetNativeSystemInfo', 'NtQueryKeyNtOpenKeyEx', 'RegOpenKeyExWRegQueryValueExWRegCloseKey', 'RegQueryValueExWRegOpenKeyExWRegQueryValueExW', 'NtCloseNtOpenKeyNtQueryValueKey', 'RegOpenKeyExARegQueryValueExARegCloseKeyRegOpenKeyExA', 'NtCreateFileNtCreateSectionNtMapViewOfSectionNtClose', 'SetFilePointerExNtWriteFileSetFilePointerExNtReadFile', 'NtOpenMutantNtCloseNtDelayExecutionNtOpenMutant', 'NtQuerySystemInformationNtQueryInformationFileGetNativeSystemInfoNtQuerySystemInformation', 'GetKeyStateGetSystemMetricsGetKeyState', 'NtCloseFindFirstFileExWNtQueryDirectoryFileNtClose', 'RegOpenKeyExWRegQueryValueExW', 'LdrGetProcedureAddressLdrGetDllHandleLdrGetProcedureAddress', 'RegOpenKeyExWRegCloseKey', 'LdrLoadDllLdrGetProcedureAddressNtWriteFileLdrLoadDll', 'FindResourceExWLoadResource', 'NtQueryDirectoryFileNtCloseFindFirstFileExW', 'LoadResourceFindResourceExW', 'RegQueryValueExWRegOpenKeyExW', 'NtCloseNtDelayExecutionNtOpenMutant', 'RegCloseKeyRegQueryValueExW', 'RegCloseKeyRegOpenKeyExW', 'NtDelayExecutionNtOpenMutantNtCloseNtDelayExecution', 'GetNativeSystemInfoNtQuerySystemInformation', 'NtOpenKeyNtQueryValueKeyNtCloseNtOpenKey', 'LdrGetProcedureAddressLdrGetDllHandle', 'RegQueryValueExWRegCloseKeyRegOpenKeyExWRegQueryValueExW', 'LdrGetDllHandleLdrGetProcedureAddressLdrGetDllHandleLdrGetProcedureAddress', 'NtWriteFileNtReadFileNtWriteFileNtReadFile', 'SetFilePointerExNtWriteFileSetFilePointerExNtReadFileSetFilePointerEx', 'NtReadFileSetFilePointerEx', 'NtCreateSectionNtMapViewOfSectionNtClose', 'RegQueryInfoKeyARegEnumKeyExARegOpenKeyExARegQueryValueExARegCloseKey', 'NtCloseNtOpenKey', 'NtReadFileNtWriteFileNtReadFile', 'RegCloseKeyRegOpenKeyExA', 'LdrGetProcedureAddressNtWriteFileLdrLoadDll', 'NtQueryInformationFileGetNativeSystemInfoNtQuerySystemInformationNtQueryInformationFile', 'RegCloseKeyRegOpenKeyExARegQueryInfoKeyA', 'NtCloseNtOpenKeyNtQueryValueKeyNtClose', 'RegEnumKeyExARegOpenKeyExARegQueryValueExA', 'RegQueryValueExARegCloseKeyRegOpenKeyExARegQueryInfoKeyA', 'RegOpenKeyExWRegQueryValueExWRegCloseKeyRegOpenKeyExWRegQueryValueExW', 'LdrLoadDllLdrGetProcedureAddressLdrLoadDllLdrGetProcedureAddress', 'NtReadFileNtWriteFile', 'NtQuerySystemInformationNtQueryInformationFileGetNativeSystemInfoNtQuerySystemInformationNtQueryInformationFile', 'LdrLoadDllLdrGetProcedureAddress', 'LdrGetProcedureAddressLdrLoadDllLdrGetProcedureAddressLdrLoadDll', 'RegQueryValueExWRegCloseKeyRegOpenKeyExW', 'NtReadFileSetFilePointerExNtWriteFile', 'GetForegroundWindowGetKeyStateGetSystemMetrics', 'GetNativeSystemInfoNtQuerySystemInformationNtQueryInformationFile', 'GetForegroundWindowGetKeyState', 'RegQueryInfoKeyARegEnumKeyExA', 'RegOpenKeyExWRegQueryValueExWRegCloseKeyRegOpenKeyExW', 'NtReadFileNtWriteFileNtReadFileNtWriteFile', 'NtCloseNtDelayExecutionNtOpenMutantNtClose', 'RegQueryValueExARegCloseKey', 'SetFilePointerExNtReadFile', 'RegOpenKeyExARegQueryValueExARegCloseKey', 'RegOpenKeyExARegQueryInfoKeyARegEnumKeyExARegOpenKeyExA', 'NtOpenMutantNtCloseNtDelayExecutionNtOpenMutantNtClose', 'NtWriteFileSetFilePointerExNtReadFileSetFilePointerExNtWriteFile', 'LdrLoadDllLdrGetProcedureAddressLdrLoadDll', 'RegQueryValueExARegCloseKeyRegOpenKeyExA', 'NtOpenKeyNtQueryValueKeyNtClose', 'LoadResourceFindResourceExWLoadResource', 'NtQueryInformationFileGetNativeSystemInfoNtQuerySystemInformationNtQueryInformationFileGetNativeSystemInfo', 'LdrGetProcedureAddressNtWriteFile', 'NtDelayExecutionNtOpenMutant', 'NtMapViewOfSectionNtClose', 'NtOpenMutantNtCloseNtDelayExecution']
    fea_ratio_list = []
    for i in range(len(feature_list)):
        fea_ratio_list.append(feature_list[i] + "_ratio")
    output = "../index/ngram_feature_" + str(label_num) + ".csv"
    with open(output, 'w') as o:
        title = "file_no," + ",".join(feature_list) + "," + ",".join(fea_ratio_list) + "\n"
        o.write(title)
        with open(input_file, 'r') as i:
            for line in i.readlines():
                label = line.strip('\n')[-1]
                file_no = line.strip('\n')[:-11].split(" ")[-1]
                if label == str(label_num):
                    sentence = " ".join(line.strip('\n')[:-11].split(" ")[:-1])
                    result = process_one_sentence_to_get_ui_bi_tri_gram(sentence, n_gram)
                    tmp_list = result.split(" ")
                    tmp_dict = dict(Counter(tmp_list))
                    new_sen = file_no
                    total = 0
                    for j in list(tmp_dict.values()):
                        total += j
                    for ngram in feature_list:
                        new_sen += "," + str(tmp_dict.get(ngram, 0))
                    for ngram in feature_list:
                        new_sen += "," + str(tmp_dict.get(ngram, 0)/total)
                    o.write(new_sen + "\n")


def gen_test_ngram_fea(input_file, n_gram=5):
    """生成测试集ngram特征"""
    feature_list = ['LoadResourceFindResourceExWLoadResource', 'RegOpenKeyExARegQueryValueExA', 'RegQueryValueExWRegCloseKey', 'NtCreateSectionNtMapViewOfSection', 'RegCloseKeyRegQueryValueExW', 'RegCloseKeyRegOpenKeyExWRegQueryValueExWRegCloseKey', 'NtQuerySystemInformationNtQueryInformationFile', 'NtQueryInformationFileGetNativeSystemInfo', 'LdrLoadDllLdrGetProcedureAddressLdrLoadDll', 'NtCreateFileNtCreateSection', 'LdrGetDllHandleLdrGetProcedureAddress', 'RegEnumKeyExARegOpenKeyExA', 'NtReadFileSetFilePointerEx', 'GetNativeSystemInfoNtQuerySystemInformationNtQueryInformationFileGetNativeSystemInfoNtQuerySystemInformation', 'NtMapViewOfSectionNtClose', 'NtQueryInformationFileGetNativeSystemInfoNtQuerySystemInformation', 'NtReadFileNtWriteFileNtReadFileNtWriteFile', 'RegQueryValueExWRegCloseKeyRegOpenKeyExW', 'NtReadFileNtWriteFile', 'NtQueryDirectoryFileNtCloseFindFirstFileExWNtQueryDirectoryFile', 'NtCloseNtOpenKey', 'NtDelayExecutionNtOpenMutant', 'NtCreateFileNtCreateSectionNtMapViewOfSection', 'GetKeyStateGetSystemMetrics', 'RegQueryValueExARegCloseKey', 'NtQueryDirectoryFileNtClose', 'RegOpenKeyExWRegQueryValueExWRegCloseKeyRegOpenKeyExWRegQueryValueExW', 'FindFirstFileExWNtQueryDirectoryFileNtClose', 'RegOpenKeyExWRegCloseKey', 'LdrGetProcedureAddressLdrLoadDll', 'LdrLoadDllLdrGetProcedureAddressLdrLoadDllLdrGetProcedureAddress', 'NtCloseFindFirstFileExWNtQueryDirectoryFileNtClose', 'NtCloseNtDelayExecution', 'RegCloseKeyRegOpenKeyExWRegQueryValueExWRegCloseKeyRegOpenKeyExW', 'NtWriteFileNtReadFile', 'NtQueryDirectoryFileNtCloseFindFirstFileExW', 'NtQuerySystemInformationNtQueryInformationFileGetNativeSystemInfoNtQuerySystemInformationNtQueryInformationFile', 'NtQueryInformationFileGetNativeSystemInfoNtQuerySystemInformationNtQueryInformationFileGetNativeSystemInfo', 'RegQueryValueExWRegCloseKeyRegOpenKeyExWRegQueryValueExW', 'LdrLoadDllLdrGetProcedureAddress', 'NtWriteFileNtReadFileNtWriteFile', 'FindFirstFileExWNtQueryDirectoryFile', 'NtOpenKeyNtQueryValueKey', 'LdrGetProcedureAddressLdrLoadDllLdrGetProcedureAddress', 'FindFirstFileExWNtQueryDirectoryFileNtCloseFindFirstFileExWNtQueryDirectoryFile', 'NtReadFileNtWriteFileNtReadFile', 'NtQuerySystemInformationNtQueryInformationFileGetNativeSystemInfo', 'GetNativeSystemInfoNtQuerySystemInformationNtQueryInformationFile', 'NtCreateSectionNtMapViewOfSectionNtClose', 'RegQueryValueExWRegCloseKeyRegOpenKeyExWRegQueryValueExWRegCloseKey', 'NtOpenMutantNtClose', 'NtQueryInformationFileGetNativeSystemInfoNtQuerySystemInformationNtQueryInformationFile', 'NtQueryValueKeyNtClose', 'RegOpenKeyExWRegQueryValueExWRegCloseKeyRegOpenKeyExW', 'NtCreateFileNtCreateSectionNtMapViewOfSectionNtClose', 'NtOpenMutantNtCloseNtDelayExecution', 'FindResourceExWLoadResource', 'LoadResourceFindResourceExW', 'FindResourceExWLoadResourceFindResourceExW', 'NtCloseNtUnmapViewOfSection', 'NtWriteFileNtReadFileNtWriteFileNtReadFileNtWriteFile', 'NtOpenKeyNtQueryValueKeyNtClose', 'RegCloseKeyRegOpenKeyExW', 'RegOpenKeyExWRegQueryValueExW', 'RegCloseKeyRegOpenKeyExWRegQueryValueExW', 'SetFilePointerExNtReadFile', 'RegOpenKeyExARegQueryValueExARegCloseKey', 'NtWriteFileNtReadFileNtWriteFileNtReadFile', 'GetNativeSystemInfoNtQuerySystemInformation', 'SetFilePointerExNtWriteFile', 'SetFilePointerExNtReadFileSetFilePointerEx', 'NtQuerySystemInformationNtQueryInformationFileGetNativeSystemInfoNtQuerySystemInformation', 'NtCloseFindFirstFileExW', 'RegCloseKeyRegOpenKeyExA', 'NtDelayExecutionNtOpenMutantNtClose', 'NtCloseFindFirstFileExWNtQueryDirectoryFile', 'FindFirstFileExWNtQueryDirectoryFileNtCloseFindFirstFileExW', 'GetNativeSystemInfoNtQuerySystemInformationNtQueryInformationFileGetNativeSystemInfo', 'RegOpenKeyExWRegCloseKeyRegQueryValueExW', 'RegOpenKeyExWRegQueryValueExWRegCloseKey', 'RegQueryValueExWRegOpenKeyExW']
    fea_ratio_list = []
    for i in range(len(feature_list)):
        fea_ratio_list.append(feature_list[i] + "_ratio")
    output = "../index/ngram_feature_test.csv"
    with open(output, 'w') as o:
        title = "file_no," + ",".join(feature_list) + "," + ",".join(fea_ratio_list) + "\n"
        o.write(title)
        with open(input_file, 'r') as i:
            for line in i.readlines():
                file_no = line.strip('\n').split(" ")[-1]
                sentence = " ".join(line.strip('\n').split(" ")[:-1])
                result = process_one_sentence_to_get_ui_bi_tri_gram(sentence, n_gram)
                tmp_list = result.split(" ")
                tmp_dict = dict(Counter(tmp_list))
                new_sen = file_no
                total = 0
                for j in list(tmp_dict.values()):
                    total += j
                for ngram in feature_list:
                    new_sen += "," + str(tmp_dict.get(ngram, 0))
                for ngram in feature_list:
                    new_sen += "," + str(tmp_dict.get(ngram, 0)/total)
                o.write(new_sen + "\n")



def get_topk_combine_ngram():
    path00 = "../index/ngram_label00.json"
    path0 = "../index/ngram_label0.json"
    path1 = "../index/ngram_label1.json"
    path2 = "../index/ngram_label2.json"
    path3 = "../index/ngram_label3.json"
    path4 = "../index/ngram_label4.json"
    path5 = "../index/ngram_label5.json"
    with open(path00, 'r') as p00:
        dict00 = json.load(p00)
    with open(path0, 'r') as p0:
        dict0 = json.load(p0)
    with open(path1, 'r') as p1:
        dict1 = json.load(p1)
    with open(path2, 'r') as p2:
        dict2 = json.load(p2)
    with open(path3, 'r') as p3:
        dict3 = json.load(p3)
    with open(path4, 'r') as p4:
        dict4 = json.load(p4)
    with open(path5, 'r') as p5:
        dict5 = json.load(p5)
    key_list = []
    key_list.extend(list(dict00.keys()))
    key_list.extend(list(dict0.keys()))
    key_list.extend(list(dict1.keys()))
    key_list.extend(list(dict2.keys()))
    key_list.extend(list(dict3.keys()))
    key_list.extend(list(dict4.keys()))
    key_list.extend(list(dict5.keys()))
    combine_list = list(set(key_list))
    print("len:" + str(len(combine_list))) # 81
    print(combine_list)


if __name__ == "__main__":
    # raw = "../data/train_small.csv"
    # out_prefix = "../tmp/label"
    # divide_dataset(raw, out_prefix)
    #
    # label = sys.argv[1]
    # label_file = "../tmp/label" + str(label) + ".csv"
    # corpus = "../tmp/api_corpus_" + str(label) + ".txt"
    # # label_file = "../tmp/label5.csv"
    # # corpus = "../tmp/corpus_5.txt"
    # gen_corpus(label_file, corpus)

    # input_file = "../data/test_small.csv"
    # output = "../tmp/api_test.txt"
    # gen_test_corpus(input_file, output)

    # input_file = "../tmp/simple_train.txt"
    # count_each_label_ngram(input_file, 5)

    get_topk_combine_ngram()

    # input_file = "../tmp/simple_corpus_0.txt"
    # label = 0
    # gen_ngram_feature(input_file, label)

    # input_file = "../tmp/simple_test.txt"
    # gen_test_ngram_fea(input_file)