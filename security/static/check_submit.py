#!/usr/bin/env python
# encoding: utf-8

submit_file = "../submit/submit_augment.csv"
output = "../submit/submit_augment_clean.csv"

# with open(output, 'w') as o:
#     o.write("file_id,prob0,prob1,prob2,prob3,prob4,prob5\n")
#     with open(submit_file, 'r') as s:
#         for line in s.readlines()[1:]:
#             sp = line.strip('\n').split(',')
#             file_no = sp[0]
#             prob0 = float(sp[1])
#             prob1 = float(sp[2])
#             prob2 = float(sp[3])
#             prob3 = float(sp[4])
#             prob4 = float(sp[5])
#             prob5 = float(sp[6])
#             if prob0 + prob1 + prob2 + prob3 + prob4 + prob5 != 1.0:
#                 cha = 1.0 - (prob0 + prob1 + prob2 + prob3 + prob4 + prob5)
#                 prob0 += cha
#             new_line = ",".join([file_no, str(prob0), str(prob1), str(prob2),\
#                                  str(prob3), str(prob4), str(prob5)]) + "\n"
#             o.write(new_line)

with open(output, 'r') as o:
    for line in o.readlines()[1:]:
        sp = line.strip('\n').split(',')
        file_no = sp[0]
        prob0 = float(sp[1])
        prob1 = float(sp[2])
        prob2 = float(sp[3])
        prob3 = float(sp[4])
        prob4 = float(sp[5])
        prob5 = float(sp[6])
        if abs(1.0-(prob0 + prob1 + prob2 + prob3 + prob4 + prob5)) < 1.0e-6:
            continue
        else:
            print("error: " + file_no)