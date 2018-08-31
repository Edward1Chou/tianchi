#!/usr/bin/env python
# encoding: utf-8
"""
0-正常/ 370157119 111545 3318.5
1-勒索病毒/ 1369564 287 4772
2-挖矿程序/ 5964169 744 8016.4
3-DDoS木马/ 6108264 598 10214.5
4-蠕虫病毒/ 343644 53 6483.8
5-感染型病毒 25688289 3397 7562.1
"""
a=0
b=0
c=0
d=0
e=0
f=0
with open("../data/train.csv","r") as file:
    for line in file.readlines():
        split_line=line.split(",")
        count=split_line[1]
        if count=="0":
            a+=1
        elif count=="1":
            b+=1
        elif count=="2":
            c+=1
        elif count=="3":
            d+=1
        elif count=="4":
            e+=1
        elif count=="5":
            f+=1

print(a)
print(b)
print(c)
print(d)
print(e)
print(f)

