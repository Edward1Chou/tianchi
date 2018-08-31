#!/usr/bin/env bash

declare -i n1
declare -i n2
declare -i n3
declare -i n4
declare -i n5
declare -i n0

m0=111545
m1=287
m2=744
m3=598
m4=53
m5=3397

n0=$m0/5
n1=$m1/5
n2=$m2/5
n3=$m3/5
n4=$m4/5
n5=$m5/5

echo $n0
echo $n1
echo $n2
echo $n3
echo $n4
echo $n5

tail -$n0 ../tmp/api_corpus_0.txt > ../tmp/api_0_eval.txt
tail -$n1 ../tmp/api_corpus_1.txt > ../tmp/api_1_eval.txt
tail -$n2 ../tmp/api_corpus_2.txt > ../tmp/api_2_eval.txt
tail -$n3 ../tmp/api_corpus_3.txt > ../tmp/api_3_eval.txt
tail -$n4 ../tmp/api_corpus_4.txt > ../tmp/api_4_eval.txt
tail -$n5 ../tmp/api_corpus_5.txt > ../tmp/api_5_eval.txt

n0=$m0-$m0/5
n1=$m1-$m1/5
n2=$m2-$m2/5
n3=$m3-$m3/5
n4=$m4-$m4/5
n5=$m5-$m5/5

echo $n0
echo $n1
echo $n2
echo $n3
echo $n4
echo $n5

head -$n0 ../tmp/api_corpus_0.txt >> ../tmp/api_train.txt
head -$n1 ../tmp/api_corpus_1.txt >> ../tmp/api_train.txt
head -$n2 ../tmp/api_corpus_2.txt >> ../tmp/api_train.txt
head -$n3 ../tmp/api_corpus_3.txt >> ../tmp/api_train.txt
head -$n4 ../tmp/api_corpus_4.txt >> ../tmp/api_train.txt
head -$n5 ../tmp/api_corpus_5.txt >> ../tmp/api_train.txt