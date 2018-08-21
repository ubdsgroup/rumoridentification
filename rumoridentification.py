import matplotlib
matplotlib.use('Agg')
import numpy as np
import math
import random as rnd
import time
from sklearn import metrics
import matplotlib.pyplot as plt

def uniq_vocab(data):
    uniq = set()
    for i in data:
        for j in i:
            uniq.add(j)
    return list(uniq)

def stat_prob(M, node, eps):
    n = M.shape[0]
    k = M.shape[1]
    c = 1/n
    mu = np.array([rnd.random() for i in range(n+k)])
    q_list = [0]*(n+k)
    q_list[node] = 1
    q = np.array(q_list)
    diff_mu = mu
    Cm = col_norm(M)
    Cmt = col_norm(np.transpose(M))
    while np.linalg.norm(diff_mu, ord=1) > eps:
        pre = np.dot(Cm, mu[n:n+k])
        suf = np.dot(Cmt,mu[:n])

        pre_mu = np.concatenate([pre, suf])
        new_mu = (1-c)*pre_mu + c*q

        diff_mu = new_mu - mu
        mu = new_mu
    return mu[:n]

def adjacency_matrix(M):
    n = M.shape[0]
    k = M.shape[1]
    zero1 = np.zeros((n,n))
    zero2 = np.zeros((k,k))
    Ma1 = np.concatenate((zero1, M),axis=1)
    Ma2 = np.concatenate((np.transpose(M), zero2), axis=1)
    Ma = np.concatenate((Ma1, Ma2), axis=0)
    return Ma

def col_norm(Ma):
    Ma_sum = Ma.sum(axis=0)
    P = np.divide(Ma,Ma_sum)
    return P

file_1 = open('clean_data_en_with_extra_stemming.txt')

uniq_final_data1 = []
mat = []
label = []
index = 0
"""<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"""
"""Label the tweets here. If the 'cellphone' is in the tweet label that tweet 1 else 0"""
for i in file_1:
    uniq_final_data1.append((i.rstrip('\n')).split(' '))
    if "cellphone" in uniq_final_data1[index]:
        label.append(1)
    else:
        label.append(0)
    index += 1

uniq_final_data = uniq_final_data1
uniq_words = uniq_vocab(uniq_final_data)
uniq_final_data[24476] = ['cellphone', 'service', 'shut', 'boston']

for data in uniq_final_data:
    temp = np.zeros(len(uniq_words))
    for i in data:
        if(i in ['cellphone', 'service', 'shut']):
            temp[uniq_words.index(i)] = 2.0
        else:
            temp[uniq_words.index(i)] = 1.0
    mat.append(temp)

mat = np.array(mat)
node = 24476
eps = 0.01

start_time = time.time()

lengths = []
avg = 0

for i in uniq_final_data:
    avg = avg + len(i)
    lengths.append([len(i)])
avg = avg/len(uniq_final_data)

lengths = np.divide(np.array(lengths), avg)

mat = np.divide(mat, lengths)
print("--- %s seconds completed avg ---" % (time.time() - start_time))

print(uniq_final_data[node])
start_time = time.time()
stat_probability = stat_prob(mat, node, eps)
print("--- %s seconds completed stat_probability ---" % (time.time() - start_time))

b = []
max_a = []
for i in stat_probability:
    b.append(i)

for i in range(len(b)):
    max_a.append(b.index(max(b)))
    b[b.index(max(b))] = -1

rr = []
for i in max_a:
    rr.append(' '.join(uniq_final_data[i]))

the_file = open('sorted_all_without_tfidf_en_2_norm.txt', "w")

for doc in rr:
    the_file.write(str(doc) + "\n")
"""<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"""
"""Calculate the fpr and tpr and plot the ROC"""
tweet_label = np.array(label)
tweet_score = np.array(stat_probability)
fpr, tpr, thresholds = metrics.roc_curve(tweet_label, tweet_score)

plt.plot(fpr, tpr, 'b', label = 'ROC' )
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate or (1 - Specifity)')
plt.ylabel('True Positive Rate or (Sensitivity)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig("ROC_all.png")
plt.show()
"""<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"""
