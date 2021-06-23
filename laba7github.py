import numpy as np
import random
from datetime import datetime
import matplotlib.pyplot as plt

q = 1
r = 10 ** 6
x_n = []
alg1 = []
algb1 = []
alg2 = []
algb2 = []


for n in range(10 ** 2, 10 ** 4 + 1, 100):
    x_n.append(n)
    m = n**2//100

    # Создание графа
    G, W = [], dict()
    for k in range(n):
        G.append([])
    for k in range(m):
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if (i != j) and (not ((i, j) in W)):
            G[i].append(j)
            W[(i, j)] = random.randint(q, r)

    # Алгоритм Форда-Беллмана
    alg1.append(datetime.now())
    d = [np.inf] * n
    d[0] = 0
    for k in range(1, n):
        for j, i in W.keys():
            if d[j] + W[j, i] < d[i]:
                d[i] = d[j] + W[j, i]
    alg1[len(alg1) - 1] = (datetime.now() - alg1[len(alg1) - 1]).total_seconds()
    print('При n =', n, 'время:', alg1[len(alg1) - 1])

    # Алгоритм Дейкстры
    algb1.append(datetime.now())
    d = [np.inf] * n
    d[0], min_d, min_v, used = 0, 0, 0, [False] * n
    while min_d < np.inf:
        i = min_v
        used[i] = True
        for j in G[i]:
            if d[i] + W[(i, j)] < d[j]:
                d[j] = d[i] + W[(i, j)]
        min_d = np.inf
        for j in range(n):
            if not used[j] and d[j] < min_d:
                min_d = d[j]
                min_v = j
    algb1[len(algb1) - 1] = (datetime.now() - algb1[len(algb1) - 1]).total_seconds()
    print('При n =', n, 'время:', algb1[len(algb1) - 1])

    m = n**2//1000

    # Создание графа
    G, W = [], dict()
    for k in range(n):
        G.append([])
    for k in range(m):
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if (i != j) and (not ((i, j) in W)):
            G[i].append(j)
            W[(i, j)] = random.randint(q, r)

    # Алгоритм Форда-Беллмана
    alg2.append(datetime.now())
    d = [np.inf] * n
    d[0] = 0
    for k in range(1, n):
        for j, i in W.keys():
            if d[j] + W[j, i] < d[i]:
                d[i] = d[j] + W[j, i]
    alg2[len(alg2) - 1] = (datetime.now() - alg2[len(alg2) - 1]).total_seconds()
    print('При n =', n, 'время:', alg2[len(alg2) - 1])

    # Алгоритм Дейкстры
    algb2.append(datetime.now())
    d = [np.inf] * n
    d[0], min_d, min_v, used = 0, 0, 0, [False] * n
    while min_d < np.inf:
        i = min_v
        used[i] = True
        for j in G[i]:
            if d[i] + W[(i, j)] < d[j]:
                d[j] = d[i] + W[(i, j)]
        min_d = np.inf
        for j in range(n):
            if not used[j] and d[j] < min_d:
                min_d = d[j]
                min_v = j
    algb2[len(algb2) - 1] = (datetime.now() - algb2[len(algb2) - 1]).total_seconds()

    print('При n =', n, 'время:', algb2[len(algb2) - 1])


def plots():
    fig = plt.figure()

    nalg1 = fig.add_subplot(221)
    nalg1.set_xlabel('n')
    nalg1.set_ylabel('t')
    nalg1.set_title('Алгоритм FB в случае а)')
    plt.plot(x_n, alg1)

    nalgb1 = fig.add_subplot(223)
    nalgb1.set_xlabel('n')
    nalgb1.set_ylabel('t')
    nalgb1.set_title('Алгоритм D в случае а)')
    plt.plot(x_n, algb1)

    nalg2 = fig.add_subplot(222)
    nalg2.set_xlabel('n')
    nalg2.set_ylabel('t')
    nalg2.set_title('Алгоритм FB в случае б)')
    plt.plot(x_n, alg2)

    nalgb2 = fig.add_subplot(224)
    nalgb2.set_xlabel('n')
    nalgb2.set_ylabel('t')
    nalgb2.set_title('Алгоритм D в случае б)')
    plt.plot(x_n, algb2)

    plt.tight_layout()

    plt.show()


plots()
# Алгоритм Дейкстры намного быстрее алгоритма Форда-Беллмана