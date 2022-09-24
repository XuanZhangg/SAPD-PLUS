from turtle import color
import numpy as np
import matplotlib.pyplot as plt

def isPrime(x):
    if x < 2:
        return False
    if x == 2 or x == 3:
        return True
    for i in range(2, int(np.sqrt(x)) + 1):
        if x % i == 0:
            return False

    return True

def getNthPrime(x):
    cnt = 0
    num = 2
    while True:
        if isPrime(num):
            cnt = cnt + 1
            if cnt == x:
                return num
        num = num + 1

def shadowplot(x: list, y: list, alg_name: str, center=0, alpha=0.5, label_input=None, is_var=False, linestyle_input='-',is_log=False, ):
    temp = []
    colors = {'SAPD-VR': 'b', 'SAPD':'c', 'SMDA':'saddlebrown','SMDA-VR':'orangered', 'PASGDA':'blueviolet', 'SREDA':'g' }
    for i in range(len(y)):
        y[i] = [val-center for val in y[i]]

        if is_log:
            y[i]=np.log(y[i])
        temp.append(y[i])
    y = temp

    mid_line = [np.average(val) for val in zip(*y)]
    var =  [np.var(val) for val in zip(*y)]

    if is_var:
        lowline = [x*(1+y) for x,y in zip(mid_line,var)]
        highline = [x*(1-y) for x,y in zip(mid_line,var)]
    else:
        lowline = [np.min(val) for val in zip(*y)]
        highline = [np.max(val)for val in zip(*y)]

    plt.plot(x,mid_line, label = label_input, linestyle = linestyle_input, color=colors[alg_name])
    plt.fill_between(x, lowline, highline, alpha=0.2, facecolor=colors[alg_name])
    # plt.fill_between(x, lowline, highline, facecolor='green', alpha=0.2)
    return
