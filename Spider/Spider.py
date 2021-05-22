#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import sys

import Analise
import Create_model
import Load_Data
import Traning

print('Êàêóþ ñåòü Âû õîòèòå îòêðûòü? Ïîñòàâüòå 0, åñëè Fashion è 1, åñëè Conv')
our_net = input()
model = Create_model(tFashion = 1 - our_net, tConv = our_net)
print('Õîòèòå ëè Âû îáó÷èòü ìîäåëü íå ïî óìîë÷àíèþ? Åñëè õîòèòå, ïîñòàâüòå 1, è 0 â îáðàòíîì ñëó÷àå')
f_tran = input()
if f_tran:
    print('Âûáåðåòå îïòèìèçàòîð: 0 - Adam, 1 - Adagrad, 2 - Adadelta, 3 - sgv')
    f_opt = int(input())
print('Åñëè Âû õîòèòå èçìåíèòü ñïîñîá âû÷èòñëåíèÿ îøèáêè, ïîñòàâüòå 0, åñëè íåò è 1 â îáðàòíîì ñëó÷àå')
f_mist = input()
if f_mist:
    print('Âûáåðåòå ñïîñîá: 0 - MSE, 1 - MAE, 2 - SCC')
    mist = int(input())
print('Åñëè ÂÛ õîòèòå èçìåíèòü êîë-âî ýïîõ, íàæìèòå 1. 0 â îáðàòíîì ñëó÷àå')
f_ep = input()
if f_ep:
    print('Ââåäèòå êîë-âî ýïîõ')
    num_ep = int(input())
if f_tran:
    Train = True
    Adam = False
    Adagrad = False
    Adadelta = False
    SGD = False
    if (f_opt == 0):
        Adam = True
    if (f_opt == 1):
        Adagrad = True
    if (f_opt == 2):
        Adadelta = True
    if (f_opt == 3):
        SGD = True
else:
    Train = False
if f_mist:
    Loss = True
    MSE = False
    MAE = False
    SCC = False
    if (mist == 0):
        MSE = True
    if (mist == 1):
        MAE = True
    if (mist == 2):
        SCC = True
else:
    Loss = False
if f_ep:
    Fit = True
    count = num_ep
else:
    Fit = False

result = Traninig(model, 
             Data, 
             Train, Adam, Adagrad, Adadelta, SGD,
             Loss, MSE, MAE, SCC,
             Fit, count)

our_mnist = our_mnist = tf.keras.datasets.fashion_mnist
print('Íóæåí ëè Âàì ãðàôèê òî÷íîñòü îò ýïîõè? 1 - äà, 0 - íåò')
Graf_Ac = input()

print('Íóæåí ëè Âàì äàííûå ïîòåðè òî÷íîñòè? 1 - äà, 0 - íåò')
AcLo = input()

print('Õîòèòå ïîñìîòðåòü ðåç-ò ñåòè? 1 - äà, 0 - íåò')
Images = input()
if Images:
    print('Êàêèå êàðòèíêè õîòèòå ïîñìîòðåòü? Ââåäèòå íîìåðà êàðòèíîê. Ïî óìîë÷àíèþ ïåðâûå 5. Ïî îêíî÷àíèþ ââîäà ââåäèòå -1')
    tmp = int(input())
    while (tmp != -1):
        i_range.append(tmp)
        tmp = int(input())

if (Graf_Ac == 1):
    Graf_Ac = True
else:
    Graf_Ac = False
if (AcLo == 1):
    AcLo = True
else:
    AcLo = False
if (Images == 1):
    Images = True
else:
    Images = False

Analise(result[0], result[1], our_mnist,
           Images, i_range, 
           AcLo, Graf_Ac)


        


