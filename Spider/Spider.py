import Analise
import Create_model
import Load_Data
import Traning

print('����� ���� �� ������ �������? ��������� 0, ���� Fashion � 1, ���� Conv')
our_net = input()
model = Create_model(tFashion = 1 - our_net, tConv = our_net)
print('������ �� �� ������� ������ �� �� ���������? ���� ������, ��������� 1, � 0 � �������� ������')
f_tran = input()
if f_tran:
    print('�������� �����������: 0 - Adam, 1 - Adagrad, 2 - Adadelta, 3 - sgv')
    f_opt = int(input())
print('���� �� ������ �������� ������ ����������� ������, ��������� 0, ���� ��� � 1 � �������� ������')
f_mist = input()
if f_mist:
    print('�������� ������: 0 - MSE, 1 - MAE, 2 - SCC')
    mist = int(input())
print('���� �� ������ �������� ���-�� ����, ������� 1. 0 � �������� ������')
f_ep = input()
if f_ep:
    print('������� ���-�� ����')
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
print('����� �� ��� ������ �������� �� �����? 1 - ��, 0 - ���')
Graf_Ac = input()

print('����� �� ��� ������ ������ ��������? 1 - ��, 0 - ���')
AcLo = input()

print('������ ���������� ���-� ����? 1 - ��, 0 - ���')
Images = input()
if Images:
    print('����� �������� ������ ����������? ������� ������ ��������. �� ��������� ������ 5. �� ��������� ����� ������� -1')
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


        


