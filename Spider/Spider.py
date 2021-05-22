import Analise
import Create_model
import Load_Data
import Traning

print('Какую сеть Вы хотите открыть? Поставьте 0, если Fashion и 1, если Conv')
our_net = input()
model = Create_model(tFashion = 1 - our_net, tConv = our_net)
print('Хотите ли Вы обучить модель не по умолчанию? Если хотите, поставьте 1, и 0 в обратном случае')
f_tran = input()
if f_tran:
    print('Выберете оптимизатор: 0 - Adam, 1 - Adagrad, 2 - Adadelta, 3 - sgv')
    f_opt = int(input())
print('Если Вы хотите изменить способ вычитсления ошибки, поставьте 0, если нет и 1 в обратном случае')
f_mist = input()
if f_mist:
    print('Выберете способ: 0 - MSE, 1 - MAE, 2 - SCC')
    mist = int(input())
print('Если ВЫ хотите изменить кол-во эпох, нажмите 1. 0 в обратном случае')
f_ep = input()
if f_ep:
    print('Введите кол-во эпох')
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
print('Нужен ли Вам график точность от эпохи? 1 - да, 0 - нет')
Graf_Ac = input()

print('Нужен ли Вам данные потери точности? 1 - да, 0 - нет')
AcLo = input()

print('Хотите посмотреть рез-т сети? 1 - да, 0 - нет')
Images = input()
if Images:
    print('Какие картинки хотите посмотреть? Введите номера картинок. По умолчанию первые 5. По окночанию ввода введите -1')
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


        


