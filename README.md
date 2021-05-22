NO ERROR HANDLING, IT'S NECESSARY TO ADAPT TO THE PROGRAMM REQUESTS
Our project calls Spider (it makes nets).
OS version: Ubuntu 18.04, Python version: 3.7.0 or more
To work with our program you need some libraries such as: numpy tensorflow (2.1.0 or more) matplotlib PyQt
We build our program by pyinstall.
You can watch how work some nets in programs: Test_Fashion.py (it's default one) Test_Conv2D.py (it's can be chosen such as ready-made solution)
Programm Create_model.py contains function Create_model(), that builds net (default or your own (not realise yet)) and returns it.
Programm Training.py contains function Train(), that chooses optimizer, loss and number of epoch. It returns fit model and history of fitting.
Programm Load_Data() contains function Taking_Dataset(), that returns numpy's dataset.
Programm Analise.py contains major function Analise(), that helps you to draw mathplotlib grafics, statistics of fitting and result work of net.

