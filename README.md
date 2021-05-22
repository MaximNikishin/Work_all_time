# Work_all_time
NO ERROR HANDLING, IT'S NECESSARY TO ADAPT TO THE PROGRAMM REQUESTS

Our project calls Spider (it makes nets).

OS version: Ubuntu 18.04, Python version: 3.7.0 or more

To work with our programm you need some libruaries such as:
numpy
tensorflow (2.1.0 or more)
matplotlib
PyQt

We build our programm by pyinstall.

You can watch how work some nets in programms:
Test_Fashion.py (it's default one)
Test_Conv2D.py (it's can be choose such as ready-made solution)

Programm Create_model.py contains function Create_model(),
that builds net (default or your own (not realise yet)) and returns it.

Programm Training.py contains function Train(), 
that chooses optimiser, loss and number of epoch. 
It returns fit model and history of fitting.

Programm Load_Data() contains function Taking_Dataset(), 
that returns numpy's dataset.

Programm Analise.py contains major function Analise(), 
that helps you to draw mathplotlib grafics, statistics of fitting and result work of net.
