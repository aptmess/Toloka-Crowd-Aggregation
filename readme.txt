final_class.ipynb:
	Jupyter Notebook с демонстрацией работы алгоритмов,
	будет использоваться в качестве презентации на защите
	здесь подробнее об этом:
	(https://docs.microsoft.com/ru-ru/azure/notebooks/present-jupyter-notebooks-slideshow)
	ноутбук запускается, работает в точности до воспроизведения примерно 45-50 минут 
	для всех алгоритмов, кроме GLAD из-за случайных начальных весов, задающихся нормальным распределением

final_class_.py:
	class Aggregation Methods - реализованные методы аггрегации данных

DawidScene.py:
	реализация Dawid Scene Algorithm

glad.py:
	реализация алгоритма GLAD

requirements.txt:
	необходимые библиотеки для работы ноутбука, RISE: для презентации

/toloka_2(toloka_5): 
	исходный датасет

/papers : 
	статьи, которые использовал для реализации
/predict :
	csv файлы с предсказанием меток для toloka_2 и toloka_5
	стобцы - название алгоритма, строки - сайты