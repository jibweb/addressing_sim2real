hellomake: preprocessing/graph_construction.h preprocessing/graph_construction.cpp \
           preprocessing/augmentation_preprocessing.cpp preprocessing/parameters.h \
           preprocessing/py_graph_construction.pyx
	python setup.py build_ext -i
