all:
	g++ treino_emd_laminas.cpp `pkg-config --libs --cflags opencv` -O2
