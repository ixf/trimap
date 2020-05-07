

all: trimap

trimap: trimap.cpp
	g++ -g -o trimap trimap.cpp -std=c++17
