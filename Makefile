all: test test_mnist

test: test.cpp trimap.h
	g++ -g -o test test.cpp -std=c++17

test_mnist: test_mnist.cpp trimap.h
	g++ -g -o test_mnist test_mnist.cpp -std=c++17
