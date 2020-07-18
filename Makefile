all: test_mnist

test_mnist: test_mnist.cpp trimap.h
	g++ -g -o test_mnist test_mnist.cpp -std=c++17

test_mnist_opt: test_mnist.cpp trimap.h
	g++ -O3 -o test_mnist test_mnist.cpp -std=c++17

clean:
	rm test_mnist
