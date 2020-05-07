#include <iostream>
#include <fstream>

#include "trimap.h"

int main() {
  srand(1234);

  int n_samples = 1000;
  int n_dims = 60;

  std::ifstream input_file("mnist_data");

  float r;
  vector<vector<float>> data;

  for(int i = 0; i < n_samples; i++) {
    vector<float> q;
    for(int j = 0; j < n_dims; j++) {
      input_file >> r;
      q.push_back(r);
    }
    data.push_back(q);
  }

  int c = 5;
  auto t = new Trimap(2, c*2, c, c);
  t->run(data);
  cout << "ok\n";
  delete t;
}
