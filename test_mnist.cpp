#include <fstream>
#include <iostream>

#include "trimap.h"

using std::cout;

int main()
{
  srand(1234);

  int n_samples = 1797;
  int n_dims = 64;

  std::ifstream input_file("mnist_data");
  std::ofstream output_file("impl_out");

  double r;
  vector<vector<double>> data;

  for (int i = 0; i < n_samples; i++)
  {
    vector<double> q;
    for (int j = 0; j < n_dims; j++)
    {
      input_file >> r;
      q.push_back(r);
    }
    data.push_back(q);
  }

  int c = 5;
  auto t = new Trimap(n_dims, c * 2, c, c);
  auto d = t->run(data);
  for (auto &r : d)
  {
    for (auto &c : r)
    {
      output_file << c << " ";
    }
    output_file << "\n";
  }
  delete t;
  return 0;
}
