#include <vector>
#include <iostream>

#include "trimap.h"

using std::vector, std::cout;

int main () {
  vector<vector<float>> data;
  data.push_back({1, 0, 0});
  data.push_back({1, 1, 0});
  data.push_back({1, 4, 1});
  Trimap t(2, 1, 1, 0);
  t.run(data);

  std::cout << "Triplets: " << t.triplets.size() << "\n";
  for(int i = 0; i < t.triplets.size(); i++) {
    vector<int> triplet = t.triplets[i];
    cout << triplet[0] << " " << triplet[1] << " " << triplet[2] << "\n";
  }
}
