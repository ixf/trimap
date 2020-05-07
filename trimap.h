#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <random>

#include "annoy/src/annoylib.h"
#include "annoy/src/kissrandom.h"

using std::vector, std::cout;

typedef AnnoyIndex<int, float, Euclidean, Kiss64Random> AnnoyT;

int random(int from, int to) {
  return rand() % (to-from) + from;
}

class Trimap {
public:

  int n_dims; // dimensions of final embedding

  int m; // m - number of nearest neighbours per point
  int mi; // m' - number of triplets per nearest neighbour per point
  int r; // number of random triplets per point

  vector<vector<float>> input;
  vector<vector<float>> embedding;

  vector<vector<int>> triplets;

  vector<vector<int>> neighbour_index; // n_samples x m_large
  vector<vector<float>> neighbour_dist; // n_samples x m_large

  float dist(int i, int j) {
    vector<float> a = this->input[i];
    vector<float> b = this->input[j];
    float s = 0.0;
    for(int i = 0; i < a.size(); i++) {
      float q = a[i] - b[i];
      s += q*q;
    }
    return sqrt(s);
  }

  Trimap(int n_dims, int m, int mi, int r);
  vector<vector<float>> run(vector<vector<float>> input_data);
  void select_triplets();
  void make_embedding();
  void validate();
  float loss();
};


Trimap::Trimap(int n_dims, int m, int mi, int r) : n_dims(n_dims), m(m), mi(mi), r(r) {};

vector<vector<float>> Trimap::run(vector<vector<float>> input_data) {
  this->input = input_data;
  select_triplets();
  make_embedding();
  return this->embedding;
};

void Trimap::select_triplets() {
  int n_samples = this->input.size();

  AnnoyT nn_search(this->n_dims);
  // todo maybe we could allocate more memory ahead of time (pass large i to add_item first)

  for(size_t i = 0; i < n_samples; ++i) {
    vector<float> it = this->input[i];
    nn_search.add_item(i, it.data());
  }
  nn_search.build(50);


  int m_large = m + 1;
  neighbour_index.resize(n_samples);
  neighbour_dist.resize(n_samples);

  for(int i = 0; i < n_samples; i++) {
    nn_search.get_nns_by_item(i, m_large, -1, &neighbour_index[i], &neighbour_dist[i]);
  }

  // Select some triplets
  int n_triplets = (this->m * this->mi + this->r) * n_samples;
  vector<vector<int>> triplets;

  for(int i = 0; i < n_samples; i++) {
    for(int j = 0; j < m_large; j++) {

      // annoy may return the point itself as a nn
      if(i == neighbour_index[i][j]) {
        continue;
      }

      for(int l = 0; l < this->mi; l++) {
        vector<int> triplet{i, neighbour_index[i][j]};
        int k = -1;

        // sample k and make sure that it is further to i than j
        while(true) {
          k = random(0, n_samples);
          if(k == i) continue;
          float dist_ij = neighbour_dist[i][j];
          float dist_ik = nn_search.get_distance(i, k);
          // cout << dist_ik << " x " << dist_ij << "\n";
          if(dist_ik > dist_ij) {
            break;
          }
        }

        // cout << i << " " << neighbour_index[i][j] << " " << k << "\n";
        triplet.push_back(k);
        triplets.push_back(triplet);
      }
    }

    // select some random triplets
    for(int i = 0; i < this->r; i++) {
      int j = i, k = i;
      while(j == i) j = random(0, n_samples);
      while(k == i) k = random(0, n_samples);
      float d_ij = nn_search.get_distance(i, j),
            d_ik = nn_search.get_distance(i, k);
      if(d_ij > d_ik) std::swap(j, k);
      vector<int> triplet{i, j, k};
      triplets.push_back(triplet);
    }
  }

  // WEIGHTS
  // Calculate and apply the scaling factor
  vector<float> sigmas(n_samples); // average of distance from 4th to 6th neighbour
  for(int i = 0; i < n_samples; i++){
    for(int j = 3; j < 6; j++) {
      sigmas[i] += neighbour_dist[i][j];
    }
    sigmas[i] /= 3;
  }

  vector<float> weights(n_triplets);
  for(int a = 0; a < n_triplets; a++) {
    vector<int> triplet = triplets[a];

    int i = triplet[0];
    int j = triplet[1];
    int k = triplet[2];

    float d_ij = nn_search.get_distance(i, j) / sigmas[i] / sigmas[j];
    float d_ik = nn_search.get_distance(i, k) / sigmas[i] / sigmas[k];

    weights[i] = exp(d_ik - d_ij);
  }

  // constants for calculating the scaling factor
  constexpr float gamma_factor = 500.0;
  constexpr float delta_constant = 10e-4;
  auto zeta = [](auto x, auto scaling_factor) { return log(1 + gamma_factor * (x / scaling_factor + delta_constant)); };

  float scaling_factor = *std::max_element(weights.begin(), weights.end());

  for(int i = 0; i < n_triplets; i++) {
    weights[i] = zeta(weights[i], scaling_factor);
  }
};

void Trimap::make_embedding() {
  std::default_random_engine generator;
  std::normal_distribution<float> distribution(0.0, 0.001);

  for(int i = 0; i < input.size(); i++) {
    vector<float> f(n_dims);
    for(int j = 0; j < n_dims; j++) {
      f.push_back(distribution(generator));
    }
    embedding.push_back(f);
  }


  // ...
}

void Trimap::validate() {
  for(int i = 0; i < triplets.size(); i++){
    vector<int> triplet = triplets[i];
    int a = triplet[0];
    int b = triplet[1];
    int c = triplet[2];

    float dab = dist(a, b);
    float dac = dist(a, c);

    if(dac < dab) {
      cout << i << ": " << a << " " << b << " " << c << " " << dab << " " << dac << "\n";
    }
  }
}

float Trimap::loss() {
  float s = 0.0;

  for(int i = 0; i < triplets.size(); i++) {
    
  }

  return s;
}
