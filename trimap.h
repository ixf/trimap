#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <limits>
#include <vector>
#include <algorithm>
#include <cassert>

#include "annoy/src/annoylib.h"
#include "annoy/src/kissrandom.h"

using std::vector, std::cout, std::min, std::max;

typedef AnnoyIndex<int, double, Euclidean, Kiss64Random> AnnoyT;

template <typename T>
int sgn(T val);

class Trimap
{
public:
  Trimap(int n_dims, int m, int mi, int r, int out_dims);
  vector<vector<double>> run(vector<vector<double>> input_data);

private:
  int n_dims = 0; // dimensions of final embedding
  int out_dims = 2;

  int m = 0;  // m - number of nearest neighbours per point
  int mi = 0; // m' - number of triplets per nearest neighbour per point
  int r = 0;  // number of random triplets per point

  vector<vector<double>> input;

  vector<vector<int>> triplets;
  vector<double> weights;
  vector<vector<double>> embedding;

  vector<vector<int>> neighbour_index;   // n_samples x m_large
  vector<vector<double>> neighbour_dist; // n_samples x m_large

  // main steps
  void select_triplets();
  void calculate_weights();
  void make_embedding();

  // major helpers
  void initialize_embedding();
  void gd(
      vector<vector<double>> &tmp,
      vector<vector<double>> &velocity,
      vector<vector<double>> &gain,
      int &violations,
      double &loss_value,
      double gamma,
      double learning_rate);

  // minor helpers
  double dist(int i, int j);
  double euclid_dist(vector<double> a, vector<double> b);
  int random(int from, int to);
};

Trimap::Trimap(int n_dims, int m, int mi, int r, int out_dims = 2)
    : n_dims(n_dims), m(m), mi(mi), r(r), out_dims(out_dims){};

vector<vector<double>> Trimap::run(vector<vector<double>> input_data)
{
  this->input = input_data;
  select_triplets();
  calculate_weights();
  make_embedding();
  return this->embedding;
};

void Trimap::select_triplets()
{
  int n_samples = this->input.size();

  AnnoyT nn_search(this->n_dims);
  // todo maybe we could allocate more memory ahead of time (pass large i to
  // add_item first)

  for (size_t i = 0; i < n_samples; ++i)
  {
    vector<double> it = this->input[i];
    nn_search.add_item(i, it.data());
  }
  nn_search.build(20);

  // m_large - how many nearest neighbours do we need to find with Annoy?
  // just m + 1 (the point itself, that's how Annoy returns the neighours)
  // original implementation does this differently
  int m_large = m + 1;

  neighbour_index.resize(n_samples);
  neighbour_dist.resize(n_samples);

  for (int i = 0; i < n_samples; i++)
  {
    nn_search.get_nns_by_item(i, m_large, -1, &neighbour_index[i],
                              &neighbour_dist[i]);
  }

  // Select some triplets
  int n_triplets = (this->m * this->mi + this->r) * n_samples;

  for (int i = 0; i < n_samples; i++)
  {

    int inlier_count = 0;
    for (int j = 0;
         inlier_count < m;
         j++)
    {
      // Annoy may return the point itself as a nearest neighbour
      if (i == neighbour_index[i][j])
      {
        continue;
      }

      inlier_count += 1;

      // Select this->mi random indices
      vector<int> past_outliers;
      past_outliers.reserve(this->mi);
      for (int l = 0; l < this->mi; l++)
      {
        int k = random(0, n_samples);
        if (std::count(past_outliers.begin(), past_outliers.end(), k))
        {
          l -= 1;
          continue;
        }
        past_outliers.push_back(k);

        vector<int> triplet{i, neighbour_index[i][j], k};
        triplets.push_back(triplet);
      }
    }

    // select some random triplets
    for (int x = 0; x < this->r; x++)
    {
      int j = i, k = i;
      while (j == i)
        j = random(0, n_samples);
      while (k == i)
        k = random(0, n_samples);
      double d_ij = nn_search.get_distance(i, j),
             d_ik = nn_search.get_distance(i, k);
      if (d_ij > d_ik)
        std::swap(j, k);
      vector<int> triplet{i, j, k};
      triplets.push_back(triplet);
    }

    int points_processed_so_far = i + 1;
    assert(triplets.size() == points_processed_so_far * (m * mi + r));
  }

  // At this point all the triplets are seleted
  assert(triplets.size() == n_samples * (m * mi + r));
};

void Trimap::calculate_weights()
{
  int n_samples = this->input.size();
  int n_triplets = (this->m * this->mi + this->r) * n_samples;

  // Calculate and apply the scaling factor
  vector<double> sigmas(n_samples); // average of distance from 4th to 6th neighbour
  weights.resize(n_triplets);

  for (int i = 0; i < n_samples; i++)
  {
    for (int j = 3; j < 6; j++)
    {
      sigmas[i] += neighbour_dist[i][j];
    }
    sigmas[i] /= 3;
  }

  for (int a = 0; a < n_triplets; a++)
  {
    vector<int> triplet = triplets[a];

    int i = triplet[0];
    int j = triplet[1];
    int k = triplet[2];

    double d_ij = euclid_dist(input[i], input[j]);
    d_ij = d_ij * d_ij / (sigmas[i] * sigmas[j]);
    d_ij = exp(-d_ij);

    double d_ik = euclid_dist(input[i], input[k]);
    d_ik = d_ik * d_ik / (sigmas[i] * sigmas[k]);
    d_ik = exp(-d_ik);

    // Taken straing from the python implementation
    if (d_ik < 1e-20)
    {
      d_ik = 1e-20;
    }

    double w = d_ij / d_ik;
    if (!isnan(w))
    {
      weights[a] = w;
    }
  }

  // constants for calculating the scaling factor
  constexpr double gamma_factor = 500.0;
  constexpr double delta_constant = 0.0001;
  auto zeta = [&](auto x, auto scaling_factor) {
    return log(1 + gamma_factor * (x / scaling_factor + delta_constant));
  };

  double scaling_factor = *std::max_element(weights.begin(), weights.end());

  for (int i = 0; i < n_triplets; i++)
  {
    weights[i] = zeta(weights[i], scaling_factor);
  }

  double max_weight = *std::max_element(weights.begin(), weights.end());
  for (int i = 0; i < n_triplets; i++)
  {
    weights[i] /= max_weight;
  }
}

void Trimap::make_embedding()
{
  initialize_embedding();
  // run grad descent

  int n_samples = input.size();
  int n_triplets = triplets.size();
  int iterations = 400;
  double learning_rate = 1000.0;
  learning_rate *= (double)input.size() / triplets.size();

  vector<vector<double>> velocity(n_samples, vector<double>(out_dims, 0.0));
  vector<vector<double>> gain(n_samples, vector<double>(out_dims, 1.0));

  double gamma = 0.3;

  for (int iter = 1; iter <= iterations; iter++)
  {
    vector<vector<double>> tmp(n_samples, vector<double>(out_dims, 0.0));
    int violations = 0;      // grad method output
    double loss_value = 0.0; // grad method output

    gamma = (iter >= 250) ? 0.5 : 0.3;

    for (int i = 0; i < n_samples; i++)
    {
      for (int d = 0; d < out_dims; d++)
      {
        tmp[i][d] = embedding[i][d] + gamma * velocity[i][d];
      }
    }

    gd(tmp, velocity, gain,
       violations, loss_value,
       gamma, learning_rate);

    if (iter % 25 == 0)
    {
      cout << "Iteration: " << iter << ", "
           << "Loss: " << loss_value << ", "
           << "Violated triplets: " << violations * 100.0 / n_triplets << "\n";
    }
  }
}

void Trimap::initialize_embedding()
{
  embedding.resize(input.size());
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0.0, 0.001);

  for (int i = 0; i < input.size(); i++)
  {
    embedding[i].resize(out_dims);
    for (int j = 0; j < out_dims; j++)
    {
      embedding[i][j] = distribution(generator);
    }
  }
};

// taken directly from trimap_grad in original python implementation
void Trimap::gd(
    vector<vector<double>> &tmp,
    vector<vector<double>> &velocity,
    vector<vector<double>> &gain,
    int &violations,
    double &loss_value,
    double gamma,
    double learning_rate)
{
  // n -> n_samples
  // dim -> out_dims
  int n_samples = input.size();
  int n_triplets = triplets.size();
  int n_basic_triplets = n_samples * m * mi;
  vector<vector<double>> grad(n_samples, vector<double>(out_dims, 0.0));
  vector<double> y_ij(out_dims);
  vector<double> y_ik(out_dims);
  double loss = 0.0;

  for (int t = 0; t < n_triplets; t++)
  {
    vector<int> triplet = triplets[t];
    int i = triplet[0];
    int j = triplet[1];
    int k = triplet[2];

    double d_ij = 1.0, d_ik = 1.0;

    if ((t % mi == 0) ||
        (t >= n_basic_triplets) // ???
    )
    {
      // update y_ij, y_ik, d_ij, d_ik
      d_ij = 1.0;
      d_ik = 1.0;
      for (int d = 0; d < out_dims; d++)
      {
        y_ij[d] = tmp[i][d] - tmp[j][d];
        y_ik[d] = tmp[i][d] - tmp[k][d];
        d_ij += y_ij[d] * y_ij[d];
        d_ik += y_ik[d] * y_ik[d];
      }
    }
    else
    {
      // update y_ik and d_ik only
      d_ik = 1.0;
      for (int d = 0; d < out_dims; d++)
      {
        y_ik[d] = tmp[i][d] - tmp[k][d];
        d_ik += y_ik[d] * y_ik[d];
      }
    }

    if (d_ij > d_ik)
    {
      violations += 1;
    }

    loss += (double)weights[t] / (1.0 + d_ik / d_ij);
    double dsq = (d_ij + d_ik);
    double w = weights[t] / (dsq * dsq);

    for (int d = 0; d < out_dims; d++)
    {
      double gs = y_ij[d] * d_ik * w;
      double go = y_ik[d] * d_ij * w;
      grad[i][d] += gs - go;
      grad[j][d] -= gs;
      grad[k][d] += go;
    }
  }

  double min_gain = 0.01;

  for (int i = 0; i < n_samples; i++)
  {
    for (int d = 0; d < out_dims; d++)
    {
      gain[i][d] =
          (sgn<double>(velocity[i][d]) != sgn<double>(grad[i][d]))
              ? (gain[i][d] + 0.2)
              : std::max(gain[i][d] * 0.8, min_gain);
      velocity[i][d] = gamma * velocity[i][d] - learning_rate * gain[i][d] * grad[i][d];
      embedding[i][d] += velocity[i][d];
    }
  }

  loss_value = loss;
};

// minor helpers

double Trimap::dist(int i, int j)
{
  vector<double> a = input[i];
  vector<double> b = input[j];
  return euclid_dist(a, b);
};

int Trimap::random(int from, int to) { return rand() % (to - from) + from; };

double Trimap::euclid_dist(vector<double> a, vector<double> b)
{
  double tmp = 0.0, r = 0.0;
  int s = a.size();
  for (int i = 0; i < s; i++)
  {
    tmp = a[i] - b[i];
    r += tmp * tmp;
  }
  return sqrt(r);
};

template <typename T>
int sgn(T val)
{
  return (T(0) < val) - (val < T(0));
}