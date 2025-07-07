#include "lin_reg.h"
#include <random>
#include <stdexcept>

using namespace std;

LinearRegression::LinearRegression(double lr, int epochs) {
  this->m_lr = lr;
  this->m_epochs = epochs;
  this->m_x = vector<vector<double>>();
  this->m_y = vector<double>();
  this->m_w = vector<double>();
  this->m_bias = 0.0;
}

template <typename T>
void LinearRegression::initialize_weight(const vector<T> &x) {
  m_w.clear();
  m_w.resize(x.size());
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(-1.0, 1.0);

  // Initialize each weight with a random value
  for (size_t i = 0; i < m_w.size(); ++i) {
    m_w[i] = dist(gen);
  }
};

/**
    @param x_test n x m matrix where n = training_data_size and m = number of
   feature per data entry
    @param y_test corresponding labels to the datasets

 */
void LinearRegression::load_test_data(vector<vector<double>> x_test,
                                      vector<double> y_test) {
  if (x_test.size() != y_test.size()) {
    throw std::invalid_argument("label size doesn't match entry size");
  } else if (x_test.empty() || y_test.empty()) {
    throw std::invalid_argument("empty dataset");
  }
  for (const auto &row : x_test) {
    if (row.size() != x_test[0].size()) {
      throw std::invalid_argument("inconsistent feature size in dataset");
    }
  }
  // --- Initialize weights based on the first row of x_test - part before is
  // just error handlign
  this->initialize_weight(x_test[0]);

  this->m_w.clear();
  this->m_w.resize(x_test[0].size());

  this->m_y.clear();
  this->m_y.resize(y_test.size());

  this->m_x = x_test;
  this->m_y = y_test;
};

/**
  @param x being the feature vector we want to predict
 */
double LinearRegression::predict(vector<double> x) const {
  if (x.size() != m_w.size()) {
    throw std::invalid_argument("Feature vec not same size as weight vector");
  }

  double prediction = m_bias;
  for (size_t i = 0; i < m_w.size(); i++) {
    prediction += x[i] + m_w[i];
  }
  return prediction;
}

template void LinearRegression::initialize_weight(const vector<double> &);
template void LinearRegression::initialize_weight(const vector<float> &);
template void LinearRegression::initialize_weight(const vector<int> &);