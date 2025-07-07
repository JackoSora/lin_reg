#include "lin_reg.h"
#include <cassert>
#include <random>
#include <stdexcept>

using namespace std;

LinearRegression::LinearRegression(double lr, unsigned int epochs) {
  this->m_lr = lr;
  this->m_epochs = epochs;
  this->m_x = vector<vector<double>>();
  this->m_y = vector<double>();
  this->m_y_hat = vector<double>();
  this->m_w = vector<double>();
  this->m_bias = 0.0;
}

void LinearRegression::checkInvariants() {
  // Data consistency: if we have training data, X and Y must have same number
  // of samples
  if (!m_x.empty() && !m_y.empty()) {
    assert(m_x.size() == m_y.size() &&
           "X and Y must have same number of samples");
  }

  // If we have predictions, they must match the number of training samples
  if (!m_y_hat.empty()) {
    assert(m_y_hat.size() == m_y.size() &&
           "Predictions must match number of training samples");
  }

  // If we have training data, all feature vectors must have same dimensionality
  if (!m_x.empty()) {
    size_t feature_dim = m_x[0].size();
    for (const auto &row : m_x) {
      assert(row.size() == feature_dim &&
             "All feature vectors must have same dimensionality");
    }
  }

  // If we have weights, they must match the feature dimensionality
  if (!m_w.empty() && !m_x.empty()) {
    assert(m_w.size() == m_x[0].size() &&
           "Weight vector must match feature dimensionality");
  }

  // Learning rate must be positive
  assert(m_lr > 0.0 && "Learning rate must be positive");

  // Epochs must be positive
  assert(m_epochs > 0 && "Number of epochs must be positive");
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
  checkInvariants();
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
    prediction += x[i] * m_w[i];
  }
  return prediction;
}

template void LinearRegression::initialize_weight(const vector<double> &);
template void LinearRegression::initialize_weight(const vector<float> &);
template void LinearRegression::initialize_weight(const vector<int> &);