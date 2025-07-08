#include "lin_reg.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <__ostream/basic_ostream.h>

LinearRegression::LinearRegression(double lr, unsigned int epochs) {
  this->m_lr = lr;
  this->m_epochs = epochs;
  this->m_x = std::vector<std::vector<double>>();
  this->m_y = std::vector<double>();
  this->m_y_hat = std::vector<double>();
  this->m_w = std::vector<double>();
  this->m_bias = 0.0;
}

void LinearRegression::checkInvariants() const {
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
void LinearRegression::initialize_weight(const std::vector<T> &x) {
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
void LinearRegression::load_test_data(std::vector<std::vector<double>> x_test,
                                      std::vector<double> y_test) {
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

  this->m_x = x_test;
  this->m_y = y_test;
  checkInvariants();
};



/**
  @param x being the feature vector we want to predict
 */
double LinearRegression::predict(std::vector<double> x) const {
  if (x.size() != m_w.size()) {
    throw std::invalid_argument("Feature vec not same size as weight vector");
  }

  double prediction = m_bias;
  for (size_t i = 0; i < m_w.size(); i++) {
    prediction += x[i] * m_w[i];
  }
  return prediction;
}

double LinearRegression::mse() const {
  this->checkInvariants();

  if (m_y.size() != m_y_hat.size() || m_y.empty()) {
    throw std::invalid_argument("label or prediction data is corrupted");
  }

  double mse = 0;
  for (size_t i = 0; i < m_y.size(); i++) {
    mse += std::pow((m_y[i] - m_y_hat[i]), 2);
  }
  mse = mse / static_cast<double>(m_y.size());
  return mse;
}

void LinearRegression::fit(std::vector<std::vector<double>> x_test, std::vector<double> y_test){
  this->load_test_data(std::move(x_test), std::move(y_test));
 m_y_hat.resize(m_y.size());
  for (unsigned int epoch = 0; epoch < m_epochs; epoch++){
    for (size_t i = 0; i < m_x.size(); i++)
    {
      m_y_hat[i] = predict(m_x[i]);
    }

    std::vector<double> dw(m_x[0].size(), 0.0);
    double db = 0.0;

    for (size_t i = 0; i < m_y.size(); i++)
    {
      double error = m_y_hat[i] - m_y[i];
      db += error;
      for (size_t j = 0; j < m_w.size(); j++)
      {
        dw[j] += error * m_x[i][j];

      }
    }
    db /= static_cast<double>(m_y.size());
    for (double & j : dw) {
      j /= static_cast<double>(m_y.size());
    }

    m_bias -= m_lr * db;
    for (size_t j = 0; j < m_w.size(); j++) {
      m_w[j] -= m_lr * dw[j];
    }

    if (epoch % 100 == 0) {
      double current_mse = mse();
      std::cout << "MSE: " << current_mse << " At epoch:" << epoch << std::endl;
    }
  }

}

template void LinearRegression::initialize_weight(const std::vector<double> &);
template void LinearRegression::initialize_weight(const std::vector<float> &);
template void LinearRegression::initialize_weight(const std::vector<int> &);
