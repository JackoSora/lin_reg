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

void LinearRegression::load_test_data(vector<vector<double>> x_test,
                                      vector<double> y_test) {
  if (x_test.size() != y_test.size()) {
    throw std::invalid_argument("label size doesn't match entry size");
  }

  this->m_x = x_test;
  this->m_y = y_test;
};

template void LinearRegression::initialize_weight(const vector<double> &);
template void LinearRegression::initialize_weight(const vector<float> &);
template void LinearRegression::initialize_weight(const vector<int> &);