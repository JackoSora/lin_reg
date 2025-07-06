#include "lin_reg.h"
#include <cmath>

using namespace std;

LinearRegression::LinearRegression(double lr, int epochs) {
  this->m_lr = lr;
  this->m_epochs = epochs;
  this->m_x = vector<vector<double>>();
  this->m_y = vector<double>();
  this->m_w = vector<double>();
  this->m_bias = 0.0;
}
// next add method to add training data and initiliaze weight