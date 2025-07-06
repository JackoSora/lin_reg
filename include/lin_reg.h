#include <vector>

using namespace std;

class LinearRegression {
private:
  vector<vector<double>> m_x; // Input features
  vector<double> m_y;         // Output values
  vector<double> m_w;         // Weight (slope)
  double m_bias;              // Bias (intercept)
  double m_lr;                // Learning rate (for gradient descent)
  int m_epochs;               // Number of iterations

public:
  LinearRegression(double lr = 0.01, int epochs = 1000);

  void fit(const vector<vector<double>> x,
           const vector<double>
               y); // fit the model to the feature vectors with labels
  double predict(vector<double> x) const; // Predict output for a feature vector

  // Metrics and helpers
  double get_w() const;
  double get_b() const;
  double get_loss() const;
  double get_mse() const;
  double get_rmse() const;

  void load_test_data(vector<vector<double>> x_test, vector<double> y_test);

  template <typename T> void initialize_weight(const vector<T> &x);

  // Optional: for evaluating on new data
  double score(const std::vector<double> &x_test,
               const std::vector<double> &y_test) const;

  // Optional: setters for learning rate and epochs
  void set_learning_rate(double lr);
  void set_epochs(int epochs);
};