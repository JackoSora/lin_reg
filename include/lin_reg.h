#include <vector>

using namespace std;

class LinearRegression {
private:
  vector<vector<double>> m_x; // Input features
  vector<double> m_y;         // Output values
  vector<double> m_w;         // Weight (slope)
  vector<double> m_y_hat;
  double m_bias;         // Bias (intercept)
  double m_lr;           // Learning rate (for gradient descent)
  unsigned int m_epochs; // Number of iterations

public:
  explicit LinearRegression(double lr = 0.01, unsigned int epochs = 1000);
  void checkInvariants() const;
  [[nodiscard]] unsigned int getEpochs() const
  {
      return m_epochs;
  };


  void fit(vector<vector<double>> x,
           vector<double>
           y); // fit the model to the feature vectors with labels
  [[nodiscard]]double predict(vector<double> x) const; // Predict output for a feature vector

  [[nodiscard]] double mse() const;

  void load_test_data(vector<vector<double>> x_test, vector<double> y_test);

  template <typename T> void initialize_weight(const vector<T> &x);

  // Optional: for evaluating on new data
  double score(const std::vector<double> &x_test,
               const std::vector<double> &y_test) const;

  // Optional: setters for learning rate and epochs
  void set_learning_rate(double lr);
  void set_epochs(int epochs);
};