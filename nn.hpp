#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include <Eigen/Dense>

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> FloatMatrix;
typedef Eigen::Matrix<float, Eigen::Dynamic, 1> FloatVector;
using GradientsPair = std::pair<std::vector<FloatMatrix>, std::vector<FloatVector>>;

class network{
private:
  int Num_layers;
  std::vector<int> Layer_sizes;
  std::vector<FloatVector> Layers; // to store activations 
  std::vector<FloatMatrix> Weight_matrices;
  std::vector<FloatVector> Biases;
  float Learning_rate=1.0f;
  // pointers for the activation function of each layer
  std::vector<float (*)(float)> Activation_functions;
  std::vector<float (*)(float)> Activation_function_derivatives;
public:
  // constructors
  network(const std::vector<int> & layer_sizes); // supposing no layer is larger than max int
  // methods
  FloatVector feed_forward(const FloatVector & x);
  std::vector<FloatVector> ff_activations_list(const FloatVector & x);
  GradientsPair back_propagation(const FloatVector & x, const FloatVector & y);
  GradientsPair cook_batch(const std::vector<FloatVector> & X, const std::vector<FloatVector> & Y, const std::vector<int> indices);
  void apply_gradients(const GradientsPair & grad_pair);
  void stochastic_descent(const std::vector<FloatVector> & X, const std::vector<FloatVector> & Y, int epochs, int batch_size);
};

FloatVector map_func_to_vec(float (*f)(float), const FloatVector & v);

float sigmoid(float x);
float sigmoid_derivative(float x);

float MSE(const FloatVector & x, const FloatVector & y);
FloatVector MSE_gradient(const FloatVector & x, const FloatVector & y);
std::vector<int> rand_int_list(int num, int min, int max);
