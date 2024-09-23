#include "nn.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include <Eigen/Dense>


typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> FloatMatrix;
typedef Eigen::Matrix<float, Eigen::Dynamic, 1> FloatVector;
using GradientsPair = std::pair<std::vector<FloatMatrix>, std::vector<FloatVector>>;

// CONSTRUCTORS
network::network(const std::vector<int> & layer_sizes){
  Layer_sizes = layer_sizes;
  Num_layers = Layer_sizes.size()-1; // the input is not a layer
  // initialize the weights
  int current_layer_size;
  int next_layer_size;
  for (int i=0; i<Num_layers; i++){
    current_layer_size = Layer_sizes[i];
    next_layer_size = Layer_sizes[i+1];
    FloatMatrix new_weights = FloatMatrix::Random(next_layer_size, current_layer_size);
    FloatVector new_bias = FloatVector::Random(next_layer_size);
    Weight_matrices.push_back(new_weights);
    Biases.push_back(new_bias);
  }
  // for now, all activations are sigmoid
  Activation_functions = std::vector<float (*)(float)>(Num_layers, &sigmoid);
  Activation_function_derivatives = std::vector<float (*)(float)>(Num_layers, &sigmoid_derivative);
};

// METHODS
FloatVector network::feed_forward(const FloatVector & x){
  FloatVector v = x;
  for (int i=0; i<Num_layers; i++){
    // apply weights
    v = Weight_matrices[i] * v;
    // apply bias
    v += Biases[i];
    // apply activation function
    v = map_func_to_vec(Activation_functions[i], v);
  }
  return v;
};

std::vector<FloatVector> network::ff_activations_list(const FloatVector & x){
  std::vector<FloatVector> activations;
  activations.push_back(x);
  for (int i=0; i<Num_layers; i++){
    FloatVector v = activations.back();
    // apply weights
    v = Weight_matrices[i] * v;
    // apply bias
    v += Biases[i];
    // apply activation function
    v = map_func_to_vec(Activation_functions[i], v);
    activations.push_back(v);
  }
  return activations;
};


GradientsPair network::back_propagation(const FloatVector & x, const FloatVector & y){
  std::vector<FloatMatrix> weight_gradient_matrices;
  // CAREFUL, THESE ARE STORED IN REVERSE ORDER
  std::vector<FloatVector> bias_gradient_vectors;
  // we need the activations to convert error at each level to a weight gradient
  std::vector<FloatVector> activations= ff_activations_list(x);
  FloatVector level_l_error;
  for (int i=0; i<Num_layers; i++){
    int l = Num_layers-i; // clearer than a reverse loop
    if(i==0){
      // TO DO: ADD POSSIBILITY FOR OTHER COST FUNCTIONS HERE
      FloatVector cost_to_last_layer_grad = MSE_gradient(activations.back(), y);
      level_l_error = map_func_to_vec(Activation_function_derivatives.back(), cost_to_last_layer_grad);
    } 
    else{
      level_l_error = Weight_matrices[l].transpose() * level_l_error; 
      level_l_error = map_func_to_vec(Activation_function_derivatives[l-1], level_l_error);
    }
    FloatMatrix new_matrix_gradient = level_l_error * activations[l-1].transpose();
    // add new gradients
    weight_gradient_matrices.push_back(new_matrix_gradient);
    bias_gradient_vectors.push_back(level_l_error);
  }
  return GradientsPair(weight_gradient_matrices, bias_gradient_vectors);
};

void network::apply_gradients(const GradientsPair & grad_pair){
  // KEEP IN MIND THAT THE GRADIENTS ARE STORED IN REVERSE ORDER, IT AINT CALLED FORWARD PROPAGATION
  for (int i=0; i<Num_layers; i++){
    int l = Num_layers-1-i;
    Biases[i] -= Learning_rate*grad_pair.second[l];
    Weight_matrices[i] -= Learning_rate*grad_pair.first[l];
  }
};

GradientsPair network::cook_batch(const std::vector<FloatVector> & X, const std::vector<FloatVector> & Y, const std::vector<int> indices){
  int batch_size = indices.size();
  std::vector<FloatMatrix> avg_mat_gradients;
  std::vector<FloatVector> avg_bias_gradients;
  for (int i=0; i<batch_size; i++){
    int index = indices[i];
    GradientsPair backp = back_propagation(X[index], Y[index]);
    if (i==0){
      avg_mat_gradients = backp.first;
      avg_bias_gradients = backp.second;
    }
    else{
      // sum term by term
      for (int j=0; j<Num_layers; j++){
        avg_mat_gradients[j] += backp.first[j];
        avg_bias_gradients[j] += backp.second[j];
      }
    }
    // add the 1/N factor to make it an average
    float f = 1.0f/batch_size;
    for (int j=0; j<Num_layers; j++){
        avg_mat_gradients[j] *= f;
        avg_bias_gradients[j] *= f;
    }
  }
  return GradientsPair(avg_mat_gradients, avg_bias_gradients);
};

void network::stochastic_descent(const std::vector<FloatVector> & X, const std::vector<FloatVector> & Y, int epochs, int batch_size){
  int dataset_size=X.size();
  GradientsPair weights_biases_gradients;
  for (int ep=0; ep<epochs; ep++){
    // choose a random batch
    std::vector<int> indices = rand_int_list(batch_size, 0, dataset_size-1);
    // get the average gradient for that batch
    GradientsPair batch_gradients = cook_batch(X, Y, indices);
    // descend
    apply_gradients(batch_gradients);
  }
};

// OTHER FUNCTIONS
FloatVector map_func_to_vec(float (*f)(float), const FloatVector & v){
  int nrows = v.rows();
  FloatVector image(nrows);
  for (int i=0; i<nrows; i++){
    image(i) = f(v(i));
  }
  return image;
};

float sigmoid(float x){
  return 1 / (1 + std::exp(-1 * x));
};

float sigmoid_derivative(float x){
  float s = sigmoid(x);
  return s * (1 - s);
};

float MSE(const FloatVector & x, const FloatVector & y){
  float S=0;
  for (int i=0; i<x.rows(); i++){
    S += pow(x[i] - y[i], 2);
  }
  return S;
};

FloatVector MSE_gradient(const FloatVector & x, const FloatVector & y){
  return 2.0f * (x - y);
};

//std::vector<int> rand_int_list(int num, int min, int max) {
//  std::random_device random_device;
//  std::mt19937 random_engine(random_device());
//  std::uniform_int_distribution<int> distribution(min, max);
//  std::vector<int> numbers(num);
//  for (size_t i=0; i<numbers.size(); i++){
//      numbers[i] = distribution(random_engine);
//  }
//  return numbers;
//};

std::vector<int> rand_int_list(int num, int min, int max) {
  std::vector<int> numbers(num);
  for (size_t i=0; i<numbers.size(); i++){
      numbers[i] = i;
  }
  return numbers;
};

int main(){
  std::vector<int> size_list = {2,2,1};
  network firstnet(size_list);
  // create dataset
  // inputs
  std::vector<FloatVector> inputs = {
        FloatVector(2),
        FloatVector(2),
        FloatVector(2),
        FloatVector(2)
    };
  inputs[0] << 0.0f, 0.0f;
  inputs[1] << 0.0f, 1.0f;
  inputs[2] << 1.0f, 0.0f;
  inputs[3] << 1.0f, 1.0f;
  // outputs
  std::vector<FloatVector> outputs = {
        FloatVector(1),
        FloatVector(1),
        FloatVector(1),
        FloatVector(1)
    };
  outputs[0] << 0.0f;
  outputs[1] << 1.0f;
  outputs[2] << 1.0f;
  outputs[3] << 0.0f;
  // without training
    std::cout << firstnet.feed_forward(inputs[0])
      << "\n"
      << firstnet.feed_forward(inputs[1])
      << "\n"
      << firstnet.feed_forward(inputs[2])
      << "\n"
      << firstnet.feed_forward(inputs[3])
      << std::endl;

  // train the network
  firstnet.stochastic_descent(inputs, outputs, 1000, 1);
  std::cout << "\n" << std::endl;
  std::cout << firstnet.feed_forward(inputs[0])
    << "\n"
    << firstnet.feed_forward(inputs[1])
    << "\n"
    << firstnet.feed_forward(inputs[2])
    << "\n"
    << firstnet.feed_forward(inputs[3])
    << std::endl;
  return 0;
};
