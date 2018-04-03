#pragma once

#include <cmath>
#include <cstdlib>
#include <vector>
#include <iostream>

#define SIGMOID 1
#define TANH 2
#define RELU 3

using namespace std;

class Neuron;
typedef vector<Neuron> Layer;

struct Weights {
	double currentWeight;
	double deltaWeight;
};

class Neuron
{
public:
	Neuron(const unsigned int &numberOfOutputs, const unsigned int &neuronIndexInLayer, const unsigned int &chosenActivationFunction);
	~Neuron();

	void setOutputValue(const double &outputvalue);
	double getOutputValue() const;
	void feedForward(const Layer &previousLayer);
	void calculateOutputGradients(const double &targetvalue);
	void calculateHiddenGradients(const Layer &nextLayer);
	double sumWeightsAndGradients(const Layer &nextLayer);
	void updateWeights(Layer &previousLayer);

private:
	double activationFunction(const double &x);
	double activationFunctionDerivative(const double &x);
	static double sigmoid(const double &x);
	static double ReLU(const double &x, const bool &derivative = false);
	static double randomStartValue();
	static double learningRate;
	static double momentum;
	unsigned int m_activationFunction;
	double m_outputValue;
	double m_outputSum;
	vector<Weights> m_outputWeights;
	double m_indexInLayer;
	double m_gradient;
};

