#pragma once

#include <cmath>
#include <cstdlib>
#include <vector>
#include <iostream>


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
	Neuron(const unsigned int &numberOfOutputs, const unsigned int &neuronIndexInLayer);
	~Neuron();
	void setOutputValue(const double &outputvalue);
	double getOutputValue() const;
	void feedForward(const Layer &previousLayer);
	void calculateOutputGradients(const double &targetvalue);
	void calculateHiddenGradients(const Layer &nextLayer);
	double sumWeightsAndGradients(const Layer &nextLayer);
	void updateWeights(Layer &previousLayer);

private:
	static double activationFunction(const double &x);
	static double activationFunctionDerivative(const double &x);
	static double sigmoid(const double &x);
	static double randomStartValue();
	static double learningRate;
	static double momentum;
	double m_outputValue;
	double m_outputSum;
	vector<Weights> m_outputWeights;
	double m_indexInLayer;
	double m_gradient;
};

