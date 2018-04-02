#pragma once

#include "Neuron.h"


class NeuralNet
{
public:
	NeuralNet(const vector<unsigned int> &dataStructure, const vector<unsigned int> &hiddenLayerStructure);
	~NeuralNet();
	void feedForward(const vector<double> &inputValues);
	void backPropagation(const vector<double> &targetValues);
	void getResults(vector<double> &resultValues);

private:
	vector<unsigned int> m_hiddenLayerStructure;
	vector<Layer> m_layers;
	double m_error;
};

