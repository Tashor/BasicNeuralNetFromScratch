#include "NeuralNet.h"



NeuralNet::NeuralNet(const vector<unsigned int> &dataStructure, const vector<unsigned int> &hiddenLayerStructure, const unsigned int chosenActivationFunction) {
	
	// combine input-, output- and hidden layers
	vector<unsigned int> fullNetStructure;
	if (dataStructure.size() == 2)		// dataStructure has to have 2 values (number of inputs and -outputs)
		fullNetStructure.push_back(dataStructure[0]);	// add input layer
	else
		return;

	for (unsigned int i = 0; i < hiddenLayerStructure.size(); i++) {	// add all hidden layers
		fullNetStructure.push_back(hiddenLayerStructure[i]);
	}

	fullNetStructure.push_back(dataStructure[1]);	// add output layer


	unsigned int numberOfLayers = fullNetStructure.size();		// number of hidden layers + input- & output layer

	// create layers
	for (unsigned int layerNumber = 0; layerNumber < numberOfLayers; layerNumber++) {
		m_layers.push_back(vector<Neuron>());

		// determine how many outputs each neuron has to have
		unsigned int numberOfOutputs;
		if (layerNumber == numberOfLayers - 1)
			numberOfOutputs = 0;	// last layer has no outputs
		else
			numberOfOutputs = fullNetStructure[layerNumber + 1];	// set numberOfOutputs to number of neurons in the next layer (fully connected neural network)

		//cout << "Layer " << layerNumber << ":" << endl;
		// create neurons in current layer + bias (<=)
		for (unsigned int neuronNumber = 0; neuronNumber <= fullNetStructure[layerNumber]; neuronNumber++) {
			m_layers[layerNumber].push_back(Neuron(numberOfOutputs, neuronNumber, chosenActivationFunction));
			//cout << "Created Neuron!" << endl;
		}

		// set bias outputValue to -1.0 (last neuron)
		m_layers[layerNumber].back().setOutputValue(-1.0);
	}
}


NeuralNet::~NeuralNet()
{
}

void NeuralNet::feedForward(const vector<double>& inputValues)
{
	//cout << endl << "FeedForward: " << endl;
	if (inputValues.size() != m_layers[0].size() - 1) {
		cout << "Number of neurons in first layer != number of input values!" << endl;
		return;
	}

	// initialize input layer
	for (unsigned int currentNeuron = 0; currentNeuron < inputValues.size(); currentNeuron++) {
		m_layers[0][currentNeuron].setOutputValue(inputValues[currentNeuron]);
	}

	// feed forward to last layer
	for (unsigned int layerNumber = 1; layerNumber < m_layers.size(); layerNumber++) {
		//cout << "Layer " << layerNumber << ":" << endl;
		// calculate output value for each neuron (excluding bias neuron) in the current layer based on outputvalues and weights of previous layer
		for (unsigned int currentNeuron = 0; currentNeuron < m_layers[layerNumber].size() - 1; currentNeuron++) {
			m_layers[layerNumber][currentNeuron].feedForward(m_layers[layerNumber - 1]);
		}
	}
}

void NeuralNet::backPropagation(const vector<double> &targetValues)
{
	double m_error = 0.0;

	// calculate mean-squared-error (MSE)
	for (unsigned int currentNeuron = 0; currentNeuron < m_layers.back().size() - 1; currentNeuron++) {
		double delta = m_layers.back()[currentNeuron].getOutputValue() - targetValues[currentNeuron];	// current output value - desired output value
		m_error += delta * delta;	// sum all squared deltas 
	}
	m_error /=  (m_layers.back().size() - 1);	// devide by N 
	//cout << "Error: " << m_error << endl;

	// calculate the output layer gradients
	for (unsigned int currentNeuron = 0; currentNeuron < m_layers.back().size() - 1; currentNeuron++) {
		m_layers.back()[currentNeuron].calculateOutputGradients(targetValues[currentNeuron]);
	}

	// calculate hidden layer gradients
	for (unsigned int layerNumber = m_layers.size() - 2; layerNumber > 0; layerNumber--) {
		// calculate gradients for all neurons in the current layer
		for (unsigned int currentNeuron = 0; currentNeuron < m_layers[layerNumber].size(); currentNeuron++) {
			m_layers[layerNumber][currentNeuron].calculateHiddenGradients(m_layers[layerNumber + 1]);
		}
	}

	// update weights for all layers (from output layer to first layer)
	for (unsigned int layerNumber = m_layers.size() - 1; layerNumber > 0; layerNumber--) {
		for (unsigned int currentNeuron = 0; currentNeuron < m_layers[layerNumber].size() - 1; currentNeuron++) {	// ToDo: -1 or not??
			m_layers[layerNumber][currentNeuron].updateWeights(m_layers[layerNumber - 1]);
		}
	}
}

void NeuralNet::getResults(vector<double>& resultValues)
{
	resultValues.clear();
	for (unsigned int currentNeuron = 0; currentNeuron < m_layers.back().size() - 1; currentNeuron++) {
		resultValues.push_back(m_layers.back()[currentNeuron].getOutputValue());
	}
}
