#include "Neuron.h"

double Neuron::learningRate = 0.01;
double Neuron::momentum = 0.9;

Neuron::Neuron(const unsigned int &numberOfOutputs, const unsigned int &neuronIndexInLayer, const unsigned int &chosenActivationFunction)
{
	for (unsigned int i = 0; i < numberOfOutputs; i++) {
		m_outputWeights.push_back(Weights());
		m_outputWeights[i].currentWeight = randomStartValue();
		// cout << "Weight: " << m_outputWeights[i].currentWeight << endl;
	}
	m_outputValue = randomStartValue();
	m_indexInLayer = neuronIndexInLayer;
	m_activationFunction = chosenActivationFunction;
	//cout << "Created " << m_outputWeights.size() << " outputs for neuron: " << m_indexInLayer << endl;
}

Neuron::~Neuron()
{
}

void Neuron::setOutputValue(const double &outputvalue)
{
	m_outputValue = outputvalue;
}

double Neuron::getOutputValue() const
{
	return m_outputValue;
}

void Neuron::feedForward(const Layer &previousLayer)
{
	double sum = 0.0;

	// calculate sum of all inputs * input-weights for this neuron based on previous layer
	for (unsigned int currentNeuron = 0; currentNeuron < previousLayer.size(); currentNeuron++) {
		sum += previousLayer[currentNeuron].getOutputValue() * previousLayer[currentNeuron].m_outputWeights[m_indexInLayer].currentWeight;
	}
	//cout << "outputValue: " << m_outputValue << endl;
	m_outputSum = sum;
	m_outputValue = activationFunction(sum);
	//cout << "sum before activationFunction: " << sum << " outputValue after: " << m_outputValue << endl;
}

void Neuron::calculateOutputGradients(const double &targetvalue)
{
	m_gradient = activationFunctionDerivative(m_outputSum) * (m_outputValue - targetvalue);
	//cout << "Gradient: " << m_gradient << endl;
}

void Neuron::calculateHiddenGradients(const Layer &nextLayer)
{
	m_gradient = activationFunctionDerivative(m_outputSum) * sumWeightsAndGradients(nextLayer);
	//cout << "Hidden-Gradient: " << m_gradient << endl;
}

double Neuron::sumWeightsAndGradients(const Layer & nextLayer)
{
	double sum = 0.0;
	for (unsigned int currentNeuron = 0; currentNeuron < nextLayer.size() - 1; currentNeuron++) {
		sum += m_outputWeights[currentNeuron].currentWeight * nextLayer[currentNeuron].m_gradient;
	}

	return sum;
}

void Neuron::updateWeights(Layer & previousLayer)
{
	for (unsigned int currentNeuron = 0; currentNeuron < previousLayer.size(); currentNeuron++) {
		double oldDeltaWeight = previousLayer[currentNeuron].m_outputWeights[m_indexInLayer].deltaWeight;
		double newDeltaWeight = -learningRate * previousLayer[currentNeuron].getOutputValue() * m_gradient + momentum * oldDeltaWeight;
		//cout << "newDeltaWeight: " << newDeltaWeight << endl;

		previousLayer[currentNeuron].m_outputWeights[m_indexInLayer].currentWeight += newDeltaWeight;
		previousLayer[currentNeuron].m_outputWeights[m_indexInLayer].deltaWeight = newDeltaWeight;
	}
}

double Neuron::activationFunction(const double &x)
{
	double result = 0.0;
	switch (m_activationFunction) 
	{
		case SIGMOID:	// sigmoid as activation function
			result = sigmoid(x);
			break;

		case TANH:		// hyperbolic tangens as activation function
			result = tanh(x);
			break;

		case RELU:
			result = ReLU(x);
			break;

		case LEAKYRELU:
			result = LeakyReLU(x);
			break;

		default:
			cerr << "No valid activation function!" << endl;
			break;
	}
	return result;
}

double Neuron::activationFunctionDerivative(const double & x)
{
	double result = 0.0;

	switch (m_activationFunction) 
	{
		case SIGMOID:	// derivative of the sigmoid function
			result = sigmoid(x) * (1.0 - sigmoid(x));
			break;

		case TANH:		// derivative of the hyperbolic tangens function
			result = 1.0 - tanh(x) * tanh(x);
			break;

		case RELU:
			result = ReLU(x, true);
			break;

		case LEAKYRELU:
			result = LeakyReLU(x, true);
			break;

		default:
			cerr << "No valid activation function!" << endl;
			break;
	}
	return result;
}

double Neuron::sigmoid(const double &x)
{
	return 1.0 / (1 + exp(-x));
}

double Neuron::ReLU(const double &x, const bool &derivative)
{
	double result = 0.0;
	if (derivative) {
		if (x < 0)
			result = 0.0;
		else
			result = 1.0;
	}
	else {
		if (x < 0)
			result = 0.0;
		else
			result = x;
	}

	return result;
}

double Neuron::LeakyReLU(const double & x, const bool & derivative)
{
	double result = 0.0;
	double alphaParameter = 0.01;
	if (derivative) {
		if (x < 0)
			result = alphaParameter;
		else
			result = 1.0;
	}
	else {
		if (x < 0)
			result = alphaParameter * x;
		else
			result = x;
	}

	return result;
}

double Neuron::randomStartValue()
{
	return rand() / double(RAND_MAX);	// returns value between [0.0 .. 1.0]
}
