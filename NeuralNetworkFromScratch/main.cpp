#include "NeuralNet.h"
#include <string>

void showVectorValues( vector<double> &output, vector<double> &target) {
	if (output.size() == target.size()) {
		for (unsigned int i = 0; i < output.size(); i++) {
			cout << "Output" << i << ": " << output[i] << "; Target" << i << ": " << target[i] << endl;
		}
		cout << endl;
	}
	else {
		cout << "Output vector size != target vector size" << endl;
	}
	
}

int main() {
	
	unsigned int numberOfCycles = 10000;
	vector<unsigned int> dataStructure{ 2, 1 };		// inputs, outputs
	vector<unsigned int> hiddenLayerStructure{ 4 };		// <-- change hidden-layer-structure here

	vector<vector<double>> inputValues{ {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0} };
	vector<vector<double>> targetValues{ {0.0}, {1.0}, {1.0}, {0.0} };

	// dataStructure is a vector with { numberOfInputs, numberOfOutputs }
	// hiddenLayerStructure is a vector with { numberOfNeuronsLayer1, numberOfNeuronsLayer2, ... }, hiddenLayerStructure.size() == number of hidden Layers
	NeuralNet myNeuralNet(dataStructure, hiddenLayerStructure);

	vector<double> results;
	vector<double> fullResults;
	vector<double> fullTargetValues{ targetValues[0][0], targetValues[1][0], targetValues[2][0], targetValues[3][0] };	// ToDo: better solution

	for (unsigned int cycle = 0; cycle < numberOfCycles; cycle++) {
		fullResults.clear();
		for (unsigned int currentData = 0; currentData < inputValues.size(); currentData++) {
			myNeuralNet.feedForward(inputValues[currentData]);
			
			myNeuralNet.getResults(results);
			fullResults.push_back(results[0]);

			myNeuralNet.backPropagation(targetValues[currentData]);
		}
		if (cycle % 500 == 0) {
			cout << "Cycle: " << cycle << endl;
			showVectorValues(fullResults, fullTargetValues);
		}
	}
	cout << "Final results:" << endl;
	showVectorValues(fullResults, fullTargetValues);

	char keepConsoleOpen;
	cin >> keepConsoleOpen;

	return 0;
}