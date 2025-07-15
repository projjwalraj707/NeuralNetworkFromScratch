#include <vector>
#include <random>
#include <iostream>

using namespace std;

//generate random values 
random_device rd;
mt19937 gen(rd());
uniform_real_distribution<> dist(-1.0, 1.0);

class NeuralNetwork {
private:
	int layers; //count of the number of layers in the Neural Network
	vector<vector<double>> activations; //activations[i][j] = activation of jth node of ith layer
	vector<vector<double>> biases; //biases[i][j] = bias of jth node of ith layer; It is UNDEFINED (as verified from muliple sources) for the input layer
	vector<vector<double>> z; //z[i][j] = Summation(weight * activPrev) + bias = z-val of jth node of ith layer; z-val for the first layer is undefined
	vector<vector<double>> deltas; //deltas[i][j] = delta value of the jth node of the ith layer. UNDEFINED for the input layer. deltas are coefficients. Delta of a node when multiplied by the activation of any prev-layer node gives dC/dw. C: Cost, w: weight between the current node and the prev-layer node. It is equal to dC/dbias.
	vector<vector<vector<double>>> weights; //weights[i][j][k] = weight of the edge from node j (ith layer) to node k (i+1 th layer)
	vector<int> countOfNodesInLayers;

public:
	NeuralNetwork(vector<int> countOfNodesInLayers) {
		this->countOfNodesInLayers = countOfNodesInLayers;
		layers = countOfNodesInLayers.size();

		//Initialize activations
		activations = vector<vector<double>> (layers);
		for (int i=0; i<layers; i++) activations[i] = vector<double>(countOfNodesInLayers[i]);
		randomizeValues(activations);

		//Initialize biases, z, deltas. They have the same dimensions as weights
		biases = z = deltas = activations;
		randomizeValues(z);
		//biases are generally initialized with zeroes.
		//deltas need not be initialized
		
		//Initialize weights
		weights = vector<vector<vector<double>>> (layers-1);
		for (int i=0; i<layers-1; i++) {
			weights[i] = vector<vector<double>>(countOfNodesInLayers[i], vector<double>(countOfNodesInLayers[i+1]));
		}
		randomizeValues(weights);

		/*
		for (int currLayer = 0; currLayer < layers; currLayer++) {
			vector<double> temp;
			for (int i=0; i<countOfNodesInLayers[currLayer]; i++) temp.push_back(dist(gen));
			activations.push_back(temp);
		}

		//Initialize weights
		for (int i=0; i<layers-1; i++) {
			vector<vector<double>> temp1;
			for (int j=0; j<countOfNodesInLayers[i+1]; j++) {
				vector<double> temp2;
				for (int k=0; k<countOfNodesInLayers[i]; k++) {
					temp2.push_back(dist(gen));
				}
				temp1.push_back(temp2);
			}
			weights.push_back(temp1);
		}

		z = biases = delta = activations; //dimensions of all these are the same (ignore the values being copied)
		*/
	}

	void randomizeValues(vector<double>& arr) {
		for (double& x: arr) x = dist(gen);
	}
	void randomizeValues(vector<vector<double>>& arr) {
		for (vector<double>& x: arr) randomizeValues(x);
	}
	void randomizeValues(vector<vector<vector<double>>>& arr) {
		for (vector<vector<double>>& x: arr) randomizeValues(x);
	}
	double ReLU(double input) {
		return max((double)0, input);
	}
	double ReLUDiff(double input) {
		return input >= 0;
	}
	double sqFunc(double expected, double actual) {
		return (expected - actual) * (expected - actual);
	}
	double sqFuncDiff(double expected, double actual) {
		return (expected - actual) * 2;
	}
	double activFunc(double input) {//Activation Function
		return ReLU(input);
	}
	double activFuncDiff(double input) {//differentiation of the Activation Function
		return ReLUDiff(input);
	}
	double costFunc(double expected, double actual) {
		return sqFunc(expected, actual);
	}
	double costFuncDiff(double expected, double actual) {
		return sqFuncDiff(expected, actual);
	}

public:
	vector<double> processInput(vector<double> input) {
		activations[0] = input;
		for (int l=1; l<layers; l++) {
			for (int node2=0; node2<activations[l].size(); node2++) {
				double temp = 0;
				for (int node1=0; node1<activations[l-1].size(); node1++) temp += weights[l-1][node1][node2] * activations[l-1][node1];
				z[l][node2] = temp + biases[l][node2];
				activations[l][node2] = activFunc(z[l][node2]);
			}
		}
		return activations.back();
	}
	void backPropagation(vector<double>& expectedValues) {
		//initialize deltas for the last layer
		for (int node=0; node<expectedValues.size(); node++)
			 deltas[layers-1][node] = costFuncDiff(expectedValues[node], activations[layers-1][node]) * activFuncDiff(z[layers-1][node]);

		//find for the rest of the layers
		for (int l=layers-2; l>=0; l--) {
			for (int node1 = 0; node1 < countOfNodesInLayers[l]; node1++) {
				double temp = 0;
				for (int node2 = 0; node2 < countOfNodesInLayers[l+1]; node2++) {
					temp += weights[l][node1][node2] * deltas[l+1][node2];
				}
				temp *= activFuncDiff(z[l][node1]);
				deltas[l][node1] = temp;
			}
		}
		//verified till here
	}
	void showActivations() {
		cout << "Printing Activations for each layer:\n";
		for (int i=0; i<layers; i++) {
			cout << "Layer " << i+1 << ": ";
			for (double& x: activations[i]) cout << x << ' ';
			cout << endl;
		}
		cout << endl;
	}
	void showWeights() {
		cout << "Printing Weights for each edge\n\n";
		for (int i=0; i<layers-1; i++) {
			cout << "Weights between " << i+1 << "th and " << i+2 << "th layers" << endl;
			for (vector<double>& x: weights[i]) {
				for (double y: x) cout << y << ' ';
				cout << endl;
			}
			cout << endl;
		}
		cout << endl;
	}
};

int main() {
	NeuralNetwork NN({4,2,3});
	NN.showActivations();
	NN.showWeights();
	return 0;
}

