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
	vector<vector<double>> biases; //biases[i][j] = bias of jth node of ith layer
	vector<vector<double>> z; //z[i][j] = Sigma(weight * activPrev) + bias = z-val of jth node of ith layer; z-val for the first layer is undefined
	vector<vector<double>> delta;
	vector<vector<vector<double>>> weights; //weights[i][j][k] = weight of the edge from node j (ith layer) to node k (i+1 th layer)
	vector<int> countOfNodesInLayers;

	NeuralNetwork(vector<int> countOfNodesInLayers) {
		this->countOfNodesInLayers = countOfNodesInLayers;
		layers = countOfNodesInLayers.size();

		//Initialize activations
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
	}

	void dotProd(vector<double>& ip, vector<vector<double>>& weights, vector<double>& op) {
		//obsolete
		if (ip.size() != weights[0].size()) {
			cout << "Illegal Dot Product!";
			return;
		}
		for (int opLayerNode = 0; opLayerNode < op.size(); opLayerNode++) {
			double temp = 0;
			for (int ipLayerNode = 0; ipLayerNode < ip.size(); ipLayerNode++)
				temp += weights[opLayerNode][ipLayerNode] * ip[ipLayerNode];
			op[opLayerNode] = temp;
		}
	}
	double ReLU(double input) {
		return max((double)0, input);
	}
	double ReLUDiff(double input) {
		return input >= 0;
	}
	double sqFunc(double expected, double real) {
		return (expected - real) * (expected - real);
	}
	double sqFuncDiff(double expected, double real) {
		return (expected - real) * 2;
	}

	double activFunc(double input) {
		//Activation Function
		return ReLU(input);
	}
	double activFuncDiff(double input) {
		//differentiation of the Activation Function
		return ReLUDiff(input);
	}
	double costFunc(double expected, double real) {
		return sqFunc(expected, real);
	}
	double costFuncDiff(double expected, double real) {
		return sqFuncDiff(expected, real);
	}

public:
	vector<double> processInput(vector<double> input) {
		activations[0] = input;
		for (int i=1; i<layers; i++) {
			for (int j=0; j<activations[i].size(); i++) {
				double temp = 0;
				for (int k=0; k<activations[i-1].size(); k++) temp += weights[i-1][k][j] * activations[i-1][k];
				z[i][j] = temp + biases[i][j];
				activations[i][j] = activFunc(z[i][j]);
			}
		}
		return activations.back();
	}
	void backPropagation(vector<double>& expectedValues) {
		assert(expectedValues.size() == weights.back().size());
		//initialize deltas for the last layer
		for (int i=0; i<expectedValues.size(); i++)
			 deltas[layers-1][i] = costFuncDiff(expected[i], expectedValues[i]) * activFuncDiff(z[layers-1][i]);

		//find for the rest of the layers
		for (int i=layers-2; i>=0; i--) {
			for (int node = 0; node < countOfNodesInLayers[i]; node++) {
				
			}
		}
	}
	void showActivations() {
		for (int i=0; i<layers; i++) {
			cout << "Layer " << i+1 << ": ";
			for (double& x: activations[i]) cout << x << ' ';
			cout << endl;
		}
		cout << endl;
	}
	void showWeights() {
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

