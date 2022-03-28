#include <cmath>
#include <ctime>
#include <string>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <iostream>
#define LOG(x) cout << x << endl
using namespace std;

static const float e = 2.718281828459045;

void PRINT(vector<float> x){
    cout << "[";
    for(int i = 0; i < x.size(); i++){
        if(i != x.size() - 1){
            cout << x[i] << ", ";
        }
        else{
            cout << x[i];
        }
    }
    cout << "]\n";
};

class Net{
    
public:
    float difference;                      // cost of last attempt
    
private:
    string function;                       // choice of ReLU, sigmoid, or leaky
    vector<int> layers;                    // list of numbers of layers
    vector<vector<float>> sums;            // list of pre-sigmoid activations
    vector<vector<float>> biases;          // list of bias lists
    vector<vector<float>> activations;     // list of activation lists
    vector<vector<vector<float>>> weights; // list of weight matrices -> insanity
    
public:
    Net(vector<int> totl){ // constructor
        difference = 100.0;
        layers = totl;
        sums.resize(totl.size() - 1, {});
        biases.resize(totl.size() - 1, {});
        for(int i = 0; i < totl.size() - 1; i++){
            biases[i].resize(totl[i + 1], 0.0);
        }
        activations.resize(totl.size(), {});
        weights.resize(totl.size() - 1, {});
        for(int i = 0; i < weights.size(); i++){
            weights[i].resize(totl[i + 1], {});
            for(int j = 0; j < weights[i].size(); j++){
                weights[i][j].resize(totl[i], 0.0);
            }
        }
        randomize();
    }
    vector<float> relu(vector<float> v){ // vector ReLU
        vector<float> temp = v;
        for(int i = 0; i < temp.size(); i++){
            if(temp[i] < 0){
                temp[i] = 0.0;
            }
        }
        return temp;
    }
    vector<float> relu_prime(vector<float> v){ // vector ReLU derivative
        vector<float> temp = v;
        for(int i = 0; i < temp.size(); i++){
            if(temp[i] <= 0){
                temp[i] = 0;
            }
            else if(temp[i] > 0){
                temp[i] = 1.0;
            }
        }
        return temp;
    }
    vector<float> sigmoid(vector<float> v){ // vector sigmoid
        vector<float> temp = v;
        for(int i = 0; i < temp.size(); i++){
            temp[i] = 1.0 / (1.0 + pow(e, -temp[i]));
        }
        return temp;
    }
    vector<float> sigmoid_prime(vector<float> v){ // vector sigmoid derivative
        vector<float> temp1 (v.size(), 1.0);
        vector<float> temp2 = sigmoid(v);
        return had(temp2, sub(temp1, temp2));
    }
    vector<float> softmax(vector<float> v){ // softmax
        vector<float> temp = v;
        float sum = 0.0;
        for(int i = 0; i < temp.size(); i++){
            float et = pow(e, temp[i]);
            temp[i] = et;
            sum += et;
        }
        for(int i = 0; i < temp.size(); i++){
            temp[i] = temp[i] / sum;
        }
        return temp;
    }
    vector<float> softmax_prime(vector<float> v){ // softmax derivative?
        vector<float> temp (v.size(), 1.0);
        return had(softmax(v), sub(temp, softmax(v)));
    }
    float cost(vector<float> output, vector<float> desired){ // finds cost of output
        float fin = 0;
        for(int i = 0; i < output.size(); i++){
            fin += pow(output[i] - desired[i], 2);
        }
        return fin;
    }
    vector<float> cost_prime(vector<float> output, vector<float> desired){ // finds derivative of cost of output
        vector<float> fin(output.size(), 0.0);
        for(int i = 0; i < output.size(); i++){
            fin[i] = output[i] - desired[i] < 0 ? -pow(abs(output[i] - desired[i]), 0.2) : pow(output[i] - desired[i], 0.2);
        }
        return fin;
    }
    float dot(const vector<float> &v1, const vector<float> &v2){ // vector multiplication
        float fin = 0;
        for(int i = 0; i < v1.size(); i++){
            fin += v1[i] * v2[i];
        }
        return fin;
    }
    vector<float> had(const vector<float> &v1, const vector<float> &v2){ // hadamard multiplication
        vector<float> fin(v1.size(), 0.0);
        for(int i = 0; i < v1.size(); i++){
            fin[i] = v1[i] * v2[i];
        }
        return fin;
    }
    vector<float> add(const vector<float> &v1, const vector<float> &v2){ // vector addition
        vector<float> fin (v1.size(), 0);
        for(int i = 0; i < v1.size(); i++){
            fin[i] = (v1[i] + v2[i]);
        }
        return fin;
    }
    vector<float> sub(const vector<float> &v1, const vector<float> &v2){ // vector subtraction
        vector<float> fin (v1.size(), 0);
        for(int i = 0; i < v1.size(); i++){
            fin[i] = v1[i] - v2[i];
        }
        return fin;
    }
    vector<float> scale(const vector<float> &v, const float &s){ // vector scaling
        vector<float> fin(v.size());
        for(int i = 0; i < fin.size(); i++){
            fin[i] = v[i] * s;
        }
        return fin;
    }
    vector<float> mult(const vector<vector<float>> &m, const vector<float> &v) { // matrix vector multiplication
        vector<float> fin (m.size(), 0);
        for(int i = 0; i < m.size(); i++){
            fin[i] = dot(m[i], v);
        }
        return fin;
    }
    vector<float> trans(const vector<vector<float>> &m, const vector<float> &v) { // transposed multiplication
        vector<float> fin (m[0].size(), 0);
        for(int i = 0; i < m[0].size(); i++){
            for(int j = 0; j < v.size(); j++){
                fin[i] += m[j][i] * v[j];
            }
        }
        return fin;
    }
    vector<vector<float>> mat(const vector<float> &w, const vector<float> &h){ // matrix
        vector<vector<float>> fin(h.size(), vector<float> {});
        for(int i = 0; i < h.size(); i++){
            fin[i] = scale(w, h[i]);
        }
        return fin;
    }
    vector<vector<float>> sum(const vector<vector<float>> &m1, const vector<vector<float>> &m2){ // matrix subtraction
        vector<vector<float>> fin = m1;
        for(int i = 0; i < fin.size(); i++){
            for(int j = 0; j < fin[i].size(); j++){
                fin[i][j] -= m2[i][j];
            }
        }
        return fin;
    }
    vector<vector<float>> product(const vector<vector<float>> &m, float s){ // matrix subtraction
        vector<vector<float>> fin = m;
        for(int i = 0; i < fin.size(); i++){
            for(int j = 0; j < fin[i].size(); j++){
                fin[i][j] *= s;
            }
        }
        return fin;
    }
    void randomize(){ // randomizes the network - used in initialization
        srand(static_cast<unsigned int>(time(0)));
        for(int i = 1; i < layers.size(); i++){
            for(int j = 0; j < layers[i]; j++){
                biases[i - 1][j] = ((static_cast<float>(rand() % 1000) + 1) / 1000.0 - 0.5) * 2;
                for(int k = 0; k < layers[i - 1]; k++){
                    weights[i - 1][j][k] = (static_cast<float>(rand() % 1000) + 1) / 1000.0 - 0.5;
                }
            }
        }
    }
    vector<float> teach(vector<float> input){ // forward propagation
        activations[0] = input;
        for(int i = 1; i < layers.size(); i++){
            sums[i - 1] = add(mult(weights[i - 1], activations[i - 1]), biases[i - 1]);
            activations[i] = (i == layers.size() - 1) ? sigmoid(sums[i - 1]) : sigmoid(sums[i - 1]);
        }
        return activations[activations.size() - 1];
    }
    void learn(vector<float> input, vector<float> desired){
        vector<vector<float>> bias_change (biases.size());
        vector<vector<vector<float>>> weight_change (weights.size());
        teach(input);
        difference = cost(activations[activations.size() - 1], desired);
        vector<float> delta = had(cost_prime(activations[activations.size() - 1], desired), sigmoid_prime(sums[sums.size() - 1]));
        bias_change[bias_change.size() - 1] = delta;
        weight_change[weight_change.size() - 1] = mat(activations[activations.size() - 2], delta);
        for(int l = 2; l <= layers.size() - 1; l++){
            delta = had(trans(weights[weights.size() - l + 1], delta), sigmoid_prime(sums[sums.size() - l]));
            bias_change[bias_change.size() - l] = delta;
            weight_change[weight_change.size() - l] = mat(activations[activations.size() - l - 1], delta);
        }
        biases = product(sum(biases, bias_change), 1);
        for(int i = 0; i < weights.size(); i++){
            weights[i] = product(sum(weights[i], weight_change[i]), 1);
        }
    }
    void print(bool out = false){
        if(!out){
            cout << "Last Cost: " << difference << "\n\n";
            LOG("Biases:");
            for(int i = 0; i < biases.size(); i++){
                PRINT(biases[i]);
            }
            LOG("");
            LOG("Sums:");
            for(int i = 0; i < sums.size(); i++){
                PRINT(sums[i]);
            }
            LOG("");
            LOG("Activations:");
            for(int i = 0; i < activations.size(); i++){
                PRINT(activations[i]);
            }
            LOG("");
            LOG("Weights:");
            for(int i = 0; i < weights.size(); i++){
                for(int j = 0; j < weights[i].size(); j++){
                    PRINT(weights[i][j]);
                }
                LOG("");
            }
        }
        else{
            LOG("Input:");
            PRINT(activations[0]);
            LOG("");
            LOG("Output:");
            PRINT(activations[activations.size() - 1]);
            LOG("");
        }
    }
    
    
    
};

bool isPrime(int n){
    if(n <= 3){
        return n > 1;
    }
    else if(n % 2 == 0 || n % 3 == 0){
        return false;
    }
    int k = 5;
    while(k * k <= n){
        if(n % k == 0 || n % (k + 2) == 0){
            return false;
        }
        k += 6;
    }
    return true;
}

vector<float> bin(unsigned int num){
    vector<float> fin (32);
    unsigned int temp = num;
    for(int i = 0; i < 32; i++){
        fin[31 - i] = static_cast<float>(temp & 1);
        temp = (temp >> 1) | 0;
    }
    return fin;
};

int main() {
    int counter = 0;
    vector<float> oops = {0.0, 0.0};
    Net primeTester(vector<int> {32, 50, 40, 30, 20, 10, 2});
    
    while(primeTester.difference > 0.00001 && counter < 100000){
        
        unsigned int num = rand() % 10;
        
        vector<float> tst1 = bin(num);
        vector<float> tst2;
        
        if(isPrime(num)){
            tst2 = {1.0, 0.0};
            oops[0]++;
        }
        else{
            tst2 = {0.0, 1.0};
            oops[1]++;
        }
        
        primeTester.learn(tst1, tst2);
        counter++;
        LOG(primeTester.difference);
        
    }
    
    LOG(counter);
    PRINT(oops);
    
    for(int i = 0; i < 10; i++){
        primeTester.teach({static_cast<float>(i)});
        primeTester.print(true);
    }
    
    return 0;
}
