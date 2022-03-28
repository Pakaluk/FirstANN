/*
 Notes:
 
 Evolutionary Training Idea:
 Create a population of random networks. Allow only top X% to
 reproduce (change them slightly in random directions according
 to difference)
 
 Random Walker Training Idea:
 Create one random network. Test it. Try a random slightly changed
 version with the same input. If it's cost is lower, keep it.
 If not, try again. Continue until the cost lowers.
 
 This version forgoes the node structure completely to facilitate
 high amounts of epicly confusing loops and (hopefully) more
 efficient and consolidated code.
 
 Looping through multiple vectors with multiple vectors is confusing.
 Period.
 
 After implementing backpropagation at last, the network learns
 with ease (although ReLU is giving me quite a bit of trouble)
 
 What should I test it on??
 
 Name: ANNA (Artificial Neural Network A)
 
 Copyright Nicholas Pakaluk 10.11.2020
 
 Second Attempt
*/


#include <cmath>
#include <ctime>
#include <string>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <iostream>
#define LOG(x) std::cout << x << std::endl
#define SWAP(x) ((x << 24) | (x << 8 & 0xff0000) | (x >> 8 & 0xff00) | (x >> 24))
using namespace std;
void PRINT(vector<float>);

class Net{
    
public:
    float difference;                      // cost of last attempt
    
private:
    string function;                       // choice of ReLU, sigmoid, or leaky
    vector<int> layers;                    // list of numbers of layers
    vector<vector<float>> sums;            // list of pre-sigmoid activations
    vector<vector<float>> biases;          // list of bias lists
    vector<vector<float>> activations;     // list of activation lists
    vector<vector<vector<float>>> weights; // list of weight matrices
    float e = 2.718281828459045;
    
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
        return scale(sub(output, desired), 1);
        /*vector<float> fin(output.size(), 0.0);
        for(int i = 0; i < output.size(); i++){
            fin[i] = output[i] - desired[i] < 0 ? -pow(abs(output[i] - desired[i]), 0.2) : pow(output[i] - desired[i], 0.2);
        }
        return fin;*/
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
    vector<vector<float>> ad(const vector<vector<float>> &m1, const vector<vector<float>> &m2){ // matrix subtraction
        vector<vector<float>> fin = m1;
        for(int i = 0; i < fin.size(); i++){
            for(int j = 0; j < fin[i].size(); j++){
                fin[i][j] += m2[i][j];
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
                biases[i - 1][j] = ((static_cast<float>(rand() % 1000)) / 1000.0 - 0.5);
                for(int k = 0; k < layers[i - 1]; k++){
                    weights[i - 1][j][k] = ((static_cast<float>(rand() % 1000)) / 1000.0 - 0.5);
                }
            }
        }
    }
    vector<float> teach(vector<float> input, vector<float> desired){ // forward propagation
        activations[0] = input;
        for(int i = 1; i < layers.size(); i++){
            sums[i - 1] = add(mult(weights[i - 1], activations[i - 1]), biases[i - 1]);
            activations[i] = sigmoid(sums[i - 1]);
        }
        difference = cost(activations[activations.size() - 1], desired);
        return activations[activations.size() - 1];
    }
    void learn(vector<float> input, vector<float> desired){
        vector<vector<float>> bias_change (biases.size());
        vector<vector<vector<float>>> weight_change (weights.size());
        teach(input, desired);
        vector<float> delta = had(cost_prime(activations[activations.size() - 1], desired), sigmoid_prime(sums[sums.size() - 1]));
        bias_change[bias_change.size() - 1] = delta;
        weight_change[weight_change.size() - 1] = mat(activations[activations.size() - 2], delta);
        for(int l = 2; l <= layers.size() - 1; l++){
            delta = had(trans(weights[weights.size() - l + 1], delta), sigmoid_prime(sums[sums.size() - l]));
            bias_change[bias_change.size() - l] = delta;
            weight_change[weight_change.size() - l] = mat(activations[activations.size() - l - 1], delta);
        }
        biases = sum(biases, bias_change);
        for(int i = 0; i < weights.size(); i++){
            weights[i] = sum(weights[i], weight_change[i]);
        }
    }
    void batch_learn(vector<vector<float>> input, vector<vector<float>> desired){
        vector<vector<float>> bias_change (biases);
        vector<vector<vector<float>>> weight_change (weights);
        for(int i = 1; i < layers.size(); i++){
            for(int j = 0; j < layers[i]; j++){
                bias_change[i - 1][j] = 0.0;
                for(int k = 0; k < layers[i - 1]; k++){
                    weight_change[i - 1][j][k] = 0.0;
                }
            }
        }
        difference = 0.0;
        for(int i = 0; i < input.size(); i++){
            activations[0] = input[i];
            for(int i = 1; i < layers.size(); i++){
                sums[i - 1] = add(mult(weights[i - 1], activations[i - 1]), biases[i - 1]);
                activations[i] = sigmoid(sums[i - 1]);
            }
            difference += cost(activations[activations.size() - 1], desired[i]);
            vector<float> delta = had(cost_prime(activations[activations.size() - 1], desired[i]), sigmoid_prime(sums[sums.size() - 1]));
            bias_change[bias_change.size() - 1] = add(bias_change[bias_change.size() - 1], delta);
            weight_change[weight_change.size() - 1] = mat(activations[activations.size() - 2], delta);
            for(int l = 2; l <= layers.size() - 1; l++){
                delta = had(trans(weights[weights.size() - l + 1], delta), sigmoid_prime(sums[sums.size() - l]));
                bias_change[bias_change.size() - l] = add(bias_change[bias_change.size() - l], delta);
                weight_change[weight_change.size() - l] = ad(weight_change[weight_change.size() - l], mat(activations[activations.size() - l - 1], delta));
            }
        }
        /*biases = sum(biases, product(bias_change, 1.0f / float(input.size())));
        for(int i = 0; i < weights.size(); i++){
            weights[i] = sum(weights[i], product(weight_change[i], 1.0f / float(input.size())));
        }*/
        biases = sum(biases, bias_change);
        for(int i = 0; i < weights.size(); i++){
            weights[i] = sum(weights[i], weight_change[i]);
        }
    }
    int winner(){
        int index = 0;
        float temp = activations[activations.size() - 1][0];
        for(int i = 1; i < activations[activations.size() - 1].size(); i++){
            if(activations[activations.size() - 1][i] > temp){
                temp = activations[activations.size() - 1][i];
                index = i;
            }
        }
        return index;
    }
    int affirmative(){
        return activations[activations.size() - 1][0] > 0.5 ? 1 : 0;
    }
    float act(){
        return activations[activations.size() - 1][0];
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
                    /*cout << "[";
                    for(int i = 0; i < weights[i][j].size(); i++){
                        if(i != weights[i][j].size() - 1){
                            cout << weights[i][j][i] << ", ";
                        }
                        else{
                            cout << weights[i][j][i];
                        }
                    }
                    cout << "],\n";*/
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
    cout << "],\n";
}
bool thresh(float cost, float threshhold){
    static float past[10];
    for(int i = 9; i > 0; i--){
        past[i] = past[i - 1];
    }
    past[0] = cost;
    for(int i = 0; i < 10; i++){
        if(past[i] > threshhold)
            return true;
    }
    return false;
}
void draw(vector<float> v){
    char shade[11] = {' ', '.',',','~','-','+','=','*','&','@','#'};
    for(int i = 0; i < 784; i++){
        cout << shade[int(floor(v[i] * 10.0))];
        if(i % 28 == 27)
            cout << endl;
    }
}

// shifts useful data to the upper left corner of the image
vector<float> shift(vector<float> d){
    float s[2] = {500, 500};
    vector<float> fin = d;
    for(float y = 0; y < 400; y += 400 / 28 + 0.00001){
        for(float x = 0; x < 400; x += 400 / 28 + 0.00001){
            int l = round((y * 28 + x) / (400 / 28));
            if(d[l] > 0){
                if(x < s[0]){
                    s[0] = x;
                }
                if(y < s[1]){
                    s[1] = y;
                }
            }
        }
    }
    for(float y = 0; y < 400; y += 400 / 28 + 0.00001){
        for(float x = 0; x < 400; x += 400 / 28 + 0.00001){
            int l1 = round((y * 28 + x) / (400 / 28));
            int l2 = round(((y + s[1]) * 28 + (x + s[0])) / (400 / 28));
            if(x < 400 - s[0] && y < 400 - s[1]){
                d[l1] = d[l2];
            }
            else{
                d[l1] = 0;
            }
        }
    }
    return fin;
}

// reads training data from binary files
void get_train_data(vector<vector<float>> & images, vector<vector<float>> & labels, int & length){
    ifstream train_labels("train-labels.idx1-ubyte", ios::in | ios::binary);
    ifstream train_images("train-images.idx3-ubyte", ios::in | ios::binary);
    
    int burner, size;
    train_images.read(reinterpret_cast<char*>(&burner), 4);
    train_images.read(reinterpret_cast<char*>(&size), 4);
    for(int i = 0; i < 2; i++){
        train_images.read(reinterpret_cast<char*>(&burner), 4);
        train_labels.read(reinterpret_cast<char*>(&burner), 4);
    }
    size = SWAP(size);
    
    images.clear();
    labels.clear();
    
    images.reserve(size);
    labels.reserve(size);
    
    for(int i = 0; i < size; i++){
        vector<float> example(784, 0.0);
        vector<float> correct(10, 0.0);
        
        char num;
        train_labels.read(reinterpret_cast<char*>(&num), 1);
        correct[static_cast<int>(num)] = 1.0;
        
        for(int j = 0; j < 784; j++){
            int col;
            train_images.read(reinterpret_cast<char*>(&col), 1);
            example[j] = static_cast<float>(col & 0xff) / 255.0f;
        }
        
        images.emplace_back(example);
        labels.emplace_back(correct);
    }
    length = size;
    
    train_images.close();
    train_labels.close();
}

// reads testing data from binary files
void get_test_data(vector<vector<float>> & images, vector<vector<float>> & labels, int & length){
    ifstream test_labels("t10k-labels.idx1-ubyte", ios::in | ios::binary);
    ifstream test_images("t10k-images.idx3-ubyte", ios::in | ios::binary);
    
    int burner, size;
    test_images.read(reinterpret_cast<char*>(&burner), 4);
    test_images.read(reinterpret_cast<char*>(&size), 4);
    for(int i = 0; i < 2; i++){
        test_images.read(reinterpret_cast<char*>(&burner), 4);
        test_labels.read(reinterpret_cast<char*>(&burner), 4);
    }
    size = SWAP(size);
    
    images.clear();
    labels.clear();
    
    images.reserve(size);
    labels.reserve(size);
    
    for(int i = 0; i < size; i++){
        vector<float> example(784, 0.0);
        vector<float> correct(10, 0.0);
        
        char num;
        test_labels.read(reinterpret_cast<char*>(&num), 1);
        correct[static_cast<int>(num)] = 1.0;
        
        for(int j = 0; j < 784; j++){
            int col;
            test_images.read(reinterpret_cast<char*>(&col), 1);
            example[j] = static_cast<float>(col & 0xff) / 255.0f;
        }
        
        images.emplace_back(example);
        labels.emplace_back(correct);
    }
    length = size;
    
    test_images.close();
    test_labels.close();
}

// randomize a vector
void shuffle(vector<vector<float>> & v){
    
}

void general_purpose(){
    int counter = 0, epoch = 1;
    Net test({784, 15, 10});
    
    // save all epochs
    // shuffle
    // bigger net later
    // [OpenGL | Cuda | Vulkan | Metal]: use GPU!
    
    int train_length, test_length;
    vector<vector<float>> train_images, train_labels, test_images, test_labels;
    get_train_data(train_images, train_labels, train_length);
    get_test_data(test_images, test_labels, test_length);

    for(int k = 0; k < epoch; k++){
        
        float stats[2] = {0, 0};
        
        for(int i = 0; i < train_length; i++){
            
            //for(int j = 0; j < batch; j++)
                
            test.learn(train_images[i], train_labels[i]);
            counter++;
            cout << "Training batch: " << counter << "   Cost: " << test.difference << endl;
            
        }
        counter = 0;
        
        cout << "\n\nTraining Complete.\n\n\n";
        
        for(int i = 0; i < test_length; i++){
            
            vector<float> winner = test.teach(test_images[i], test_labels[i]);
            
            int num = 0;
            for(int j = 0; j < test_labels[i].size(); j++){
                if(test_labels[i][j] == 1.0){
                    num = j;
                }
            }
            
            if(test_labels[i][test.winner()] == 1.0){
                stats[1]++;
            }
            else if(test_labels[i][test.winner()] == 0.0){
                stats[0]++;
                cout << "\nNetwork choice: " << test.winner();
                cout << "   Correct choice: " << num << "\n";
                draw(test_images[i]);
            }
            counter++;
            cout << "Testing example: " << counter << "   Cost: " << test.difference << endl;
        }
        counter = 0;
        
        cout << endl;// << "[" << stats[0] << ", " << stats[1] << "]\n" << endl;
        cout << "Epoch: " << k + 1 << endl;
        cout << stats[1] / 100.0f << "% correct, ";
        cout << stats[0] / 100.0f << "% incorrect\n" << endl;
        
        //test.print();
    }
    
}
void train_on_tests(){
    int counter = 0, batch = 10;
    Net test({784, 50, 25, 10});
    
    int stats[2] = {0, 0};
    
    ifstream train_labels("train-labels.idx1-ubyte", ios::in | ios::binary);
    ifstream train_images("train-images.idx3-ubyte", ios::in | ios::binary);
    
    int burner, size;
    train_images.read(reinterpret_cast<char*>(&burner), 4);
    train_images.read(reinterpret_cast<char*>(&size), 4);
    for(int i = 0; i < 2; i++){
        train_images.read(reinterpret_cast<char*>(&burner), 4);
        train_labels.read(reinterpret_cast<char*>(&burner), 4);
    }
    
    vector<vector<float>> examples(batch);
    vector<vector<float>> corrects(batch);
    for(int i = 0; i < SWAP(size); i++){
        
        vector<float> example(784, 0.0);
        vector<float> correct(10, 0.0);
        
        char num;
        train_labels.read(reinterpret_cast<char*>(&num), 1);
        correct[static_cast<int>(num)] = 1.0;
        corrects[i % batch] = correct;
        
        for(int j = 0; j < 784; j++){
            int col;
            train_images.read(reinterpret_cast<char*>(&col), 1);
            example[j] = static_cast<float>(col & 0xff) / 255.0f;
        }
        examples[i % batch] = shift(example);
        
        if(i % batch == batch - 1){
            test.batch_learn(examples, corrects);
            examples.clear();
            examples.resize(batch);
            corrects.clear();
            corrects.resize(batch);
            counter++;
            cout << "Training batch: " << counter << "   Cost: " << test.difference << endl;
        }
        
    }
    train_images.close();
    train_labels.close();
    counter = 0;
    
    
    cout << "\n\nTraining Complete.\n\n\n";
    
    
    ifstream test_labels("t10k-labels.idx1-ubyte", ios::in | ios::binary);
    ifstream test_images("t10k-images.idx3-ubyte", ios::in | ios::binary);
    
    test_images.read(reinterpret_cast<char*>(&burner), 4);
    test_images.read(reinterpret_cast<char*>(&size), 4);
    for(int i = 0; i < 2; i++){
        test_images.read(reinterpret_cast<char*>(&burner), 4);
        test_labels.read(reinterpret_cast<char*>(&burner), 4);
    }
    
    for(int i = 0; i < SWAP(size); i++){
        
        vector<float> example(784);
        vector<float> correct(10, 0.0);
        
        char num;
        test_labels.read(reinterpret_cast<char*>(&num), 1);
        correct[static_cast<int>(num)] = 1.0;
        
        for(int j = 0; j < 784; j++){
            int col;
            test_images.read(reinterpret_cast<char*>(&col), 1);
            example[j] = static_cast<float>(col & 0xff) / 255.0f;
        }
        
        test.learn(shift(example), correct);
        if(correct[test.winner()] == 1.0){
            stats[1]++;
        }
        else if(correct[test.winner()] == 0.0){
            stats[0]++;
            cout << "\nNetwork choice: " << test.winner();
            cout << "   Correct choice: " << static_cast<int>(num) << "\n";
            draw(shift(example));
        }
        counter++;
        cout << "Testing example: " << counter << "   Cost: " << test.difference << endl;
    }
    test_images.close();
    test_labels.close();
    counter = 0;
    
    cout << endl;// << "[" << stats[0] << ", " << stats[1] << "]\n" << endl;
    cout << stats[1] / 100.0f << "% correct, ";
    cout << stats[0] / 100.0f << "% incorrect\n" << endl;
    
    //test.print();
    
}
void seperate_purpose(){
    int counter = 0;
    vector<Net> nets;
    for(int i = 0; i < 10; i++){
        Net temp = Net({784, 10, 1});
        nets.emplace_back(temp);
    }
    
    int stats[2] = {0, 0};
    
    ifstream train_labels("train-labels.idx1-ubyte", ios::in | ios::binary);
    ifstream train_images("train-images.idx3-ubyte", ios::in | ios::binary);
    
    int burner, size;
    train_images.read(reinterpret_cast<char*>(&burner), 4);
    train_images.read(reinterpret_cast<char*>(&size), 4);
    for(int i = 0; i < 2; i++){
        train_images.read(reinterpret_cast<char*>(&burner), 4);
        train_labels.read(reinterpret_cast<char*>(&burner), 4);
    }
    
    //vector<vector<float>> examples(batch);
    //vector<vector<float>> corrects(batch);
    for(int i = 0; i < SWAP(size); i++){
        
        vector<float> example(784, 0.0);
        vector<float> correct(10, 0.0);
        
        char num;
        train_labels.read(reinterpret_cast<char*>(&num), 1);
        correct[static_cast<int>(num)] = 1.0;
        //corrects[i % batch] = correct;
        
        for(int j = 0; j < 784; j++){
            int col;
            train_images.read(reinterpret_cast<char*>(&col), 1);
            example[j] = static_cast<float>(col & 0xff) / 255.0f;
        }
        //examples[i % batch] = example;
        
        for(int k = 0; k < 10; k++){
            nets[k].learn(example, {correct[k]});
        }
        
        //if(i % batch == batch - 1){
            //test.batch_learn(examples, corrects);
            //examples.clear();
            //examples.resize(batch);
            //corrects.clear();
            //corrects.resize(batch);
            counter++;
        
        float tp = 0.0;
        for(int l = 0; l < 10; l++){
            tp += nets[l].difference;
        }
        
            cout << "Training sample: " << counter << "   Cost: " << tp << endl;
        //}
        
    }
    train_images.close();
    train_labels.close();
    counter = 0;
    
    
    cout << "\n\nTraining Complete.\n\n\n";
    
    
    ifstream test_labels("t10k-labels.idx1-ubyte", ios::in | ios::binary);
    ifstream test_images("t10k-images.idx3-ubyte", ios::in | ios::binary);
    
    test_images.read(reinterpret_cast<char*>(&burner), 4);
    test_images.read(reinterpret_cast<char*>(&size), 4);
    for(int i = 0; i < 2; i++){
        test_images.read(reinterpret_cast<char*>(&burner), 4);
        test_labels.read(reinterpret_cast<char*>(&burner), 4);
    }
    
    for(int i = 0; i < SWAP(size); i++){
        
        vector<float> example(784);
        vector<float> correct(10, 0.0);
        
        char num;
        test_labels.read(reinterpret_cast<char*>(&num), 1);
        correct[static_cast<int>(num)] = 1.0;
        
        for(int j = 0; j < 784; j++){
            int col;
            test_images.read(reinterpret_cast<char*>(&col), 1);
            example[j] = static_cast<float>(col & 0xff) / 255.0f;
        }
        
        for(int k = 0; k < 10; k++){
            nets[k].teach(example, {correct[k]});
        }
        
        int winner = 0;
        float temp = nets[0].act();
        for(int l = 1; l < 10; l++){
            if(nets[l].act() > temp){
                temp = nets[l].act();
                winner = l;
            }
        }
        
        if(correct[winner] == 1.0){
            stats[1]++;
        }
        else if(correct[winner] == 0.0){
            stats[0]++;
            cout << "\nNetwork choice: " << winner;
            cout << "   Correct choice: " << static_cast<int>(num) << "\n";
            draw(example);
        }
        counter++;
        float tp = 0.0;
        for(int l = 0; l < 10; l++){
            tp += nets[l].difference;
        }
        cout << "Testing example: " << counter << "   Cost: " << tp << endl;
    }
    test_images.close();
    test_labels.close();
    counter = 0;
    
    cout << endl;
    cout << stats[1] / 100.0f << "% correct, ";
    cout << stats[0] / 100.0f << "% incorrect\n" << endl;
}

int main(){
    general_purpose();
    
    //train_on_tests();
    
    //seperate_purpose();
    
    return (0400 - 0xff) << 8;
}

/*
 
 Time trials:
 
 N. [input, hidden layers, output]; test samples; CPU/GPU; hh.mm.ss
 
 1. [750, 2500, 2000, 1500, 1000, 500, 10]; 1000; CPU; 00.13.45
 
 2.
 
 */
