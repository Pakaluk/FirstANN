/*
 Copyright Nicholas Pakaluk 10.10.2020
 First Attempt
*/

#include <cmath>
#include <string>
#include <vector>
#include <iostream>
#define LOG(x) cout << x << endl
using namespace std;

// linear algebra stuff
float RELU(float x){
    if(x > 0){
        return x;
    }
    return 0;
};
/*float dot(vector<float> v1, vector<float> v2){
    float fin = 0;
    for(int i = 0; i < v1.size(); i++){
        fin += v1[i] * v2[i];
    }
    return fin;
};
vector<float> matMult(vector<vector<float>> m, vector<float> v){
    vector<float> fin;
    fin.reserve(v.size());
    for(int i = 0; i < v.size(); i++){
        fin[i] = dot(m[i], v);
    }
    return fin;
}*/
void copy(vector<float> &a, vector<float> &b){
    a.reserve(b.size());
    for(int i = 0; i < b.size(); i++){
        a[i] = b[i];
    }
}

// node structure
struct Node {
    
public:
    int type;
    vector<float> weight;
    float activation, bias;
    
public:
    Node(int t, float b, float a, vector<float> w){
        type = t;
        bias = b;
        copy(weight, w);
        activation = a;
    }
    
    void setAttributes(float b, vector<float> w){
        bias = b;
        weight = w;
    }
    
    void activate(float a){
        activation = a;
    }
    
};

// network class
class Net {
    
private:
    int size;
    vector<Node> input;
    vector<vector<Node>> layers;
    vector<Node> output;
    
public:
    Net(int s){
        size = s;
        LOG("The termination of mankind has arrived!");
    }
    ~Net(){
        LOG("No! SkyNet has fallen!");
    }
    
    void target(Node &node){
        if(size > 2){
            if(node.type == 0){
                LOG("lol why are you doing this?");
            }
            else if(node.type == 1){
                float fin = 0;
                for(int i = 0; i < input.size(); i++){
                    fin += input[i].activation * node.weight[i];
                }
                node.activate(fin);
            }
            else if(node.type > 1 && node.type < size){
                float fin = 0;
                for(int i = 0; i < layers[node.type - 1].size(); i++){
                    fin += layers[node.type - 1][i].activation * node.weight[i];
                }
                node.activate(fin);
            }
            else{
                float fin = 0;
                for(int i = 0; i < layers[size - 3].size(); i++){
                    fin += layers[size - 3][i].activation * node.weight[i];
                }
                node.activate(fin);
            }
        }
        else{
            if(node.type == 0){
                LOG("lol why are you doing this?");
            }
            else{
                float fin = 0;
                for(int i = 0; i < input.size(); i++){
                    fin += input[i].activation * node.weight[i];
                }
                node.activate(fin);
            }
        }
        
    }
    
    void compute(vector<float> in){
        for(int i = 0; i < input.size(); i++){
            input[i].activation = in[i];
        }
        for(int j = 0; j < layers.size(); j++){
            for(int i = 0; i < layers[j].size(); i++){
                target(layers[j][i]);
            }
        }
        for(int i = 0; i < output.size(); i++){
            target(output[i]);
        }
    }
    
    float cost(vector<float> desired){
        float score = 0;
        for(int i = 0; i < output.size(); i++){
            score += pow(desired[i] - output[i].activation, 2);
        }
        return score;
    }
    
};

int main() {
    
    return 0;
}
