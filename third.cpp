#include <cmath>
#include <vector>
#include <memory>
#include <cstdlib>
#include <iostream>

static const float e = 2.718281828459045;
void setup();
void LOG(float);
void PRINT(std::vector<float>);
std::vector<float> sort(std::vector<float>);
std::vector<float> merge(std::vector<float>, std::vector<float>);

struct vec {
    float x;
    float y;
    float z;
    vec add(vec a, vec b){
        return vec {a.x + b.x, a.y + b.y, a.z + a.z};
    }
    vec sub(vec a, vec b){
        return vec {a.x - b.x, a.y - b.y, a.z - a.z};
    }
    vec mult(vec a, vec b){
        return vec {a.x * b.x, a.y * b.y, a.z * a.z};
    }
};

struct Node {
    
public:
    vec pos;
    
private:
    float sum;
    float bias;
    float activation;
    std::vector<float> weights;
    
public:
    Node(){
        sum = 0;
        activation = 0;
        bias = static_cast<float>(rand() & 3) - 1.5;
        weights.resize(10, 0.0);
        for(int i = 0; i < weights.size(); i++){
            weights[i] = static_cast<float>(rand() & 1023) / 1023 - 0.5;
        }
    }
    void setPos(float x, float y, float z){
        pos.x = x;
        pos.y = y;
        pos.z = z;
    }
    float activate(float x){
        return 1.0 / (1.0 + (pow(e, -x)));
    }
    float activate_prime(float x){
        return (1.0 / (1.0 + pow(e, -x))) * (1.0 - (1.0 / (1.0 + pow(e, -x))));
    }
    
};



class Net {
private:
    
    
public:
    Net(){
        
    }
    
private:
    float dist(Node a, Node b){
        return pow(a.pos.x - b.pos.x, 2) + pow(a.pos.y - b.pos.y, 2) + pow(a.pos.x - b.pos.x, 2);
    }
    std::vector<Node> sort(std::vector<Node> v, Node n){
        if(v.size() < 2){
            return v;
        }
        else if(v.size() == 2){
            if(dist(n, v[0]) > dist(n, v[1])){
                return std::vector<Node> {v[1], v[0]};
            }
            return v;
        }
        return merge(sort(std::vector<Node> (&v[0], &v[floor(v.size()/2)]), n), sort(std::vector<Node> (&v[floor(v.size()/2)], &v[v.size()]), n), n);
    }
    std::vector<Node> merge(std::vector<Node> v1, std::vector<Node> v2, Node n){
        std::vector<Node> fin(v1.size() + v2.size());
        int counter[2] = {0, 0};
        for(int i = 0; i < fin.size(); i++){
            if(counter[0] == v1.size()){
                fin[i] = v2[counter[1]++];
                continue;
            }
            else if(counter[1] == v2.size()){
                fin[i] = v1[counter[0]++];
                continue;
            }
            fin[i] = dist(n, v1[counter[0]]) < dist(n, v2[counter[1]]) ? v1[counter[0]++] : v2[counter[1]++];
        }
        return fin;
    }
    
};



int main() {
    setup();
    
    std::vector<float> x = {3, 2, 1, 0, 2.5, 6, -1, 4.2, 1.24, -0.32};
    
    PRINT(sort(x));
    
    return 0;
}





void LOG(float x){
    std::cout << x << std::endl;
};
void PRINT(std::vector<float> x){
    std::cout << "[";
    for(int i = 0; i < x.size(); i++){
        if(i != x.size() - 1){
            std::cout << x[i] << ", ";
        }
        else{
            std::cout << x[i];
        }
    }
    std::cout << "]\n";
};
void setup(){
    srand(static_cast<int>(time(0)));
};
std::vector<float> sort(std::vector<float> v){
    if(v.size() < 2){
        return v;
    }
    if(v.size() == 2){
        if(v[0] > v[1]){
            return std::vector<float> {v[1], v[0]};
        }
        return v;
    }
    return merge(sort(std::vector<float> (&v[0], &v[floor(v.size()/2)])), sort(std::vector<float> (&v[floor(v.size()/2)], &v[v.size()])));
}
std::vector<float> merge(std::vector<float> v1, std::vector<float> v2){
    std::vector<float> fin(v1.size() + v2.size());
    int counter[2] = {0, 0};
    for(int i = 0; i < fin.size(); i++){
        if(counter[0] == v1.size()){
            fin[i] = v2[counter[1]++];
            continue;
        }
        else if(counter[1] == v2.size()){
            fin[i] = v1[counter[0]++];
            continue;
        }
        fin[i] = v1[counter[0]] < v2[counter[1]] ? v1[counter[0]++] : v2[counter[1]++];
    }
    return fin;
}
