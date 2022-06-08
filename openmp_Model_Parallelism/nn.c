#include "nn.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mnist.h"


static void initLayer(int numberOfNodes, int numberOfWeights, Layer* layer);
static void initNode(int numberOfWeights, Node* node);
static double sigmoid(double value);
static double sigmoidDerivative(double nodeOutput);
static void feedForwardLayer(Layer* previousLayer, Layer* layer);
static void feedForward(Network* network, int img, int train);
static void updateNode(Layer* previousLayer, double backPropValue, Node* node);
static void backPropagate(Network* network, int label);
static u_int8_t getClassification(Layer* layer);

void initNetwork(Network* network){
    initLayer(IMAGE_SIZE, 0, &network->inputLayer);
    initLayer(HIDDEN_LAYER_SIZE, IMAGE_SIZE, &network->hiddenLayer);
    initLayer(OUTPUT_SIZE, HIDDEN_LAYER_SIZE, &network->outputLayer);
}

void trainNetwork(Network* network){
    for(int i=0; i<NUM_TRAIN; i++){
        feedForward(network, i, 1);
        backPropagate(network, train_label[i]);
    }
}

void testNetwork(Network *network){
    int errCount = 0;
    for(int i=0; i<NUM_TEST; i++){
        feedForward(network, i, 0);

        u_int8_t classification = getClassification(&network->outputLayer);
        if (classification!=test_label[i]){
            errCount++;
        }
    }
    printf("Test Accuracy: %0.2f%%\n", ((double)(NUM_TEST - errCount) / NUM_TEST) * 100);  
}

static void initLayer(int numberOfNodes, int numberOfWeights, Layer* layer){
    Node* nodes = malloc(numberOfNodes * sizeof(Node));
    #pragma omp parallel for
    for(int hn=0; hn<numberOfNodes; ++hn){
        Node* node = &nodes[hn];
        initNode(numberOfWeights, node);
    }

    layer->numberOfNodes = numberOfNodes;
    layer->nodes = nodes;
}

static void initNode(int numberOfWeights, Node* node){
    //Initialize weights between -0.7 and 0.7
    double* weights = malloc(numberOfWeights * sizeof(double));
    // #pragma omp parallel for
    for(int w=0; w<numberOfWeights; ++w){
        weights[w] = 0.7 * (rand()/(double)(RAND_MAX));
        if (w%2){
            weights[w] = -weights[w];
        }
    }

    node->numberOfWeights = numberOfWeights;
    node->weights = weights;
    node->bias = rand()/(double)(RAND_MAX);
}

static double sigmoid(double value){
    return 1.0 / (1.0 + exp(-value));
}

static double sigmoidDerivative(double nodeOutput){
    return nodeOutput * (1- nodeOutput);
}

static void feedForwardLayer(Layer* previousLayer, Layer* layer){          
    for(int hn=0; hn<layer->numberOfNodes; ++hn){
        Node* node = &layer->nodes[hn];
        node->output = node->bias;
        float temp=node->bias;
        #pragma omp parallel for reduction(+: temp)
            for(int w=0; w<previousLayer->numberOfNodes; ++w)
                temp += previousLayer->nodes[w].output * node->weights[w];

        node->output = sigmoid(temp);
    }
}

static void feedForward(Network* network, int img, int train){
    if(train==1){
      #pragma omp parallel for
        for(int i=0; i<IMAGE_SIZE; ++i)
            network->inputLayer.nodes[i].output = (double)(train_image[img][i]);
    }
    else{
      #pragma omp parallel for  
        for(int i=0; i<IMAGE_SIZE; ++i)
            network->inputLayer.nodes[i].output = (double)(test_image[img][i]);
    }

    feedForwardLayer(&network->inputLayer, &network->hiddenLayer);
    feedForwardLayer(&network->hiddenLayer, &network->outputLayer);
}

static void updateNode(Layer* previousLayer, double backPropValue, Node* node){
    // #pragma omp parallel for 
    for(int hn=0; hn<previousLayer->numberOfNodes; ++hn){
        Node* previousLayerNode = &previousLayer->nodes[hn];
        node->weights[hn] += LEARNING_RATE * previousLayerNode->output * backPropValue;
    }
    node->bias += LEARNING_RATE * backPropValue;
}

static void backPropagate(Network* network, int label){
    // #pragma omp barrier
    Layer* hiddenLayer = &network->hiddenLayer;
    Layer* outputLayer = &network->outputLayer;
    #pragma omp parallel for
    for(int on=0; on<outputLayer->numberOfNodes; ++on){
        Node* outputNode = &outputLayer->nodes[on];

        int nodeTarget = (on==label) ? 1:0;
        double errorDelta = nodeTarget - outputNode->output;
        double backPropValue = errorDelta * sigmoidDerivative(outputNode->output);

        outputNode->backPropValue = backPropValue;
        updateNode(&network->hiddenLayer, outputNode->backPropValue, outputNode);
    }
    #pragma omp parallel for
    for(int hn=0; hn<hiddenLayer->numberOfNodes; ++hn){
        Node* hiddenNode = &hiddenLayer->nodes[hn];

        double outputNodesBackPropSum = 0;

        for(int on=0; on<outputLayer->numberOfNodes; ++on){
            Node* outputNode = &outputLayer->nodes[on];
            outputNodesBackPropSum += outputNode->backPropValue * outputNode->weights[hn];
        }

        double hiddenNodeBackPropValue = outputNodesBackPropSum * sigmoidDerivative(hiddenNode->output);
        updateNode(&network->inputLayer, hiddenNodeBackPropValue, hiddenNode);
    }
}

static u_int8_t getClassification(Layer* layer){
    double maxOutput = 0;
    int maxIndex = 0;
    for(int on=0; on<layer->numberOfNodes; ++on){
        double nodeOutput = layer->nodes[on].output;
        if(nodeOutput > maxOutput){
            maxOutput = nodeOutput;
            maxIndex = on;
        }
    }
    return (u_int8_t)maxIndex;
}
