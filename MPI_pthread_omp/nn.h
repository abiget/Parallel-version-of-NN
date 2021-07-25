#include "config.h"
// #if !defined()
// #include "../mnist.h"
// #endif
typedef struct Node_{
    double bias;
    double output;
    double backPropValue;
    int numberOfWeights;
    double* weights;
} Node;

typedef struct Layer_{
    int numberOfNodes;
    Node* nodes;
} Layer;

typedef struct Network_{
    Layer inputLayer;
    Layer hiddenLayer;
    Layer outputLayer;
} Network;
typedef struct Networks_{
    int numberOfNetworks;
    Network * network;
} Networks;
typedef struct threadInfo_{
    Network * network;
    int start, end;
}ThreadInfo;

void initNetworks(Networks* networks);
void * trainNetwork(void *myThreadInfo);
void testNetwork(Network *network);
