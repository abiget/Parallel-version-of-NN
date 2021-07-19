#include "dataLoader.h"
#include "config.h"

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
typedef struct ThreadInfo_{
    Network network;
    int start, end;
}ThreadInfo;

void initNetworks(Networks* network);
void * trainNetwork(ThreadInfo* threadInfo);
// void testNetwork(Networks *network);