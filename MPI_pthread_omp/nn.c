#include "nn.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mnist.h"
static void initNetwork(Network *network);
static void initLayer(int numberOfNodes, int numberOfWeights, Layer *layer);
static void initNode(int numberOfWeights, Node *node);
static double sigmoid(double value);
static double sigmoidDerivative(double nodeOutput);
static void feedForwardLayer(Layer *previousLayer, Layer *layer);
static void feedForward(Network *network, int var,int train);
static void updateNode(Layer *previousLayer, double backPropValue, Node *node);
static void backPropagate(Network *network, int label);
static uint8_t getClassification(Layer *layer);

void initNetworks(Networks *networks)
{   
    Network *ntks = malloc(NUM_THREADS + 1* sizeof(Network));
    for (int hn = 0; hn < NUM_THREADS + 1; ++hn)
    {
        Network *ntk = &ntks[hn];
        initNetwork(ntk);

    }
        networks->network = ntks;
// #pragma omp parallel for
//     for (int i = 0; i < NUM_THREADS; i++)
//         initNetwork(&networks->network[i]);
}
void initNetwork(Network *network)
{
    initLayer(IMAGE_SIZE, 0, &network->inputLayer);
    initLayer(HIDDEN_LAYER_SIZE, IMAGE_SIZE, &network->hiddenLayer);
    initLayer(OUTPUT_SIZE, HIDDEN_LAYER_SIZE, &network->outputLayer);
}

void * trainNetwork(void *myThreadInfo)
{ 
    ThreadInfo *myThread = (ThreadInfo *)myThreadInfo;
    Network * myNetwork = myThread->network;
    int myStart = myThread->start;
    int myEnd = myThread->end;
    // FILE *imageFile;
    // FILE *labelFile;
    // ImageFileHeader imageFileHeader;
    // imageFile = openImageFile(TRAINING_SET_IMAGE_FILE_NAME, &imageFileHeader);
    // labelFile = openLabelFile(TRAINING_SET_LABEL_FILE_NAME);

    for (int q = myStart; q < myEnd; q++)
    {
        // Image img;
        // getImage(imageFile, &img);
        // uint8_t label = getLabel(labelFile);

        feedForward(myNetwork, q,1);
        backPropagate(myNetwork, q);
        // printf(" ");
    }
    // pthread_exit(NULL);

}

void testNetwork(Network *network)
{
    // FILE *imageFile;
    // FILE *labelFile;
    // ImageFileHeader imageFileHeader;
    // imageFile = openImageFile(TEST_SET_IMAGE_FILE_NAME, &imageFileHeader);
    // labelFile = openLabelFile(TEST_SET_LABEL_FILE_NAME);

    int errCount = 0;
    // #pragma omp parallel for
    // #pragma omp barrier
    // #pragma omp parallel for
    for (int test = 0; test < 10000; test++)
    {
        // Image img;
        // getImage(imageFile, &img);
        // uint8_t lbl = getLabel(labelFile);
        feedForward(network, test,0);

        uint8_t classification = getClassification(&network->outputLayer);
        if (classification != test_label[test])
        {
            errCount++;
        }
    }
    // fclose(imageFile);
    // fclose(labelFile);

    printf("Test Accuracy: %0.2f%%\n", ((double)(60000 - errCount) / 60000) * 100);
}

static void initLayer(int numberOfNodes, int numberOfWeights, Layer *layer)
{
    Node *nodes = malloc(numberOfNodes * sizeof(Node));
    #pragma omp parallel for schedule(static)
        for (int hn = 0; hn < numberOfNodes; ++hn)
        {
            Node *node = &nodes[hn];
            initNode(numberOfWeights, node);
        }

    layer->numberOfNodes = numberOfNodes;
    layer->nodes = nodes;
}

static void initNode(int numberOfWeights, Node *node)
{
    double *weights = malloc(numberOfWeights * sizeof(double));
    // #pragma omp parallel for
    for (int w = 0; w < numberOfWeights; ++w)
    {
        weights[w] = 0.7 * (rand() / (double)(RAND_MAX));
        if (w % 2)
        {
            weights[w] = -weights[w];
        }
    }

    node->numberOfWeights = numberOfWeights;
    node->weights = weights;
    node->bias = rand() / (double)(RAND_MAX);
}

static double sigmoid(double value)
{
    return 1.0 / (1.0 + exp(-value));
}

static double sigmoidDerivative(double nodeOutput)
{
    return nodeOutput * (1 - nodeOutput);
}

static void feedForwardLayer(Layer *previousLayer, Layer *layer)
{

    for (int hn = 0; hn < layer->numberOfNodes; ++hn)
    {
        Node *node = &layer->nodes[hn];
        node->output = node->bias;
        float temp = node->bias;
        // #pragma omp parallel for reduction(+: temp)
        for (int w = 0; w < previousLayer->numberOfNodes; ++w)
        {
            temp += previousLayer->nodes[w].output * node->weights[w];
        }

        node->output = sigmoid(temp);
    }
}

static void feedForward(Network *network, int var, int train)
{
    //Populate the input layer with normalized input
    // #pragma omp parallel for schedule(static)

    if (train==1){
        for (int b = 0; b < IMAGE_SIZE; ++b)
            network->inputLayer.nodes[b].output = (double)(train_image[var][b] / 255.0);
    }

    else{
        for (int b = 0; b < IMAGE_SIZE; ++b)
            network->inputLayer.nodes[b].output = (double)(test_image[var][b] / 255.0);
    }

    feedForwardLayer(&network->inputLayer, &network->hiddenLayer);
    feedForwardLayer(&network->hiddenLayer, &network->outputLayer);
}

static void updateNode(Layer *previousLayer, double backPropValue, Node *node)
{
    // #pragma omp parallel for
    for (int hn = 0; hn < previousLayer->numberOfNodes; ++hn)
    {
        Node *previousLayerNode = &previousLayer->nodes[hn];
        node->weights[hn] += LEARNING_RATE * previousLayerNode->output * backPropValue;
    }
    node->bias += LEARNING_RATE * backPropValue;
}

static void backPropagate(Network *network, int label)
{
    // #pragma omp barrier
    Layer *hiddenLayer = &network->hiddenLayer;
    Layer *outputLayer = &network->outputLayer;
// #pragma omp parallel for schedule(static, 1)
    for (int on = 0; on < outputLayer->numberOfNodes; ++on)
    {
        Node *outputNode = &outputLayer->nodes[on];

        int nodeTarget = (on == train_label[label]) ? 1 : 0;
        double errorDelta = nodeTarget - outputNode->output;
        double backPropValue = errorDelta * sigmoidDerivative(outputNode->output);

        outputNode->backPropValue = backPropValue;
        updateNode(&network->hiddenLayer, outputNode->backPropValue, outputNode);
    }
// #pragma omp parallel for schedule(static, 1)
    for (int hn = 0; hn < hiddenLayer->numberOfNodes; ++hn)
    {
        Node *hiddenNode = &hiddenLayer->nodes[hn];

        double outputNodesBackPropSum = 0;

        for (int on = 0; on < outputLayer->numberOfNodes; ++on)
        {
            Node *outputNode = &outputLayer->nodes[on];
            outputNodesBackPropSum += outputNode->backPropValue * outputNode->weights[hn];
        }

        double hiddenNodeBackPropValue = outputNodesBackPropSum * sigmoidDerivative(hiddenNode->output);
        updateNode(&network->inputLayer, hiddenNodeBackPropValue, hiddenNode);
    }
}

static uint8_t getClassification(Layer *layer)
{
    double maxOutput = 0;
    int maxIndex = 0;
#pragma omp parallel for
    for (int on = 0; on < layer->numberOfNodes; ++on)
    {
        double nodeOutput = layer->nodes[on].output;
        if (nodeOutput > maxOutput)
        {
            maxOutput = nodeOutput;
            maxIndex = on;
        }
    }
    return (uint8_t)maxIndex;
}
