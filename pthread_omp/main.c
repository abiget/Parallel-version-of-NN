#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "pthread.h"
#include "mnist.h"

double getTime()
{
  const double kMicro = 1.0e-6;
  struct timeval TV;

  const int RC = gettimeofday(&TV, NULL);
  if (RC == -1)
  {
    printf("ERROR: Bad call to gettimeofday\n");
    return (-1);
  }
  return (((double)TV.tv_sec) + kMicro * ((double)TV.tv_usec));
}

void collect_av_parameters(ThreadInfo *params[])
{
  //something
  for (int j = 1; j < NUM_THREADS; j++)
  {
      //hidden layer parameters
    for (int k = 0; k < HIDDEN_LAYER_SIZE; k++)
      for (int input = 0; input < IMAGE_SIZE; input++)
        params[0]->network->hiddenLayer.nodes[k].weights[input] += params[j]->network->hiddenLayer.nodes[k].weights[input];
      //output layer parameters
    for (int k = 0; k < OUTPUT_SIZE; k++)
      for (int input = 0; input < HIDDEN_LAYER_SIZE; input++)
        params[0]->network->outputLayer.nodes[k].weights[input] += params[j]->network->outputLayer.nodes[k].weights[input];
  }
    //avarage hidden layer parameters
  for (int k = 0; k < HIDDEN_LAYER_SIZE; k++)
    for (int input = 0; input < IMAGE_SIZE; input++)
      params[0]->network->hiddenLayer.nodes[k].weights[input] /= NUM_THREADS;
  //avarage output layer parameters
  for (int k = 0; k < OUTPUT_SIZE; k++)
    for (int input = 0; input < HIDDEN_LAYER_SIZE; input++)
      params[0]->network->outputLayer.nodes[k].weights[input] /= NUM_THREADS;
}

int main()
{
  load_mnist();
  double start, stop;

  start = getTime();
  Networks networks;
  initNetworks(&networks);
  printf("\nNumber of threads: %i\n",NUM_THREADS);
  testNetwork(&networks.network[0]); //will get back
  printf("\n\n\t========= Training Started ==========\n\n");



  int iterationsPerThread = NUM_TRAIN / NUM_THREADS;
  pthread_t *threads = (pthread_t *)malloc(sizeof(pthread_t) * NUM_THREADS);
  ThreadInfo *paramArray[NUM_THREADS];
  int rc;
  for (int i = 0; i < 10; ++i)
  {
    printf("Training epoch %i/%i: \t", i + 1, TRAINING_EPOCHS);

    for (int j = 0; j < NUM_THREADS; j++)
    {
      ThreadInfo *param = (ThreadInfo *)malloc(sizeof(ThreadInfo));

      param->start = j * iterationsPerThread;
      param->end = (j+1) * (iterationsPerThread);
      param->network = &networks.network[j];
      paramArray[j] = param;

      rc = pthread_create(&threads[j], NULL, trainNetwork, (void *)param);
      if (rc)
        printf("\nSomething wrong during thread creation!\n");
    }
    for (int j = 0; j < NUM_THREADS; j++)
      pthread_join(threads[j], NULL);

    //---------------------------------------------for vaidation begin----------------------
    double s = 0;
    for (int nt = 0; nt < NUM_THREADS; nt++)
      s += paramArray[nt]->network->hiddenLayer.nodes[32].weights[0];
   //---------------------------------------------for vaidation end----------------------
   //collect parametrs from the models 
    collect_av_parameters(paramArray);
    // for(int nt = 0; nt<NUM_THREADS; nt++)
    // testNetwork(paramArray[1]->network);
    // for(int nt = 0; nt<NUM_THREADS; nt++)
    // printf("\n\n weight:  %f,%f",*paramArray[0]->network->hiddenLayer.nodes[32].weights,s/12);
    // *paramArray[0]->network->hiddenLayer.nodes[1].weights+=*paramArray[0]->network->hiddenLayer.nodes[1].weights;
    // printf("\n\n weight: %f, %f", paramArray[0]->network->hiddenLayer.nodes[32].weights[0], s / 12);
    // paramArray[0]->network->hiddenLayer.nodes[1].weights[1] += 10;
    // printf("\n\n weight:  %f", paramArray[0]->network->hiddenLayer.nodes[1].weights[1]);

    for (int nt = 0; nt < NUM_THREADS; nt++)
    {
      Network * ptr = &networks.network[nt];
      ptr = paramArray[0]->network;
    }
    //---------------------------------------------Test the final network (i.e network[0])
    testNetwork( &networks.network[0]);
  }
  printf("\n\n\t========= Training Ended ==========\n");

  stop = getTime();
  double parallelTime = stop - start;
  printf("\nParallel Implementation of NN\n");
  printf("\nTime Elapsed=%f seconds\n", parallelTime);
  pthread_exit(NULL);
  return 0;
}
