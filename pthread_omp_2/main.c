#include "nn.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "pthread.h"
// #include<time.h>
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
void collect_av_parameters(ThreadInfo *params[], Network *final)
{
  //something

  for (int j = 1; j < NUM_THREADS; j++)
  {
// #pragma omp parallel for
//input layer
    for (int k = 0; k < IMAGE_SIZE; k++)
      *params[0]->network.inputLayer.nodes[k].weights += *params[j]->network.inputLayer.nodes[k].weights;
// #pragma omp parallel for
//hidden layer
    for (int k = 0; k < HIDDEN_LAYER_SIZE; k++)
      *params[0]->network.hiddenLayer.nodes[k].weights += *params[j]->network.hiddenLayer.nodes[k].weights;
// #pragma omp parallel for
//output layer
    for (int k = 0; k < OUTPUT_SIZE; k++)
      *params[0]->network.outputLayer.nodes[k].weights += *params[j]->network.outputLayer.nodes[k].weights;
  }
  for (int k = 0; k < IMAGE_SIZE; k++)
    *params[0]->network.inputLayer.nodes[k].weights /= NUM_THREADS;
  for (int k = 0; k < HIDDEN_LAYER_SIZE; k++)
    *params[0]->network.hiddenLayer.nodes[k].weights /= NUM_THREADS;
  for (int k = 0; k < OUTPUT_SIZE; k++)
    *params[0]->network.outputLayer.nodes[k].weights /= NUM_THREADS;
  final = &params[0]->network;
  free(params);
}
int main()
{
  // srand( static_cast<unsigned int>(time(NULL)));
  //   srand( (unsigned int) time(NULL) );
  // float start,stop;
  // start = omp_get_wtime();
  // omp_set_nested(0);
  load_mnist();
  double start, stop;
  start = getTime();

  Networks networks;
  initNetworks(&networks);

  Network final; //will get back
  // testNetwork(final);//will get back
  printf("%d\n\n", omp_get_max_threads());

  int iterationsPerThread = 10000 / NUM_THREADS;
  // pthread_t threads = (pthread_t)malloc(sizeof(pthread_t) * NUM_THREADS);
  // ThreadInfo *paramArray[NUM_THREADS];
  // ThreadInfo *param = (ThreadInfo *)malloc(sizeof(ThreadInfo));
  // int rc;

  #pragma omp num_threads(NUM_THREADS) parallel  for
  for (int i = 0; i < TRAINING_EPOCHS; ++i)
  {
    printf("Training epoch %i/%i\n", i + 1, TRAINING_EPOCHS);
    printf("hereeeeeeeeee3333");
         
    // for (int j = 0; j < NUM_THREADS; j++)
    // { 
      // printf("hereeeeeeeeee2222");
      // param->start = j * iterationsPerThread;
      // param->end = j * (iterationsPerThread + 1);

      // param->network = networks.network[j];
      // paramArray[j] = param;

    //   rc = pthread_create(threads[j], NULL, trainNetwork, (void *)paramArray[j]);
    //   if(!rc){
    //     printf("something here");
    //   }
    //   else{
    //     printf("alr4");
    //   }
    // }
    // for (int j = 0; j < NUM_THREADS; j++)
    //   pthread_join(threads[j], NULL);
    //---------------------------------------------Collector
    //pgram

    // collect_av_parameters(paramArray, &final);
    trainNetwork(&networks);
    // testNetwork(&final);

    //---------------------------------------------
  }
  stop = getTime();
  double parallelTime = stop - start;
  printf("Parallel Implementation of NN\n");
  printf("\nTime Elapsed=%f", parallelTime);
  // pthread_exit(NULL);
  return 0;
}
