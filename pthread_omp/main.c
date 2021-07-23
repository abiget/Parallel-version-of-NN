#include "nn.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "pthread.h"
// #include<time.h>
#include "mnist.h"
ThreadInfo *paramArray[NUM_THREADS];

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

void collect_av_parameters(Network *final)
{
  //something

  for (int j = 1; j < NUM_THREADS; j++)
  {
    // #pragma omp parallel for
    // input layer
      // printf("%d\n",params[0]->network->hiddenLayer.nodes[0].numberOfWeights);
      // printf("%d\n",params[0]->network->outputLayer.nodes[0].numberOfWeights);

      // printf("%d\n",final->inputLayer.nodes[0].numberOfWeights);
      // printf("%d\n",params[0]->network->inputLayer.nodes[0].numberOfWeights);


      //we don't have to compute the randome input layer weights 
      //----------------------------------------------------------
        // for (int k = 0; k < IMAGE_SIZE; k++)
          // params[0]->network->inputLayer.nodes[k].weights[k] += params[j]->network->inputLayer.nodes[k].weights[k];
       // #pragma omp parallel for
    //-----------------------------------------------------------
    // //hidden layer
        for (int k = 0; k < HIDDEN_LAYER_SIZE; k++)
        {
          for(int input = 0; input < IMAGE_SIZE; input++){
            paramArray[0]->network->hiddenLayer.nodes[k].weights[input] += paramArray[j]->network->hiddenLayer.nodes[k].weights[input];
          }
        }
        for (int k = 0; k < HIDDEN_LAYER_SIZE; k++)
        {
          for(int input = 0; input < IMAGE_SIZE; input++){
            printf("\n%f\n", paramArray[0]->network->hiddenLayer.nodes[k].weights[input]);
          }
        }

    // // #pragma omp parallel for
    // //output layer
        // for (int k = 0; k < OUTPUT_SIZE; k++)
        // {
        //   for(int input = 0; input < HIDDEN_LAYER_SIZE){
        //     *paramArray[0]->network->outputLayer.nodes[k].weights[input] += *paramArray[j]->network->outputLayer.nodes[k].weights[input];
        //   }
        // }
      // dealing with the unused paprameters, shit !
      //-------------------------------------------------------------------
      // for (int k = 0; k < IMAGE_SIZE; k++)
      //   *paramArray[0]->network->inputLayer.nodes[k].weights /= NUM_THREADS;
      //--------------------------------------------------------------------
      //Take avarage of parameters
      // for (int k = 0; k < HIDDEN_LAYER_SIZE; k++)
      //   *paramArray[0]->network->hiddenLayer.nodes[k].weights /= NUM_THREADS;
      // for (int k = 0; k < OUTPUT_SIZE; k++)
      //   *paramArray[0]->network->outputLayer.nodes[k].weights /= NUM_THREADS;
      final = paramArray[0]->network;
      // free(params);
}
}
void * printer(void * value){
  int * h = (int *) value;
  printf("thread %d was here\n",*h);
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

  Network final = networks.network [NUM_THREADS];
  testNetwork(&final);//will get back

  int iterationsPerThread = 60000 / NUM_THREADS;
  pthread_t * threads = (pthread_t *)malloc(sizeof(pthread_t) * NUM_THREADS);

  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  int rc;

  // #pragma omp parallel for
  for (int i = 0; i < TRAINING_EPOCHS; ++i)
  {
    printf("Training epoch %i/%i\n", i + 1, TRAINING_EPOCHS);
  
    for (int j = 0; j < NUM_THREADS; j++)
    { 
      ThreadInfo *param = (ThreadInfo *)malloc(sizeof(ThreadInfo));

      param->start = j * iterationsPerThread;
      param->end = j * (iterationsPerThread + 1);

      param->network = &networks.network[j];
      paramArray[j] = param;

      rc = pthread_create(&threads[j], &attr, trainNetwork, (void *)&paramArray[j]);
      if(rc)
        printf("\nSomething wrong during thread creation!");
    }
    for (int j = 0; j < NUM_THREADS; j++)
      pthread_join(threads[j], NULL);
    // printf("%d\n\n", pthread());

    //---------------------------------------------Collector
    //pgram

    collect_av_parameters(&final);
    testNetwork( paramArray[0]->network);

    //---------------------------------------------

    // trainNetwork(&networks);
  }
  stop = getTime();
  double parallelTime = stop - start;
  printf("\nParallel Implementation of NN\n");
  printf("\nTime Elapsed=%f\n", parallelTime);
  // pthread_exit(NULL);
  return 0;
}