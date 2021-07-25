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

void collect_av_parameters(ThreadInfo *params[])
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
    for (int k = 0; k < HIDDEN_LAYER_SIZE; k++){
      // for (int input = 0; input < IMAGE_SIZE; input++)
        // params[0]->network->hiddenLayer.nodes[k].weights[input] += params[j]->network->hiddenLayer.nodes[k].weights[input];
      // params[0]->network->hiddenLayer.nodes[k].bias += params[0]->network->hiddenLayer.nodes[k].bias;
      params[0]->network->hiddenLayer.nodes[k].backPropValue += params[j]->network->hiddenLayer.nodes[k].backPropValue;

    }
    // // #pragma omp parallel for
    // //output layer
    for (int k = 0; k < OUTPUT_SIZE; k++){
      // for (int input = 0; input < HIDDEN_LAYER_SIZE; input++)
        // params[0]->network->outputLayer.nodes[k].weights[input] += params[j]->network->outputLayer.nodes[k].weights[input];
      // params[0]->network->outputLayer.nodes[k].bias += params[j]->network->outputLayer.nodes[k].bias;
      params[0]->network->outputLayer.nodes[k].backPropValue += params[j]->network->outputLayer.nodes[k].backPropValue;
    }
    // dealing with the unused paprameters, shit !
    //-------------------------------------------------------------------
    // for (int k = 0; k < IMAGE_SIZE; k++)
    //   *params[0]->network->inputLayer.nodes[k].weights /= NUM_THREADS;
    //--------------------------------------------------------------------
    //Take avarage of parameters

    // free(params);
  }

  for (int k = 0; k < HIDDEN_LAYER_SIZE; k++)
    params[0]->network->hiddenLayer.nodes[k].backPropValue /= NUM_THREADS;

  for (int k = 0; k < OUTPUT_SIZE; k++)
    params[0]->network->outputLayer.nodes[k].backPropValue /= NUM_THREADS;
  
  for (int on = 0; on < params[0]->network->outputLayer.numberOfNodes; ++on){
    Node *outputNode = &params[0]->network->outputLayer.nodes[on];
    updateNode(&params[0]->network->hiddenLayer, outputNode->backPropValue, outputNode);
  }

  for (int on = 0; on < params[0]->network->hiddenLayer.numberOfNodes; ++on){
    Node *hiddenNode = &params[0]->network->hiddenLayer.nodes[on];
    updateNode(&params[0]->network->inputLayer, hiddenNode->backPropValue, hiddenNode);
  }

}
void *printer(void *value)
{
  int *h = (int *)value;
  printf("thread %d was here\n", *h);
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
  pthread_mutex_t a_mutex = PTHREAD_MUTEX_INITIALIZER;

  start = getTime();
  Networks networks;
  initNetworks(&networks);
  // Network final;
  // initNetwork(&final);

  testNetwork(&networks.network[0]);//will get back

  int iterationsPerThread = 60000 / NUM_THREADS;
  pthread_t * threads = (pthread_t *)malloc(sizeof(pthread_t) * NUM_THREADS);

  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  ThreadInfo *paramArray[NUM_THREADS];
  int rc;

  // // #pragma omp parallel for
  for (int i = 0; i < TRAINING_EPOCHS; ++i)
  {
    printf("Training epoch %i/%i\n", i + 1, TRAINING_EPOCHS);
  
    for (int j = 0; j < NUM_THREADS; j++)
    { 
      ThreadInfo *param = (ThreadInfo *)malloc(sizeof(ThreadInfo));

      param->start = j * iterationsPerThread;
      param->end = (j+1) * (iterationsPerThread);

      param->network = &networks.network[j];
      paramArray[j] = param;

      rc = pthread_create(&threads[j], &attr, trainNetwork, (void *)paramArray[j]);
      if(rc)
        printf("\nSomething wrong during thread creation!");
    }
    for (int j = 0; j < NUM_THREADS; j++)
      pthread_join(threads[j], NULL);

  //   //---------------------------------------------Collector
  //   //pgram
    double s = 0;
    for (int nt = 0; nt < NUM_THREADS; nt++)
      s += paramArray[nt]->network->hiddenLayer.nodes[32].weights[0];

    collect_av_parameters(paramArray);
  //   // for(int nt = 0; nt<NUM_THREADS; nt++)
  //   // testNetwork(paramArray[1]->network);
  //   // for(int nt = 0; nt<NUM_THREADS; nt++)
    printf("\n\n weight:  %f,%f",paramArray[0]->network->hiddenLayer.nodes[32].weights[0],s/12);
  //   // *paramArray[0]->network->hiddenLayer.nodes[1].weights+=*paramArray[0]->network->hiddenLayer.nodes[1].weights;
  //   printf("\n\n weight: %f, %f\n", paramArray[0]->network->hiddenLayer.nodes[5].weights[0],s/12);
  //   // paramArray[0]->network->hiddenLayer.nodes[1].weights[1] += 10;

    for (int nt = 0; nt < NUM_THREADS; nt++){
        networks.network[nt] = *paramArray[0]->network;
    }

    // printf("\t\t weight:  %f\n", final.hiddenLayer.nodes[32].weights[0]);

  //   //   int h =0;
  //   // for(int nt = 0; nt<64; nt++)
  //   //   if(*paramArray[0]->network->hiddenLayer.nodes[nt].weights>0.5)
  //   //     h++;
  //   // printf("\nnun empty number is :%d",h);

  //   //---------------------------------------------
    testNetwork(&networks.network[0]);

  //   // trainNetwork(&networks);
  }
  stop = getTime();
  double parallelTime = stop - start;
  printf("\nParallel Implementation of NN\n");
  printf("\nTime Elapsed=%f\n", parallelTime);
  pthread_exit(NULL);
  return 0;
}