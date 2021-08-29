#include "nn.h"
#include "mnist.h"
#include <sys/time.h>


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


int main()
{

  load_mnist();
	double start,stop;
	start = getTime();
    Network network;
    initNetwork(&network);

    testNetwork(&network);
    // #pragma omp parallel for
    for(int i=0; i<TRAINING_EPOCHS; ++i){
        printf("Training epoch %i/%i\n", i + 1, TRAINING_EPOCHS);
        trainNetwork(&network);
        testNetwork(&network);
    }
    stop = getTime();
    double elapse=stop-start;
    printf("\nTime Elapsed=%f",elapse );
    return 0;
}
