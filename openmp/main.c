#include "nn.h"
#include <omp.h>
#include<time.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
double getTime()
{
  const double kMicro = 1.0e-6;
  struct timeval TV;

  const int RC = gettimeofday(&TV, NULL);
  if(RC == -1)
    {
      printf("ERROR: Bad call to gettimeofday\n");
      return(-1);
    }
  return( ((double)TV.tv_sec) + kMicro * ((double)TV.tv_usec) );
}
int main()
{
	// srand( static_cast<unsigned int>(time(NULL)));
  //   srand( (unsigned int) time(NULL) );
	// float start,stop;
	// start = omp_get_wtime();
  // omp_set_nested(0);
	double start,stop;
	start = getTime();

    Network network;
    initNetwork(&network);

    testNetwork(&network);
    printf("%d\n\n",omp_get_max_threads());
    // #pragma omp parallel for
    for(int i=0; i<TRAINING_EPOCHS; ++i){
        printf("Training epoch %i/%i\n", i + 1, TRAINING_EPOCHS);
        trainNetwork(&network);
        testNetwork(&network);
    }
    // stop = omp_get_wtime();
    // float elapse=stop-start;
    stop = getTime();
    double serialTime= stop-start;
    printf("\nTime Elapsed=%f",serialTime );
    //  printf("\nTime Elapsed=%f",elapse );
    return 0;
}
