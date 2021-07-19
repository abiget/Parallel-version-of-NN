#include "nn.h"
// #include <omp.h>
// #include<time.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mnist.h"
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
    // srand( (unsigned int) time(NULL) );
	// double start,stop;
	// start = getTime();
  //   Network network;
  //   initNetwork(&network);

  //   testNetwork(&network);
  //   // #pragma omp parallel for
    
  //   for(int i=0; i<TRAINING_EPOCHS; ++i){
  //       printf("Training epoch %i/%i\n", i + 1, TRAINING_EPOCHS);
  //       trainNetwork(&network);
  //       testNetwork(&network);
  //   // end = time(NULL);
  //   // printf("\nTime Elapsed for %i =%ld \n",i+1,end -start);

  //   }
  //   stop = getTime();
  //   double serialTime= stop-start;
  //   printf("Serial Implementation of NN\n");
  //   printf("\nTime Elapsed=%f",serialTime );
  // int array[784][100];
  printf("everything is alrgiht\n");
  int array[10][784];
  printf("everything is alrgiht\n");

  FILE* fp;
  fp = fopen("../data/t10k-images-idx3-ubyte","wb+");
  fwrite(array, sizeof(int), 10 * 784, fp);
  //  printf("%d\n", array[0][0]);
      // free(array);
  printf("everything is alrgiht\n");
  
  int rows = sizeof(array) / sizeof(array[0]); // returns rows
  int cols = sizeof(array[0]) / sizeof(int); // returns col
  printf("rows = %d\n", rows);
  printf("cols = %d\n", cols);
  for (int i = 0; i < rows; i++){
   for (int j = 0; j < cols; j++) 
      printf("%d", array[i][j]);
      printf("\n");
  }
 


 return 0;
}
