#include "nn.h"
// #include <omp.h>
#include<time.h>
int main()
{
	// srand( static_cast<unsigned int>(time(NULL)));
    // srand( (unsigned int) time(NULL) );
	time_t start,end;
	// float start,stop;

    start = time(NULL);
    Network network;
    initNetwork(&network);

    testNetwork(&network);
    // #pragma omp parallel for
    for(int i=0; i<TRAINING_EPOCHS; ++i){
        printf("Training epoch %i/%i\n", i + 1, TRAINING_EPOCHS);
        trainNetwork(&network);
        testNetwork(&network);
    end = time(NULL);
    printf("\nTime Elapsed for %i =%ld \n",i+1,end -start);

    }
    end = time(NULL);
    // stop = omp_get_wtime();
    // float elapse=stop-start;
    printf("\nTime Elapsed=%ld",end -start);
    return 0;
}
