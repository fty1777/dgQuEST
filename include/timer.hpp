#include <time.h>
#include <sys/time.h>

static double getWallTime(){

    struct timeval time;

    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }

    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

static double getCpuTime(){
    return (double)clock() / CLOCKS_PER_SEC;
}