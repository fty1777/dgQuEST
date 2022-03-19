#include <stdio.h>

#include "QuEST.h"
#include "mytimer.hpp"

int main(int narg, char *varg[]) {
    QuESTEnv env = createQuESTEnv();
    Qureg q = createQureg(32, env);

    hadamard(q, 31);
    pauliZ(q, 31);
    hadamard(q, 0);
    hadamard(q, 1);
    hadamard(q, 2);
    hadamard(q, 3);
    hadamard(q, 4);
    hadamard(q, 5);
    hadamard(q, 6);
    hadamard(q, 7);
    hadamard(q, 8);
    hadamard(q, 9);
    hadamard(q, 10);
    hadamard(q, 11);
    hadamard(q, 12);
    hadamard(q, 13);
    hadamard(q, 14);
    hadamard(q, 15);
    hadamard(q, 16);
    hadamard(q, 17);
    hadamard(q, 18);
    hadamard(q, 19);
    hadamard(q, 20);
    hadamard(q, 21);
    hadamard(q, 22);
    hadamard(q, 23);
    hadamard(q, 24);
    hadamard(q, 25);
    hadamard(q, 26);
    hadamard(q, 27);
    hadamard(q, 28);
    hadamard(q, 29);
    hadamard(q, 30);
    controlledNot(q, 2, 31);
    controlledNot(q, 4, 31);
    controlledNot(q, 6, 31);
    controlledNot(q, 7, 31);
    controlledNot(q, 8, 31);
    controlledNot(q, 16, 31);
    controlledNot(q, 17, 31);
    controlledNot(q, 19, 31);
    controlledNot(q, 20, 31);
    controlledNot(q, 21, 31);
    controlledNot(q, 22, 31);
    controlledNot(q, 24, 31);
    controlledNot(q, 25, 31);
    controlledNot(q, 27, 31);
    controlledNot(q, 28, 31);
    hadamard(q, 0);
    hadamard(q, 1);
    hadamard(q, 2);
    hadamard(q, 3);
    hadamard(q, 4);
    hadamard(q, 5);
    hadamard(q, 6);
    hadamard(q, 7);
    hadamard(q, 8);
    hadamard(q, 9);
    hadamard(q, 10);
    hadamard(q, 11);
    hadamard(q, 12);
    hadamard(q, 13);
    hadamard(q, 14);
    hadamard(q, 15);
    hadamard(q, 16);
    hadamard(q, 17);
    hadamard(q, 18);
    hadamard(q, 19);
    hadamard(q, 20);
    hadamard(q, 21);
    hadamard(q, 22);
    hadamard(q, 23);
    hadamard(q, 24);
    hadamard(q, 25);
    hadamard(q, 26);
    hadamard(q, 27);
    hadamard(q, 28);
    hadamard(q, 29);
    hadamard(q, 30);

    searchQuregTaskPartitions(q, env);
    double t1 = get_wall_time();
    getAmp(q, 0);
    double t2 = get_wall_time();

    printf("Complete the simulation takes time %12.6f seconds.\n", t2 - t1);

    destroyQureg(q, env);
    destroyQuESTEnv(env);
    return 0;
}
