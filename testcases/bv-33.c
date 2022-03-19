#include <stdio.h>

#include "QuEST.h"
#include "mytimer.hpp"

int main(int narg, char *varg[]) {
    QuESTEnv env = createQuESTEnv();
    Qureg q = createQureg(33, env);

    hadamard(q, 32);
    pauliZ(q, 32);
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
    hadamard(q, 31);
    controlledNot(q, 0, 32);
    controlledNot(q, 1, 32);
    controlledNot(q, 2, 32);
    controlledNot(q, 3, 32);
    controlledNot(q, 6, 32);
    controlledNot(q, 9, 32);
    controlledNot(q, 12, 32);
    controlledNot(q, 15, 32);
    controlledNot(q, 18, 32);
    controlledNot(q, 20, 32);
    controlledNot(q, 22, 32);
    controlledNot(q, 23, 32);
    controlledNot(q, 25, 32);
    controlledNot(q, 27, 32);
    controlledNot(q, 30, 32);
    controlledNot(q, 31, 32);
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
    hadamard(q, 31);

    searchQuregTaskPartitions(q, env);
    double t1 = get_wall_time();
    getAmp(q, 0);
    double t2 = get_wall_time();

    printf("Complete the simulation takes time %12.6f seconds.\n", t2 - t1);

    destroyQureg(q, env);
    destroyQuESTEnv(env);
    return 0;
}
