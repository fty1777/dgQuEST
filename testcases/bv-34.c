#include <stdio.h>

#include "QuEST.h"
#include "mytimer.hpp"

int main(int narg, char *varg[]) {
    QuESTEnv env = createQuESTEnv();
    Qureg q = createQureg(34, env);

    hadamard(q, 33);
    pauliZ(q, 33);
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
    hadamard(q, 32);
    controlledNot(q, 0, 33);
    controlledNot(q, 2, 33);
    controlledNot(q, 4, 33);
    controlledNot(q, 6, 33);
    controlledNot(q, 7, 33);
    controlledNot(q, 8, 33);
    controlledNot(q, 11, 33);
    controlledNot(q, 12, 33);
    controlledNot(q, 13, 33);
    controlledNot(q, 14, 33);
    controlledNot(q, 17, 33);
    controlledNot(q, 19, 33);
    controlledNot(q, 20, 33);
    controlledNot(q, 21, 33);
    controlledNot(q, 22, 33);
    controlledNot(q, 26, 33);
    controlledNot(q, 29, 33);
    controlledNot(q, 30, 33);
    controlledNot(q, 32, 33);
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
    hadamard(q, 32);

    searchQuregTaskPartitions(q, env);
    double t1 = get_wall_time();
    getAmp(q, 0);
    double t2 = get_wall_time();

    printf("Complete the simulation takes time %12.6f seconds.\n", t2 - t1);

    destroyQureg(q, env);
    destroyQuESTEnv(env);
    return 0;
}
