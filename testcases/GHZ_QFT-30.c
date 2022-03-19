#include "QuEST.h"
#include "mytimer.hpp"
#include "stdio.h"

double piOn2n(int n) {
    double ans = 3.141526535897932384626;
    for (int i = 0; i < n; i++) {
        ans *= 0.5;
    }
    return ans;
}

int main(int narg, char *argv[]) {
    QuESTEnv Env = createQuESTEnv();

    int n = 30;
    printf("n qubit(s): %d\n", n);

    Qureg q = createQureg(n, Env);

    float q_measure[n];
    int c[n];
    for (int i = 0; i < n; i++)
        c[i] = 0;

    /* GHZ quantum circuit */
    hadamard(q, 0);
    for (int i = 1; i < n; i++) {
        controlledNot(q, i - 1, i);
    }
    /* end of GHZ circuit */

    /* QFT starts */
    hadamard(q, 0);
    for (int i = 1; i < n - 1; i++) {
        for (int j = 0; j < i; j++) {
            controlledRotateZ(q, j, i, piOn2n(i - j));
        }
        hadamard(q, i);
    }
    /* end of QFT circuit */

    searchQuregTaskPartitions(q, Env);
    double t1 = get_wall_time();
    getAmp(q, 0);
    double t2 = get_wall_time();
    for (long long int i = 0; i < n; ++i) {
        q_measure[i] = calcProbOfOutcome(q, i, 1);
        printf("  probability for q[%2lld]==1 : %lf    \n", i, q_measure[i]);
    }
    printf("\n");

    for (int i = 0; i < 10; ++i) {
        Complex amp = getAmp(q, i);
        printf("Amplitude of %dth state vector: %12.6f,%12.6f\n", i, amp.real, amp.imag);
    }
    for (long long i = (1llu << n) - 1; i < (1llu << n); ++i) {
        Complex amp = getAmp(q, i);
        printf("Amplitude of %lldth state vector: %12.6f,%12.6f\n", i, amp.real, amp.imag);
    }
    printf("Complete the simulation takes time %12.6f seconds.\n", t2 - t1);

    destroyQureg(q, Env);
    destroyQuESTEnv(Env);

    return 0;
}
