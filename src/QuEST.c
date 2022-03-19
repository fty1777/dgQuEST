// Distributed under MIT licence. See https://github.com/QuEST-Kit/QuEST/blob/master/LICENCE.txt for details

/** @file
 * Implements the QuEST.h API (and some debugging functions) in a hardware-agnostic way,
 * for both pure and mixed states. These functions mostly wrap hardware-specific functions,
 * and should never call eachother.
 *
 * Density matrices rho of N qubits are flattened to appear as state-vectors |s> of 2N qubits.
 * Operations U rho U^dag are implemented as U^* U |s> and make use of the pure state backend,
 * and often don't need to explicitly compute U^*.
 */

#include "QuEST.h"

#include <stdio.h>

#include "QuEST_internal.h"
#include "QuEST_precision.h"
#include "QuEST_validation.h"

#ifdef __cplusplus
extern "C" {
#endif


/*
 * state-vector management
 */

Qureg createQureg(int numQubits, QuESTEnv env) {
    validateCreateNumQubits(numQubits, __func__);

    Qureg qureg;
    qureg.isDensityMatrix = 0;
    qureg.numQubitsRepresented = numQubits;
    qureg.numQubitsInStateVec = numQubits;
    statevec_createQureg(&qureg, numQubits, env);

    initZeroState(qureg);
    return qureg;
}

void searchQuregTaskPartitions(Qureg qureg, QuESTEnv env) {
    statevec_searchQuregTaskPartitions(qureg, env);
}

void destroyQureg(Qureg qureg, QuESTEnv env) {
    statevec_destroyQureg(qureg, env);
}


/*
 * state initialisation
 */

void initZeroState(Qureg qureg) {
    statevec_initClassicalState(qureg, 0); // valid for both statevec and density matrices
}

void initPlusState(Qureg qureg) {
    if (qureg.isDensityMatrix)
        densmatr_initPlusState(qureg);
    else
        statevec_initPlusState(qureg);
}

void initClassicalState(Qureg qureg, long long int stateInd) {
    validateStateIndex(qureg, stateInd, __func__);

    if (qureg.isDensityMatrix)
        densmatr_initClassicalState(qureg, stateInd);
    else
        statevec_initClassicalState(qureg, stateInd);
}

void initPureState(Qureg qureg, Qureg pure) {
    validateSecondQuregStateVec(pure, __func__);
    validateMatchingQuregDims(qureg, pure, __func__);

    printf("NOT IMPLEMENTED %s %d", __func__, __LINE__);
}

void initStateFromAmps(Qureg qureg, qreal* reals, qreal* imags) {
    validateStateVecQureg(qureg, __func__);

    statevec_initStateFromAmps(qureg, reals, imags);
}

/*
 * unitary gates
 */

void hadamard(Qureg qureg, const int targetQubit) {
    validateTarget(qureg, targetQubit, __func__);

    statevec_hadamard(qureg, targetQubit);
    if (qureg.isDensityMatrix) {
        statevec_hadamard(qureg, targetQubit + qureg.numQubitsRepresented);
    }
}

void rotateX(Qureg qureg, const int targetQubit, qreal angle) {
    validateTarget(qureg, targetQubit, __func__);

    statevec_rotateX(qureg, targetQubit, angle);
    if (qureg.isDensityMatrix) {
        statevec_rotateX(qureg, targetQubit + qureg.numQubitsRepresented, -angle);
    }
}

void rotateY(Qureg qureg, const int targetQubit, qreal angle) {
    validateTarget(qureg, targetQubit, __func__);

    statevec_rotateY(qureg, targetQubit, angle);
    if (qureg.isDensityMatrix) {
        statevec_rotateY(qureg, targetQubit + qureg.numQubitsRepresented, angle);
    }
}

void rotateZ(Qureg qureg, const int targetQubit, qreal angle) {
    validateTarget(qureg, targetQubit, __func__);

    statevec_rotateZ(qureg, targetQubit, angle);
    if (qureg.isDensityMatrix) {
        statevec_rotateZ(qureg, targetQubit + qureg.numQubitsRepresented, -angle);
    }
}

void controlledRotateX(Qureg qureg, const int controlQubit, const int targetQubit, qreal angle) {
    validateControlTarget(qureg, controlQubit, targetQubit, __func__);

    statevec_controlledRotateX(qureg, controlQubit, targetQubit, angle);
    if (qureg.isDensityMatrix) {
        int shift = qureg.numQubitsRepresented;
        statevec_controlledRotateX(qureg, controlQubit + shift, targetQubit + shift, -angle);
    }
}

void controlledRotateY(Qureg qureg, const int controlQubit, const int targetQubit, qreal angle) {
    validateControlTarget(qureg, controlQubit, targetQubit, __func__);

    statevec_controlledRotateY(qureg, controlQubit, targetQubit, angle);
    if (qureg.isDensityMatrix) {
        int shift = qureg.numQubitsRepresented;
        statevec_controlledRotateY(qureg, controlQubit + shift, targetQubit + shift, angle); // rotateY is real
    }
}

void controlledRotateZ(Qureg qureg, const int controlQubit, const int targetQubit, qreal angle) {
    validateControlTarget(qureg, controlQubit, targetQubit, __func__);

    statevec_controlledRotateZ(qureg, controlQubit, targetQubit, angle);
    if (qureg.isDensityMatrix) {
        int shift = qureg.numQubitsRepresented;
        statevec_controlledRotateZ(qureg, controlQubit + shift, targetQubit + shift, -angle);
    }
}

void unitary(Qureg qureg, const int targetQubit, ComplexMatrix2 u) {
    validateTarget(qureg, targetQubit, __func__);
    validateUnitaryMatrix(u, __func__);

    statevec_unitary(qureg, targetQubit, u);
    if (qureg.isDensityMatrix) {
        statevec_unitary(qureg, targetQubit + qureg.numQubitsRepresented, getConjugateMatrix(u));
    }
}

void controlledUnitary(Qureg qureg, const int controlQubit, const int targetQubit, ComplexMatrix2 u) {
    validateControlTarget(qureg, controlQubit, targetQubit, __func__);
    validateUnitaryMatrix(u, __func__);

    statevec_controlledUnitary(qureg, controlQubit, targetQubit, u);
    if (qureg.isDensityMatrix) {
        int shift = qureg.numQubitsRepresented;
        statevec_controlledUnitary(qureg, controlQubit + shift, targetQubit + shift, getConjugateMatrix(u));
    }
}

void compactUnitary(Qureg qureg, const int targetQubit, Complex alpha, Complex beta) {
    validateTarget(qureg, targetQubit, __func__);
    validateUnitaryComplexPair(alpha, beta, __func__);

    statevec_compactUnitary(qureg, targetQubit, alpha, beta);
    if (qureg.isDensityMatrix) {
        int shift = qureg.numQubitsRepresented;
        statevec_compactUnitary(qureg, targetQubit + shift, getConjugateScalar(alpha), getConjugateScalar(beta));
    }
}

void controlledCompactUnitary(Qureg qureg, const int controlQubit, const int targetQubit, Complex alpha, Complex beta) {
    validateControlTarget(qureg, controlQubit, targetQubit, __func__);
    validateUnitaryComplexPair(alpha, beta, __func__);

    statevec_controlledCompactUnitary(qureg, controlQubit, targetQubit, alpha, beta);
    if (qureg.isDensityMatrix) {
        int shift = qureg.numQubitsRepresented;
        statevec_controlledCompactUnitary(qureg,
                                          controlQubit + shift, targetQubit + shift,
                                          getConjugateScalar(alpha), getConjugateScalar(beta));
    }
}

void pauliX(Qureg qureg, const int targetQubit) {
    validateTarget(qureg, targetQubit, __func__);

    statevec_pauliX(qureg, targetQubit);
    if (qureg.isDensityMatrix) {
        statevec_pauliX(qureg, targetQubit + qureg.numQubitsRepresented);
    }
}

void pauliY(Qureg qureg, const int targetQubit) {
    validateTarget(qureg, targetQubit, __func__);

    statevec_pauliY(qureg, targetQubit);
    if (qureg.isDensityMatrix) {
        statevec_pauliYConj(qureg, targetQubit + qureg.numQubitsRepresented);
    }
}

void pauliZ(Qureg qureg, const int targetQubit) {
    validateTarget(qureg, targetQubit, __func__);

    statevec_pauliZ(qureg, targetQubit);
    if (qureg.isDensityMatrix) {
        statevec_pauliZ(qureg, targetQubit + qureg.numQubitsRepresented);
    }
}

void sGate(Qureg qureg, const int targetQubit) {
    validateTarget(qureg, targetQubit, __func__);

    statevec_sGate(qureg, targetQubit);
    if (qureg.isDensityMatrix) {
        statevec_sGateConj(qureg, targetQubit + qureg.numQubitsRepresented);
    }
}

void tGate(Qureg qureg, const int targetQubit) {
    validateTarget(qureg, targetQubit, __func__);

    statevec_tGate(qureg, targetQubit);
    if (qureg.isDensityMatrix) {
        statevec_tGateConj(qureg, targetQubit + qureg.numQubitsRepresented);
    }
}

void phaseShift(Qureg qureg, const int targetQubit, qreal angle) {
    validateTarget(qureg, targetQubit, __func__);

    statevec_phaseShift(qureg, targetQubit, angle);
    if (qureg.isDensityMatrix) {
        statevec_phaseShift(qureg, targetQubit + qureg.numQubitsRepresented, -angle);
    }
}

void controlledPhaseShift(Qureg qureg, const int idQubit1, const int idQubit2, qreal angle) {
    validateControlTarget(qureg, idQubit1, idQubit2, __func__);

    statevec_controlledPhaseShift(qureg, idQubit1, idQubit2, angle);
    if (qureg.isDensityMatrix) {
        int shift = qureg.numQubitsRepresented;
        statevec_controlledPhaseShift(qureg, idQubit1 + shift, idQubit2 + shift, -angle);
    }
}

void controlledNot(Qureg qureg, const int controlQubit, const int targetQubit) {
    validateControlTarget(qureg, controlQubit, targetQubit, __func__);

    statevec_controlledNot(qureg, controlQubit, targetQubit);
    if (qureg.isDensityMatrix) {
        int shift = qureg.numQubitsRepresented;
        statevec_controlledNot(qureg, controlQubit + shift, targetQubit + shift);
    }
}

void controlledPauliY(Qureg qureg, const int controlQubit, const int targetQubit) {
    validateControlTarget(qureg, controlQubit, targetQubit, __func__);

    statevec_controlledPauliY(qureg, controlQubit, targetQubit);
    if (qureg.isDensityMatrix) {
        int shift = qureg.numQubitsRepresented;
        statevec_controlledPauliYConj(qureg, controlQubit + shift, targetQubit + shift);
    }
}

void controlledPhaseFlip(Qureg qureg, const int idQubit1, const int idQubit2) {
    validateControlTarget(qureg, idQubit1, idQubit2, __func__);

    statevec_controlledPhaseFlip(qureg, idQubit1, idQubit2);
    if (qureg.isDensityMatrix) {
        int shift = qureg.numQubitsRepresented;
        statevec_controlledPhaseFlip(qureg, idQubit1 + shift, idQubit2 + shift);
    }
}

void rotateAroundAxis(Qureg qureg, const int rotQubit, qreal angle, Vector axis) {
    validateTarget(qureg, rotQubit, __func__);
    validateVector(axis, __func__);

    statevec_rotateAroundAxis(qureg, rotQubit, angle, axis);
    if (qureg.isDensityMatrix) {
        int shift = qureg.numQubitsRepresented;
        statevec_rotateAroundAxisConj(qureg, rotQubit + shift, angle, axis);
    }
}

void controlledRotateAroundAxis(Qureg qureg, const int controlQubit, const int targetQubit, qreal angle, Vector axis) {
    validateControlTarget(qureg, controlQubit, targetQubit, __func__);
    validateVector(axis, __func__);

    statevec_controlledRotateAroundAxis(qureg, controlQubit, targetQubit, angle, axis);
    if (qureg.isDensityMatrix) {
        int shift = qureg.numQubitsRepresented;
        statevec_controlledRotateAroundAxisConj(qureg, controlQubit + shift, targetQubit + shift, angle, axis);
    }
}

void u1Gate(Qureg qureg, int targetQubit, qreal lambda) {
    validateTarget(qureg, targetQubit, __func__);

    qreal cos_ = cos(lambda), sin_ = sin(lambda);

    ComplexMatrix2 u = {
            .r0c0 = {1.0, 0.0},
            .r0c1 = {0.0, 0.0},
            .r1c0 = {0.0, 0.0},
            .r1c1 = {cos_, sin_},
    };
    validateUnitaryMatrix(u, __func__);

    statevec_unitary(qureg, targetQubit, u);
}

void u2Gate(Qureg qureg, int targetQubit, qreal phi, qreal lambda) {
    validateTarget(qureg, targetQubit, __func__);

    qreal Inv_sqrt2 = 1 / sqrt(2);
    qreal cos_phi = cos(phi) * Inv_sqrt2, sin_phi = sin(phi) * Inv_sqrt2;
    qreal cos_lambda = cos(lambda) * Inv_sqrt2, sin_lambda = sin(lambda) * Inv_sqrt2;
    qreal cos_ = cos(phi + lambda) * Inv_sqrt2, sin_ = sin(phi + lambda) * Inv_sqrt2;

    ComplexMatrix2 u = {
            .r0c0 = {Inv_sqrt2, 0.0},
            .r0c1 = {-cos_lambda, -sin_lambda},
            .r1c0 = {cos_phi, sin_phi},
            .r1c1 = {cos_, sin_}};
    validateUnitaryMatrix(u, __func__);

    statevec_unitary(qureg, targetQubit, u);
}

void u3Gate(Qureg qureg, int targetQubit, qreal theta, qreal phi, qreal lambda) {
    validateTarget(qureg, targetQubit, __func__);

    qreal cos_theta = cos(theta / 2), sin_theta = sin(theta / 2);
    qreal cos_phi = cos(phi), sin_phi = sin(phi);
    qreal cos_lambda = cos(lambda), sin_lambda = sin(lambda);
    qreal cos_ = cos(phi + lambda), sin_ = sin(phi + lambda);

    ComplexMatrix2 u = {
            .r0c0 = {cos_theta, 0.0},
            .r0c1 = {-cos_lambda * sin_theta, -sin_lambda * sin_theta},
            .r1c0 = {cos_phi * sin_theta, sin_phi * sin_theta},
            .r1c1 = {cos_ * cos_theta, sin_ * cos_theta},
    };
    validateUnitaryMatrix(u, __func__);

    statevec_unitary(qureg, targetQubit, u);
}

void SqX(Qureg qureg, int targetQubit) {
    validateTarget(qureg, targetQubit, __func__);

    qreal Inv_sqrt2 = 1 / sqrt(2);
    ComplexMatrix2 u = {
            .r0c0 = {Inv_sqrt2, 0.0},
            .r0c1 = {0.0, -Inv_sqrt2},
            .r1c0 = {0.0, -Inv_sqrt2},
            .r1c1 = {Inv_sqrt2, 0.0},
    };
    validateUnitaryMatrix(u, __func__);

    statevec_unitary(qureg, targetQubit, u);
}

void SqY(Qureg qureg, int targetQubit) {
    validateTarget(qureg, targetQubit, __func__);

    qreal Inv_sqrt2 = 1 / sqrt(2);
    ComplexMatrix2 u = {
            .r0c0 = {Inv_sqrt2, 0.0},
            .r0c1 = {-Inv_sqrt2, 0.0},
            .r1c0 = {Inv_sqrt2, 0.0},
            .r1c1 = {Inv_sqrt2, 0.0},
    };
    validateUnitaryMatrix(u, __func__);

    statevec_unitary(qureg, targetQubit, u);
}

void SqW(Qureg qureg, int targetQubit) {
    validateTarget(qureg, targetQubit, __func__);

    qreal Inv_sqrt2 = 1 / sqrt(2);
    ComplexMatrix2 u = {
            .r0c0 = {Inv_sqrt2, 0.0},
            .r0c1 = {-0.5, -0.5},
            .r1c0 = {0.5, -0.5},
            .r1c1 = {Inv_sqrt2, 0.0},
    };
    validateUnitaryMatrix(u, __func__);

    statevec_unitary(qureg, targetQubit, u);
}

void fSim(Qureg qureg, int targetQubit1, int targetQubit2, qreal theta, qreal phi) {
    controlledPhaseShift(qureg, targetQubit1, targetQubit2, -phi);
    controlledNot(qureg, targetQubit1, targetQubit2);
    controlledRotateX(qureg, targetQubit2, targetQubit1, theta * 2);
    controlledNot(qureg, targetQubit1, targetQubit2);
}

/*
 * register attributes
 */

int getNumQubits(Qureg qureg) {
    return qureg.numQubitsRepresented;
}

int getNumAmps(Qureg qureg) {
    validateStateVecQureg(qureg, __func__);

    return qureg.numAmpsTotal;
}

Complex getAmp(Qureg qureg, long long int index) {
    validateStateVecQureg(qureg, __func__);
    validateStateIndex(qureg, index, __func__);

    return statevec_getAmp(qureg, index);
}

/*
 * calculations
 */

qreal calcProbOfOutcome(Qureg qureg, const int measureQubit, int outcome) {
    validateTarget(qureg, measureQubit, __func__);
    validateOutcome(outcome, __func__);
    return statevec_calcProbOfOutcome(qureg, measureQubit, outcome);
}

int getQuEST_PREC(void) {
    return sizeof(qreal) / 4;
}


#ifdef __cplusplus
}
#endif
