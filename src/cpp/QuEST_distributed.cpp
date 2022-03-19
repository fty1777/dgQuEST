// Distributed under MIT licence. See https://github.com/fty1777/dgQuEST/blob/main/LICENSE for details

#include <easylogging++.h>
#include <fmt/core.h>
#include <mpi.h>

#include "../mt19937ar.h"
#include "QuEST_backend.h"
#include "timer.hpp"

INITIALIZE_EASYLOGGINGPP

QuESTEnv createQuESTEnv() {
    seedQuESTDefault();

    el::Loggers::addFlag(el::LoggingFlag::ColoredTerminalOutput);
    el::Loggers::setLoggingLevel(el::Level::Warning);

    int initialized;
    MPI_CHECK(MPI_Initialized(&initialized));
    if (!initialized) {
        int provided;
        MPI_CHECK(MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided));
        if (provided != MPI_THREAD_MULTIPLE) {
            LOG(FATAL) << "provided is not MPI_THREAD_MULTIPLE";
            exit(EXIT_FAILURE);
        }
    } else {
        LOG(FATAL) << "ERROR: Trying to initialize QuESTEnv multiple times. Ignoring...";
    }

    int numRanks;
    int numDevices;
    int rank;

    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numRanks));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    LOG(INFO) << fmt::format("{} NODEs WITH {} GPUs\n", numRanks, numDevices);

    auto fconf = fopen("QuEST.conf", "r");
    int minK;
    int maxK;
    if (fscanf(fconf, "%d%d", &minK, &maxK) != 2) {
        LOG(FATAL) << "Invalid configure for K search range";
    }

    QuESTEnvValueDistributed *env;
    int designatedK;
    if (fscanf(fconf, "%d", &designatedK) == 1) {
        env = new QuESTEnvValueDistributed(numDevices, minK, maxK, designatedK);
    } else {
        env = new QuESTEnvValueDistributed(numDevices, minK, maxK);
    }
    fclose(fconf);

    return {env, rank};
}

void statevec_createQureg(Qureg* qureg_ptr, int numQubits, QuESTEnv env) {
    env.value->statevec_createQureg(*qureg_ptr, numQubits);
}

void statevec_searchQuregTaskPartitions(Qureg qureg, QuESTEnv env) {
    env.value->searchTaskPartition(qureg);
}

void statevec_destroyQureg(Qureg qureg, QuESTEnv env) {
    env.value->statevec_destroyQureg(qureg);
}

void destroyQuESTEnv(QuESTEnv env) {
    delete env.value;
}

void seedQuESTDefault() {
    // init MT random number generator with three keys -- time and pid
    // for the MPI version, it is ok that all procs will get the same seed as random numbers will only be
    // used by the master process
    unsigned long int key[2];
    getQuESTDefaultSeedKey(key);
    init_by_array(key, 2);
}

void statevec_initClassicalState(Qureg qureg, long long int stateInd) {
    qureg.state->statevec_initClassicalState(stateInd);
}

void densmatr_initClassicalState(Qureg qureg, long long int stateInd) {
    qureg.state->densmatr_initClassicalState(stateInd);
}

void statevec_initPlusState(Qureg qureg) {
    qureg.state->statevec_initPlusState();
}

void densmatr_initPlusState(Qureg qureg) {
    qureg.state->densmatr_initPlusState();
}

void statevec_initStateFromAmps(Qureg qureg, qreal* reals, qreal* imags) {
    qureg.state->statevec_initStateFromAmps(reals, imags);
}

Complex statevec_getAmp(Qureg qureg, long long int index) {
    qureg.state->splitCircuit();
    qureg.env.value->calculateAllSections(qureg);
    return qureg.env.value->statevec_getAmp(qureg, index);
}

qreal statevec_calcProbOfOutcome(Qureg qureg, const int measureQubit, int outcome) {
    qureg.state->splitCircuit();
    qureg.env.value->calculateAllSections(qureg);
    return qureg.env.value->statevec_calcProbOfOutcome(qureg, measureQubit, outcome);
}

void statevec_pauliZ(Qureg qureg, const int targetQubit) {
    Gate gate;
    gate.type = OP_PAULI_Z;
    gate.targetMask = 0;
    gate.controlMask = 1ULL << targetQubit;
    qureg.state->addGate(gate);
}

void statevec_sGate(Qureg qureg, const int targetQubit) {
    Gate gate;
    gate.type = OP_S_GATE;
    gate.targetMask = 0;
    gate.controlMask = 1ULL << targetQubit;
    qureg.state->addGate(gate);
}

void statevec_sGateConj(Qureg qureg, const int targetQubit) {
    Gate gate;
    gate.type = OP_S_GATE_CONJ;
    gate.targetMask = 0;
    gate.controlMask = 1ULL << targetQubit;
    qureg.state->addGate(gate);
}

void statevec_tGate(Qureg qureg, const int targetQubit) {
    Gate gate;
    gate.type = OP_T_GATE;
    gate.targetMask = 0;
    gate.controlMask = 1ULL << targetQubit;
    qureg.state->addGate(gate);
}

void statevec_tGateConj(Qureg qureg, const int targetQubit) {
    Gate gate;
    gate.type = OP_T_GATE_CONJ;
    gate.targetMask = 0;
    gate.controlMask = 1ULL << targetQubit;
    qureg.state->addGate(gate);
}

void statevec_phaseShift(Qureg qureg, const int targetQubit, qreal angle) {
    Gate gate;
    gate.type = OP_PHASE_SHIFT;
    gate.targetMask = 0;
    gate.controlMask = 1ULL << targetQubit;
    gate.operand.angle = angle;
    qureg.state->addGate(gate);
}

void statevec_rotateZ(Qureg qureg, const int rotQubit, qreal angle) {
    Gate gate;
    gate.type = OP_ROTATE_Z;
    gate.targetMask = 1ULL << rotQubit;
    gate.controlMask = 0;
    gate.operand.angle = angle * 0.5;
    qureg.state->addGate(gate);
}

void statevec_pauliX(Qureg qureg, const int targetQubit) {
    Gate gate;
    gate.type = OP_PAULI_X;
    gate.targetMask = 1ULL << targetQubit;
    gate.controlMask = 0;
    qureg.state->addGate(gate);
}

void statevec_pauliY(Qureg qureg, const int targetQubit) {
    Gate gate;
    gate.type = OP_PAULI_Y;
    gate.targetMask = 1ULL << targetQubit;
    gate.controlMask = 0;
    qureg.state->addGate(gate);
}

void statevec_pauliYConj(Qureg qureg, const int targetQubit) {
    Gate gate;
    gate.type = OP_PAULI_Y_CONJ;
    gate.targetMask = 1ULL << targetQubit;
    gate.controlMask = 0;
    qureg.state->addGate(gate);
}

void statevec_hadamard(Qureg qureg, const int targetQubit) {
    Gate gate;
    gate.type = OP_HADAMARD;
    gate.targetMask = 1ULL << targetQubit;
    gate.controlMask = 0;
    qureg.state->addGate(gate);
}

void statevec_rotateX(Qureg qureg, const int rotQubit, qreal angle) {
    Gate gate;
    gate.type = OP_ROTATE_X;
    gate.targetMask = 1ULL << rotQubit;
    gate.controlMask = 0;
    gate.operand.angle = angle * 0.5;
    qureg.state->addGate(gate);
}

void statevec_rotateY(Qureg qureg, const int rotQubit, qreal angle) {
    Gate gate;
    gate.type = OP_ROTATE_Y;
    gate.targetMask = 1ULL << rotQubit;
    gate.controlMask = 0;
    gate.operand.angle = angle * 0.5;
    qureg.state->addGate(gate);
}

void statevec_compactUnitary(Qureg qureg, const int targetQubit, Complex alpha, Complex beta) {
    Gate gate;
    gate.type = OP_COMPACT_UNITARY;
    gate.targetMask = 1ULL << targetQubit;
    gate.controlMask = 0;
    gate.operand.alpha = alpha;
    gate.operand.beta = beta;
    qureg.state->addGate(gate);
}

void statevec_unitary(Qureg qureg, const int targetQubit, ComplexMatrix2 u) {
    Gate gate;
    gate.type = OP_UNITARY;
    gate.targetMask = 1ULL << targetQubit;
    gate.controlMask = 0;
    gate.operand.unitaryMatrix = u;
    qureg.state->addGate(gate);
}

void statevec_controlledPhaseFlip(Qureg qureg, const int idQubit1, const int idQubit2) {
    Gate gate;
    gate.type = OP_PAULI_Z;
    gate.targetMask = 0;
    gate.controlMask = (1ULL << idQubit1) | (1ULL << idQubit2);
    qureg.state->addGate(gate);
}

void statevec_controlledPhaseShift(Qureg qureg, const int idQubit1, const int idQubit2, qreal angle) {
    Gate gate;
    gate.type = OP_PHASE_SHIFT;
    gate.targetMask = 0;
    gate.controlMask = (1ULL << idQubit1) | (1ULL << idQubit2);
    gate.operand.angle = angle;
    qureg.state->addGate(gate);
}

void statevec_controlledRotateZ(Qureg qureg, const int controlQubit, const int targetQubit, qreal angle) {
    Gate gate;
    gate.type = OP_ROTATE_Z;
    gate.targetMask = 1ULL << targetQubit;
    gate.controlMask = 1ULL << controlQubit;
    gate.operand.angle = angle * 0.5;
    qureg.state->addGate(gate);
}

void statevec_controlledNot(Qureg qureg, const int controlQubit, const int targetQubit) {
    Gate gate;
    gate.type = OP_PAULI_X;
    gate.targetMask = 1ULL << targetQubit;
    gate.controlMask = 1ULL << controlQubit;
    qureg.state->addGate(gate);
}

void statevec_controlledPauliY(Qureg qureg, const int controlQubit, const int targetQubit) {
    Gate gate;
    gate.type = OP_PAULI_Y;
    gate.targetMask = 1ULL << targetQubit;
    gate.controlMask = 1ULL << controlQubit;
    qureg.state->addGate(gate);
}

void statevec_controlledPauliYConj(Qureg qureg, const int controlQubit, const int targetQubit) {
    Gate gate;
    gate.type = OP_PAULI_Y_CONJ;
    gate.targetMask = 1ULL << targetQubit;
    gate.controlMask = 1ULL << controlQubit;
    qureg.state->addGate(gate);
}

void statevec_controlledRotateX(Qureg qureg, const int controlQubit, const int targetQubit, qreal angle) {
    Gate gate;
    gate.type = OP_ROTATE_X;
    gate.targetMask = 1ULL << targetQubit;
    gate.controlMask = 1ULL << controlQubit;
    gate.operand.angle = angle * 0.5;
    qureg.state->addGate(gate);
}

void statevec_controlledRotateY(Qureg qureg, const int controlQubit, const int targetQubit, qreal angle) {
    Gate gate;
    gate.type = OP_ROTATE_Y;
    gate.targetMask = 1ULL << targetQubit;
    gate.controlMask = 1ULL << controlQubit;
    gate.operand.angle = angle * 0.5;
    qureg.state->addGate(gate);
}

void statevec_controlledCompactUnitary(Qureg qureg, const int controlQubit, const int targetQubit, Complex alpha, Complex beta) {
    Gate gate;
    gate.type = OP_COMPACT_UNITARY;
    gate.targetMask = 1ULL << targetQubit;
    gate.controlMask = 1ULL << controlQubit;
    gate.operand.alpha = alpha;
    gate.operand.beta = beta;
    qureg.state->addGate(gate);
}

void statevec_controlledUnitary(Qureg qureg, const int controlQubit, const int targetQubit, ComplexMatrix2 u) {
    Gate gate;
    gate.type = OP_UNITARY;
    gate.targetMask = 1ULL << targetQubit;
    gate.controlMask = 1ULL << controlQubit;
    gate.operand.unitaryMatrix = u;
    qureg.state->addGate(gate);
}

static u64 calculateControlMask(const int* controlQubits, int numControlQubits) {
    u64 controlMask = 0;
    for (int i = 0; i < numControlQubits; i++) {
        controlMask |= 1ULL << controlQubits[i];
    }
    return controlMask;
}

void multiControlledPhaseFlip(Qureg qureg, int* controlQubits, int numControlQubits) {
    Gate gate;
    gate.type = OP_PAULI_Z;
    gate.targetMask = 0;
    gate.controlMask = calculateControlMask(controlQubits, numControlQubits);
    qureg.state->addGate(gate);
}

void multiControlledPhaseShift(Qureg qureg, int* controlQubits, int numControlQubits, qreal angle) {
    Gate gate;
    gate.type = OP_PHASE_SHIFT;
    gate.targetMask = 0;
    gate.controlMask = calculateControlMask(controlQubits, numControlQubits);
    gate.operand.angle = angle;
    qureg.state->addGate(gate);
}

void multiControlledUnitary(Qureg qureg, int* controlQubits, const int numControlQubits, const int targetQubit, ComplexMatrix2 u) {
    Gate gate;
    gate.type = OP_UNITARY;
    gate.targetMask = 1ULL << targetQubit;
    gate.controlMask = calculateControlMask(controlQubits, numControlQubits);
    gate.operand.unitaryMatrix = u;
    qureg.state->addGate(gate);
}