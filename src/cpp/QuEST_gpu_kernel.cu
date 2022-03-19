// Distributed under MIT licence. See https://github.com/fty1777/dgQuEST/blob/main/LICENSE for details

#undef DEBUG

#include <cuda_pipeline_primitives.h>
#include <easylogging++.h>
#include <fmt/core.h>

#include <iostream>
#include <memory>

#include "QuEST_backend.h"
#include "timer.hpp"

#define MATCH_BIT(x, target) (((x) & (target)) == (target))

__forceinline__ __device__ void statevec_pauliZ_func_reg(qreal &real, qreal &imag) {
    qreal t_real = real;
    qreal t_imag = imag;
    real = -t_real;
    imag = -t_imag;
}

__forceinline__ __device__ void statevec_sGate_func_reg(qreal &real, qreal &imag) {
    qreal t_real = real;
    qreal t_imag = imag;
    real = -t_imag;
    imag = +t_real;
}

__forceinline__ __device__ void statevec_sGateConj_func_reg(qreal &real, qreal &imag) {
    qreal t_real = real;
    qreal t_imag = imag;
    real = +t_imag;
    imag = -t_real;
}

__forceinline__ __device__ void statevec_tGate_func_reg(qreal &real, qreal &imag) {
    qreal t_real = real;
    qreal t_imag = imag;
    real = rsqrt2 * (t_real - t_imag);
    imag = rsqrt2 * (t_imag + t_real);
}

__forceinline__ __device__ void statevec_tGateConj_func_reg(qreal &real, qreal &imag) {
    qreal t_real = real;
    qreal t_imag = imag;
    real = rsqrt2 * (t_real + t_imag);
    imag = rsqrt2 * (t_imag - t_real);
}

__forceinline__ __device__ void statevec_tGate_neg_func_reg(qreal &real, qreal &imag) {
    qreal t_real = real;
    qreal t_imag = imag;
    real = -rsqrt2 * (t_real - t_imag);
    imag = -rsqrt2 * (t_imag + t_real);
}

__forceinline__ __device__ void statevec_tGateConj_neg_func_reg(qreal &real, qreal &imag) {
    qreal t_real = real;
    qreal t_imag = imag;
    real = -rsqrt2 * (t_real + t_imag);
    imag = -rsqrt2 * (t_imag - t_real);
}

__forceinline__ __device__ void statevec_phaseShift_func_reg(qreal &real, qreal &imag, qreal cosAngle, qreal sinAngle) {
    qreal t_real = real;
    qreal t_imag = imag;
    real = cosAngle * t_real - sinAngle * t_imag;
    imag = cosAngle * t_imag + sinAngle * t_real;
}

__forceinline__ __device__ void statevec_rotateZ_func_reg(qreal &real0, qreal &imag0, qreal &real1, qreal &imag1, qreal cosAngle, qreal sinAngle) {
    qreal t_real0 = real0;
    qreal t_imag0 = imag0;
    qreal t_real1 = real1;
    qreal t_imag1 = imag1;
    real0 = cosAngle * t_real0 + sinAngle * t_imag0;
    imag0 = cosAngle * t_imag0 - sinAngle * t_real0;
    real1 = cosAngle * t_real1 - sinAngle * t_imag1;
    imag1 = cosAngle * t_imag1 + sinAngle * t_real1;
}

__forceinline__ __device__ void statevec_pauliX_func_reg(qreal &real0, qreal &imag0, qreal &real1, qreal &imag1) {
    qreal t_real0 = real0;
    qreal t_imag0 = imag0;
    qreal t_real1 = real1;
    qreal t_imag1 = imag1;
    real0 = t_real1;
    imag0 = t_imag1;
    real1 = t_real0;
    imag1 = t_imag0;
}

__forceinline__ __device__ void statevec_pauliY_func_reg(qreal &real0, qreal &imag0, qreal &real1, qreal &imag1) {
    qreal t_real0 = real0;
    qreal t_imag0 = imag0;
    qreal t_real1 = real1;
    qreal t_imag1 = imag1;
    real0 = +t_imag1;
    imag0 = -t_real1;
    real1 = -t_imag0;
    imag1 = +t_real0;
}

__forceinline__ __device__ void statevec_pauliYConj_func_reg(qreal &real0, qreal &imag0, qreal &real1, qreal &imag1) {
    qreal t_real0 = real0;
    qreal t_imag0 = imag0;
    qreal t_real1 = real1;
    qreal t_imag1 = imag1;
    real0 = -t_imag1;
    imag0 = +t_real1;
    real1 = +t_imag0;
    imag1 = -t_real0;
}

__forceinline__ __device__ void statevec_hadamard_func_reg(qreal &real0, qreal &imag0, qreal &real1, qreal &imag1) {
    qreal t_real0 = real0;
    qreal t_imag0 = imag0;
    qreal t_real1 = real1;
    qreal t_imag1 = imag1;
    real0 = rsqrt2 * (t_real0 + t_real1);
    imag0 = rsqrt2 * (t_imag0 + t_imag1);
    real1 = rsqrt2 * (t_real0 - t_real1);
    imag1 = rsqrt2 * (t_imag0 - t_imag1);
}

__forceinline__ __device__ void statevec_rotateX_func_reg(qreal &real0, qreal &imag0, qreal &real1, qreal &imag1, qreal cosAngle, qreal sinAngle) {
    qreal t_real0 = real0;
    qreal t_imag0 = imag0;
    qreal t_real1 = real1;
    qreal t_imag1 = imag1;
    real0 = cosAngle * t_real0 + sinAngle * t_imag1;
    imag0 = cosAngle * t_imag0 - sinAngle * t_real1;
    real1 = cosAngle * t_real1 + sinAngle * t_imag0;
    imag1 = cosAngle * t_imag1 - sinAngle * t_real0;
}

__forceinline__ __device__ void statevec_rotateY_func_reg(qreal &real0, qreal &imag0, qreal &real1, qreal &imag1, qreal cosAngle, qreal sinAngle) {
    qreal t_real0 = real0;
    qreal t_imag0 = imag0;
    qreal t_real1 = real1;
    qreal t_imag1 = imag1;
    real0 = cosAngle * t_real0 - sinAngle * t_real1;
    imag0 = cosAngle * t_imag0 - sinAngle * t_imag1;
    real1 = sinAngle * t_real0 + cosAngle * t_real1;
    imag1 = sinAngle * t_imag0 + cosAngle * t_imag1;
}

__forceinline__ __device__ void statevec_compactUnitary_func_reg(qreal &real0, qreal &imag0, qreal &real1, qreal &imag1, const Complex &alpha, const Complex &beta) {
    qreal t_real0 = real0;
    qreal t_imag0 = imag0;
    qreal t_real1 = real1;
    qreal t_imag1 = imag1;
    real0 = alpha.real * t_real0 - alpha.imag * t_imag0 - beta.real * t_real1 - beta.imag * t_imag1;
    imag0 = alpha.real * t_imag0 + alpha.imag * t_real0 - beta.real * t_imag1 + beta.imag * t_real1;
    real1 = beta.real * t_real0 - beta.imag * t_imag0 + alpha.real * t_real1 + alpha.imag * t_imag1;
    imag1 = beta.real * t_imag0 + beta.imag * t_real0 + alpha.real * t_imag1 - alpha.imag * t_real1;
}

__forceinline__ __device__ void statevec_unitary_func_reg(qreal &real0, qreal &imag0, qreal &real1, qreal &imag1, const ComplexMatrix2 &u) {
    qreal t_real0 = real0;
    qreal t_imag0 = imag0;
    qreal t_real1 = real1;
    qreal t_imag1 = imag1;
    real0 = u.r0c0.real * t_real0 - u.r0c0.imag * t_imag0 + u.r0c1.real * t_real1 - u.r0c1.imag * t_imag1;
    imag0 = u.r0c0.real * t_imag0 + u.r0c0.imag * t_real0 + u.r0c1.real * t_imag1 + u.r0c1.imag * t_real1;
    real1 = u.r1c0.real * t_real0 - u.r1c0.imag * t_imag0 + u.r1c1.real * t_real1 - u.r1c1.imag * t_imag1;
    imag1 = u.r1c0.real * t_imag0 + u.r1c0.imag * t_real0 + u.r1c1.real * t_imag1 + u.r1c1.imag * t_real1;
}

FusedOp hostFusedOperators[4];
__constant__ PartitionedFusedOperator devicePartitionedFusedOperator;

__global__ void statevec_fusedKernel(qComplex *stateVec, int numQubits) {
    __shared__ qComplex stateVecShared[SHARED_MEMORY_LENGTH];
    __shared__ u32 qubit[SHARED_MEMORY_LENGTH];

    if (threadIdx.x < (1 << (10 - COALESCED_QUBITS))) {
        u32 mi0 = devicePartitionedFusedOperator.memsel[0];
        u32 qb0 = (blockIdx.x & ((1 << (mi0 - COALESCED_QUBITS)) - 1)) << COALESCED_QUBITS;
        u32 qbb = blockIdx.x >> (mi0 - COALESCED_QUBITS);
        u32 qbt = threadIdx.x;

        for (int i = 1; i < SHARED_MEMORY_QUBITS - COALESCED_QUBITS; i++) {
            qb0 |= (qbt & 1) << mi0;
            qbt >>= 1;
            mi0 += 1;

            int mi1 = devicePartitionedFusedOperator.memsel[i];
            qb0 |= (qbb & ((1 << (mi1 - mi0)) - 1)) << mi0;
            qbb >>= mi1 - mi0;
            mi0 = mi1;
        }

        qb0 |= (qbb & ((1 << (numQubits - mi0)) - 1)) << (mi0 + 1);
        u32 qb1 = qb0 | (1 << mi0);

        qubit[threadIdx.x << COALESCED_QUBITS] = qb0;
        qubit[(threadIdx.x << COALESCED_QUBITS) + blockDim.x] = qb1;
    }

    __syncthreads();
    const u32 offset0 = qubit[threadIdx.x & ((1 << 10) - (1 << COALESCED_QUBITS))] | (threadIdx.x & ((1 << COALESCED_QUBITS) - 1));
    const u32 offset1 = qubit[(threadIdx.x & ((1 << 10) - (1 << COALESCED_QUBITS))) + blockDim.x] | (threadIdx.x & ((1 << COALESCED_QUBITS) - 1));
    const u32 index0 = threadIdx.x;
    const u32 index1 = index0 + blockDim.x;

    switch (devicePartitionedFusedOperator.initFunction) {
        case INIT_ALL_ZERO:
            stateVecShared[index0] = {0.0, 0.0};
            stateVecShared[index1] = {0.0, 0.0};
            break;
        case INIT_CLASSICAL_STATE:
            stateVecShared[index0] = {offset0 == devicePartitionedFusedOperator.initParam.stateIndex ? 1.0 : 0.0, 0.0};
            stateVecShared[index1] = {offset1 == devicePartitionedFusedOperator.initParam.stateIndex ? 1.0 : 0.0, 0.0};
            break;
        case INIT_PLUS_STATE:
            stateVecShared[index0] = {devicePartitionedFusedOperator.initParam.probFactor, 0.0};
            stateVecShared[index1] = {devicePartitionedFusedOperator.initParam.probFactor, 0.0};
            break;
        case INIT_FROM_AMPS:
            __pipeline_memcpy_async(&stateVecShared[index0], &stateVec[offset0], sizeof(qComplex));
            __pipeline_memcpy_async(&stateVecShared[index1], &stateVec[offset1], sizeof(qComplex));
            break;
    }

    __pipeline_commit();
    qubit[index0] = offset0;
    qubit[index1] = offset1;
    __pipeline_wait_prior(0);
    __syncthreads();

    int pOpPtr = 0;
    for (int groupIdx = 0; groupIdx < devicePartitionedFusedOperator.numGroups; groupIdx++) {
        auto &group = devicePartitionedFusedOperator.groups[groupIdx];
        u32 targetMaskLocal = group.targetMaskLocal;
        u32 localIdx0 = ((threadIdx.x & (-targetMaskLocal)) << 1) | (threadIdx.x & (targetMaskLocal - 1));
        u32 localIdx1 = localIdx0 | targetMaskLocal;

        __syncthreads();
        qComplex val0 = stateVecShared[localIdx0];
        qComplex val1 = stateVecShared[localIdx1];
        qreal &real0 = val0.x;
        qreal &imag0 = val0.y;
        qreal &real1 = val1.x;
        qreal &imag1 = val1.y;
        for (int _ = 0; _ < group.size; _++) {
            PartialGateOp &op = devicePartitionedFusedOperator.pOps[pOpPtr++];
            u32 controlMask = op.controlMask;
            switch (op.type) {
                case OP_PAULI_Z:
                    if (MATCH_BIT(qubit[localIdx0], controlMask))
                        statevec_pauliZ_func_reg(real0, imag0);
                    if (MATCH_BIT(qubit[localIdx1], controlMask))
                        statevec_pauliZ_func_reg(real1, imag1);
                    break;
                case OP_S_GATE:
                    if (MATCH_BIT(qubit[localIdx0], controlMask))
                        statevec_sGate_func_reg(real0, imag0);
                    if (MATCH_BIT(qubit[localIdx1], controlMask))
                        statevec_sGate_func_reg(real1, imag1);
                    break;
                case OP_S_GATE_CONJ:
                    if (MATCH_BIT(qubit[localIdx0], controlMask))
                        statevec_sGateConj_func_reg(real0, imag0);
                    if (MATCH_BIT(qubit[localIdx1], controlMask))
                        statevec_sGateConj_func_reg(real1, imag1);
                    break;
                case OP_T_GATE:
                    if (MATCH_BIT(qubit[localIdx0], controlMask))
                        statevec_tGate_func_reg(real0, imag0);
                    if (MATCH_BIT(qubit[localIdx1], controlMask))
                        statevec_tGate_func_reg(real1, imag1);
                    break;
                case OP_T_GATE_CONJ:
                    if (MATCH_BIT(qubit[localIdx0], controlMask))
                        statevec_tGateConj_func_reg(real0, imag0);
                    if (MATCH_BIT(qubit[localIdx1], controlMask))
                        statevec_tGateConj_func_reg(real1, imag1);
                    break;
                case OP_T_GATE_NEG:
                    if (MATCH_BIT(qubit[localIdx0], controlMask))
                        statevec_tGate_neg_func_reg(real0, imag0);
                    if (MATCH_BIT(qubit[localIdx1], controlMask))
                        statevec_tGate_neg_func_reg(real1, imag1);
                    break;
                case OP_T_GATE_CONJ_NEG:
                    if (MATCH_BIT(qubit[localIdx0], controlMask))
                        statevec_tGateConj_neg_func_reg(real0, imag0);
                    if (MATCH_BIT(qubit[localIdx1], controlMask))
                        statevec_tGateConj_neg_func_reg(real1, imag1);
                    break;
                case OP_PHASE_SHIFT:
                    if (MATCH_BIT(qubit[localIdx0], controlMask))
                        statevec_phaseShift_func_reg(real0, imag0, op.operand.cosAngle, op.operand.sinAngle);
                    if (MATCH_BIT(qubit[localIdx1], controlMask))
                        statevec_phaseShift_func_reg(real1, imag1, op.operand.cosAngle, op.operand.sinAngle);
                    break;
                case OP_ROTATE_Z_FAKE:
                    if (MATCH_BIT(qubit[localIdx0], controlMask)) {
                        qreal cosAngle = op.operand.cosAngle;
                        qreal sinAngle = MATCH_BIT(qubit[localIdx0], op.targetMask) ? op.operand.sinAngle : -op.operand.sinAngle;
                        statevec_phaseShift_func_reg(real0, imag0, cosAngle, sinAngle);
                    }
                    if (MATCH_BIT(qubit[localIdx1], controlMask)) {
                        qreal cosAngle = op.operand.cosAngle;
                        qreal sinAngle = MATCH_BIT(qubit[localIdx1], op.targetMask) ? op.operand.sinAngle : -op.operand.sinAngle;
                        statevec_phaseShift_func_reg(real1, imag1, cosAngle, sinAngle);
                    }
                    break;
                case OP_ROTATE_Z:
                    if (MATCH_BIT(qubit[localIdx0], controlMask))
                        statevec_rotateZ_func_reg(real0, imag0, real1, imag1, op.operand.cosAngle, op.operand.sinAngle);
                    break;
                case OP_PAULI_X:
                    if (MATCH_BIT(qubit[localIdx0], controlMask))
                        statevec_pauliX_func_reg(real0, imag0, real1, imag1);
                    break;
                case OP_PAULI_Y:
                    if (MATCH_BIT(qubit[localIdx0], controlMask))
                        statevec_pauliY_func_reg(real0, imag0, real1, imag1);
                    break;
                case OP_PAULI_Y_CONJ:
                    if (MATCH_BIT(qubit[localIdx0], controlMask))
                        statevec_pauliYConj_func_reg(real0, imag0, real1, imag1);
                    break;
                case OP_HADAMARD:
                    if (MATCH_BIT(qubit[localIdx0], controlMask))
                        statevec_hadamard_func_reg(real0, imag0, real1, imag1);
                    break;
                case OP_ROTATE_X:
                    if (MATCH_BIT(qubit[localIdx0], controlMask))
                        statevec_rotateX_func_reg(real0, imag0, real1, imag1, op.operand.cosAngle, op.operand.sinAngle);
                    break;
                case OP_ROTATE_Y:
                    if (MATCH_BIT(qubit[localIdx0], controlMask))
                        statevec_rotateY_func_reg(real0, imag0, real1, imag1, op.operand.cosAngle, op.operand.sinAngle);
                    break;
                case OP_COMPACT_UNITARY:
                    if (MATCH_BIT(qubit[localIdx0], controlMask))
                        statevec_compactUnitary_func_reg(real0, imag0, real1, imag1, op.operand.alpha, op.operand.beta);
                    break;
                case OP_UNITARY:
                    if (MATCH_BIT(qubit[localIdx0], controlMask))
                        statevec_unitary_func_reg(real0, imag0, real1, imag1, op.operand.unitaryMatrix);
                    break;
                case OP_ONE:
                    break;
            }
        }
        stateVecShared[localIdx0] = val0;
        stateVecShared[localIdx1] = val1;
    }

    __syncthreads();
    stateVec[offset0] = stateVecShared[index0];
    stateVec[offset1] = stateVecShared[index1];
}

static void localizeTargetBits(FusedOp &fusedOperator, FILE *logFile) {
    for (int i = 0; i < fusedOperator.numOperators; i++) {
        if (logFile)
            fprintf(logFile, "%s %08x %08x -> ", getGateTypeString(fusedOperator.ops[i].type), fusedOperator.ops[i].targetMask, fusedOperator.ops[i].controlMask);
        if ((fusedOperator.targetMask & fusedOperator.ops[i].targetMask) == 0) {
            if ((int) fusedOperator.ops[i].type <= (int) OP_ROTATE_Z) {
                fusedOperator.ops[i].targetMaskLocal = 0;
            } else {
                LOG(FATAL) << fmt::format("{}: {:x} -> {:x}", getGateTypeString(fusedOperator.ops[i].type), fusedOperator.ops[i].controlMask, fusedOperator.ops[i].targetMask);
            }
        } else {
            u32 targetMask = fusedOperator.targetMask & ((fusedOperator.ops[i].targetMask << 1) - 1);
            fusedOperator.ops[i].targetMaskLocal = 1U << (countOneUint32(targetMask) - 1);
        }
        if (logFile)
            fprintf(logFile, "%s %08x\n", getGateTypeString(fusedOperator.ops[i].type), fusedOperator.ops[i].targetMaskLocal);
    }
}

static PartitionedFusedOperator partitionFusedOperator(const FusedOp &hostFusedOperator) {
    PartitionedFusedOperator pFusedOp;
    pFusedOp.controlMask = hostFusedOperator.controlMask;
    pFusedOp.targetMask = hostFusedOperator.targetMask;
    pFusedOp.initFunction = hostFusedOperator.initFunction;
    pFusedOp.numOperators = hostFusedOperator.numOperators;
    memcpy(&pFusedOp.memsel, &hostFusedOperator.memsel, sizeof(pFusedOp.memsel));
    memcpy(&pFusedOp.initParam, &hostFusedOperator.initParam, sizeof(pFusedOp.initParam));

#ifdef DEBUG
    FILE *f1, *f2;
    f1 = fopen("real.log", "a");
    f2 = fopen("fake.log", "a");
#endif

    pFusedOp.numGroups = 0;
    if (hostFusedOperator.numOperators > 0) {
        u32 prevTargetMaskLocal;

#ifdef DEBUG
        for (int i = 0; i < hostFusedOperator.numOperators; i++) {
            auto &op = hostFusedOperator.ops[i];
            fprintf(f1, "gate[%4d]                 %-20s %8x -> %8x(%8x)\n",
                    i, getOperatorNameString(op.type), op.controlMask, op.targetMaskLocal, op.targetMask);
        }
#endif

        pFusedOp.groups[0].size = 1;
        pFusedOp.numGroups = 1;
        prevTargetMaskLocal = pFusedOp.groups[0].targetMaskLocal = hostFusedOperator.ops[0].targetMaskLocal;

        if (hostFusedOperator.ops[0].targetMaskLocal == 0 && (int) hostFusedOperator.ops[0].type >= (int) OP_ROTATE_Z) {
            pFusedOp.pOps[0].type = OP_ROTATE_Z_FAKE;
        } else {
            pFusedOp.pOps[0].type = hostFusedOperator.ops[0].type;
        }
        pFusedOp.pOps[0].controlMask = hostFusedOperator.ops[0].controlMask;
        pFusedOp.pOps[0].targetMask = hostFusedOperator.ops[0].targetMask;
        pFusedOp.pOps[0].operand = hostFusedOperator.ops[0].operand;

        for (int i = 1; i < hostFusedOperator.numOperators; i++) {
            auto &op = hostFusedOperator.ops[i];

            if (op.targetMaskLocal == 0) {
                if ((int) op.type >= (int) OP_ROTATE_Z) {
                    pFusedOp.pOps[i].type = OP_ROTATE_Z_FAKE;
                } else {
                    pFusedOp.pOps[i].type = op.type;
                }
            } else {
                if ((op.targetMaskLocal | prevTargetMaskLocal) != op.targetMaskLocal) { // prevTargetMaskLocal not included in new op.targetMaskLocal
                    pFusedOp.groups[pFusedOp.numGroups].size = 0;
                    pFusedOp.numGroups++;
                }
                prevTargetMaskLocal = pFusedOp.groups[pFusedOp.numGroups - 1].targetMaskLocal = op.targetMaskLocal;
                pFusedOp.pOps[i].type = op.type;
            }
            pFusedOp.groups[pFusedOp.numGroups - 1].size++;
            pFusedOp.pOps[i].controlMask = op.controlMask;
            pFusedOp.pOps[i].targetMask = op.targetMask;
            pFusedOp.pOps[i].operand = op.operand;
        }

        for (int i = 0; i < pFusedOp.numGroups; i++) {
            if (pFusedOp.groups[i].targetMaskLocal == 0x0) {
                pFusedOp.groups[i].targetMaskLocal = 1024;
            }
        }

        for (int i = 0; i < pFusedOp.numOperators; i++) {
            auto &pOp = pFusedOp.pOps[i];
            switch (pOp.type) {
                case OP_S_GATE:
                case OP_PAULI_Z:
                case OP_S_GATE_CONJ:
                case OP_T_GATE:
                case OP_T_GATE_CONJ_NEG:
                case OP_T_GATE_NEG:
                case OP_T_GATE_CONJ:
                    break;

                case OP_ROTATE_Z:
                case OP_ROTATE_X:
                case OP_ROTATE_Y:
                case OP_PHASE_SHIFT:
                case OP_ROTATE_Z_FAKE:
                    pOp.operand.sinAngle = sin(hostFusedOperator.ops[i].operand.angle);
                    pOp.operand.cosAngle = cos(hostFusedOperator.ops[i].operand.angle);
                    break;

                case OP_PAULI_X:
                case OP_PAULI_Y:
                case OP_PAULI_Y_CONJ:
                case OP_HADAMARD:
                case OP_COMPACT_UNITARY:
                case OP_UNITARY:
                case OP_ONE:
                    break;
            }
        }

#ifdef DEBUG
        int ptr = 0;
        for (int i = 0; i < pFusedOp.numGroups; i++) {
            auto &group = pFusedOp.groups[i];
            for (int _ = 0; _ < group.size; _++) {
                auto &op = pFusedOp.pOps[ptr++];
                fprintf(f2, "group[%2d] gate[%4d] %-20s %8x -> %8x(%8x)\n", i, ptr, getOperatorNameString(op.type),
                        op.controlMask, group.targetMaskLocal, op.targetMask);
            }
        }
#endif
    }

#ifdef DEBUG
    fclose(f1);
    fclose(f2);
#endif

    return pFusedOp;
}

void statevec_fused(FusedOp &hostFusedOperator, int numQubits, qComplex *vec, FILE *logFile) {

    // 如果融合qubit数量不够11个，从低到高补齐
    int t = 1 << COALESCED_QUBITS;
    while (countOneUint32(hostFusedOperator.targetMask) < SHARED_MEMORY_QUBITS) {
        hostFusedOperator.targetMask |= t;
        t <<= 1;
    }

    for (int i = COALESCED_QUBITS, j = 0; i < numQubits && j < SHARED_MEMORY_QUBITS - COALESCED_QUBITS; i++) {
        if ((1U << i) & hostFusedOperator.targetMask) {
            hostFusedOperator.memsel[j++] = i;
            if (logFile)
                fprintf(logFile, "%d ", i);
        }
    }
    if (logFile)
        fprintf(logFile, "\n");

    localizeTargetBits(hostFusedOperator, logFile);
    auto pFusedOp = partitionFusedOperator(hostFusedOperator);
    //    LOG(INFO) << "Num gates: " << pFusedOp.numOperators;
    CUDA_CHECK(cudaMemcpyToSymbolAsync(devicePartitionedFusedOperator, &pFusedOp, sizeof devicePartitionedFusedOperator));
    statevec_fusedKernel<<<(1 << (numQubits - 11)), 1024>>>(vec, numQubits);
    CUDA_CHECK(cudaGetLastError());
}

void launchKernel(const FusedGate &fusedGate, int numQubitsLogical, int numQubitsKernel, int deviceId, qComplex *vec, bool mergeGates, FILE *logFile) {
    std::shared_ptr<DAG> fusedDag(new DAG(numQubitsLogical, mergeGates));
    for (const auto &gate : fusedGate.gates) {
        fusedDag->addGate(gate);
    }

    std::list<FusedGate> kernelGates;
    kernelGates.emplace_back();

    if (!fusedDag->empty()) {
        while (true) {
            fusedDag.reset(fusedDag->setSectionGates(kernelGates.back().gates, kernelGates.back().targetMask, 1, 666));
            if (fusedDag->empty()) {
                break;
            }
            kernelGates.emplace_back();
        }
    }

    auto &hostFusedOperator = hostFusedOperators[deviceId];
    hostFusedOperator.numOperators = 0;
    hostFusedOperator.controlMask = 0;
    hostFusedOperator.targetMask = fusedGate.targetMask;
    hostFusedOperator.initFunction = fusedGate.initFunction;
    memcpy(&hostFusedOperator.initParam, &fusedGate.initParam, sizeof(FusedOp::initParam));
    for (const auto &gatesPerQubit : kernelGates) {
        for (const auto &gate : gatesPerQubit.gates) {
            GateOp op{gate.operand, gate.type, (u32) gate.controlMask, (u32) gate.targetMask, 0u};
            hostFusedOperator.ops[hostFusedOperator.numOperators++] = op;
            hostFusedOperator.controlMask |= op.controlMask;
        }
    }
    statevec_fused(hostFusedOperator, numQubitsKernel, vec, logFile);
}

void launchKernelForSampling(const FusedGate &fusedGate, int numQubitsLogical, int numQubitsKernel, int deviceId, qComplex *vec) {
    auto &hostFusedOperator = hostFusedOperators[deviceId];
    hostFusedOperator.numOperators = 0;
    hostFusedOperator.controlMask = 0;
    hostFusedOperator.targetMask = fusedGate.targetMask;
    hostFusedOperator.initFunction = fusedGate.initFunction;
    memcpy(&hostFusedOperator.initParam, &fusedGate.initParam, sizeof(FusedOp::initParam));
    for (const auto &gate : fusedGate.gates) {
        GateOp op{gate.operand, gate.type, (u32) gate.controlMask, (u32) gate.targetMask, 0u};
        hostFusedOperator.ops[hostFusedOperator.numOperators++] = op;
        hostFusedOperator.controlMask |= op.controlMask;
    }
    statevec_fused(hostFusedOperator, numQubitsKernel, vec, nullptr);
}

void calculateSectionGpu(Qureg &qureg, int deviceId, int taskId) {
    auto *state = (QuregStateDistributed *) qureg.state;

    const auto &secLV = qureg.state->sectionGlobalViewList.front().localViews[taskId];

    for (const auto &fusedGate : secLV.fusedGates) {
        launchKernel(fusedGate, qureg.numQubitsInStateVec, state->numQubitsPerTask, deviceId, state->deviceStateVecs[deviceId], true, log_files[taskId]);
    }
}

__forceinline__ __device__ u64 insertBit_gpu(u64 x, u64 bit, u32 pos) {
    u64 low_half = x & ((1 << pos) - 1);
    u64 high_half = (x & (~((1 << pos) - 1))) << 1;
    return high_half | (bit << pos) | low_half;
}

__forceinline__ __device__ u64 selectBit_gpu(u64 x, u32 pos) {
    return (x >> pos) & 1;
}

__global__ void reorder_data_kernel(int bitNum, u32 votedMask, u32 victimMask, qComplex stateVec[]) {
    /**
     * bitNum:      the number of bits in local (without upper bits for taskId) bits
     * votedMask:   mask with bits voted to be new page number bits
     * victimMask:  mask with bits to be downgraded from page number bits
     */

    // initialize both index with unchanged bits (neither in votedMask nor in victimMask)
    u32 oldIdx = blockIdx.x * blockDim.x + threadIdx.x;
    u32 newIdx = blockIdx.x * blockDim.x + threadIdx.x;

    u32 oldBlockId = threadIdx.y;
    u32 newBlockId = threadIdx.z;

    // insert block id bits into LOWER bits (votedMask bits)
    u32 ptr = 0;
    for (int i = 0; i < bitNum; i++) {
        if ((votedMask >> i) & 1) {
            oldIdx = insertBit_gpu(oldIdx, selectBit_gpu(newBlockId, ptr), i);
            newIdx = insertBit_gpu(newIdx, selectBit_gpu(oldBlockId, ptr), i);
            ptr++;
        }
    }

    // insert block id bits into UPPER bits (victimMask bits)
    ptr = 0;
    for (int i = 0; i < bitNum; i++) {
        if ((victimMask >> i) & 1) {
            oldIdx = insertBit_gpu(oldIdx, selectBit_gpu(oldBlockId, ptr), i);
            newIdx = insertBit_gpu(newIdx, selectBit_gpu(newBlockId, ptr), i);
            ptr++;
        }
    }

    qComplex amp = stateVec[oldIdx];
    __syncthreads();
    stateVec[newIdx] = amp;
}

void reorderDataGpu(Qureg &qureg, int deviceId, int taskId) {
    auto *state = (QuregStateDistributed *) qureg.state;

    auto *sectionGlobalView = &qureg.state->sectionGlobalViewList.front();
    if (sectionGlobalView->victimMask == 0) {
        return;
    }
    qComplex *stateVec = state->deviceStateVecs[deviceId];
    dim3 dimThread;
    int blockIdBitNum = countOneUint32(sectionGlobalView->votedMask);
    int padding = (5 - 2 * blockIdBitNum > 0) ? (5 - 2 * blockIdBitNum) : 0;
    dimThread.x = 1u << padding;
    dimThread.y = 1u << blockIdBitNum;
    dimThread.z = 1u << blockIdBitNum;
    u32 dimBlock = 1u << (state->numQubitsPerTask - 2 * blockIdBitNum - padding);
    reorder_data_kernel<<<dimBlock, dimThread>>>(state->numQubitsPerTask, sectionGlobalView->votedMask, sectionGlobalView->victimMask, stateVec);
}

template<class T>
__inline__ __device__ T warpReduceAllSum(T val) {
    val += __shfl_xor_sync(0xFFFFFFFF, val, 16);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 8);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 4);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 2);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 1);
    return val;
}

template<class T>
__inline__ __device__ T blockReduceSum(T val) {
    static __shared__ T shared[32]; // Shared mem for 32 partial sums

    u32 lane = threadIdx.x & (32 - 1);
    u32 wid = threadIdx.x >> 5;

    val = warpReduceAllSum(val); // Each warp performs partial reduction
    if (lane == 0)
        shared[wid] = val; // Write reduced value to shared memory
    __syncthreads();       // Wait for all partial reductions

    //read from shared memory only if that warp existed
    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0;
    __syncthreads();
    if (wid == 0)
        val = warpReduceAllSum(val); //Final reduce within first warp

    return val;
}

template<class T>
__global__ void reduceGlobalKernel(T *in, T *out, int N) {
    T sum = 0;
    u32 arrayId = blockIdx.y;
    u32 globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    u32 globalStride = blockDim.x * gridDim.x;
    //reduce multiple elements per thread
    for (u32 i = globalIdx; i < N; i += globalStride) {
        sum += in[arrayId * N + i];
    }
    sum = blockReduceSum(sum);
    if (threadIdx.x == 0) {
        out[gridDim.x * arrayId + blockIdx.x] = sum;
    }
}

template<size_t N, size_t blockSize, class T>
__device__ T reduceAllShared(volatile T *sdata) {
    T sum = 0;
#pragma unroll 4
    //reduce multiple elements per thread
    for (u32 i = 0; i < N; i += blockSize) {
        sum += sdata[i + threadIdx.x];
    }

    u32 lane = threadIdx.x & (32 - 1); // tid % 32

    sum = warpReduceAllSum(sum); // Each warp performs partial reduction
    sdata[threadIdx.x] = sum;    // Write reduced value to shared memory
    __syncthreads();             // Wait for all partial reductions

    sum = (lane < (blockSize >> 5)) ? sdata[lane << 5 | lane] : 0; // Avoid bank conflict
    sum = warpReduceAllSum(sum);                                   //Final reduce
    __syncthreads();
    return sum;
}


template<class T>
void reducePingpong(T *data, T *buf, u32 N, u32 arrays) {
    if (N < 32 || N > (1 << 20)) {
        printf("Reduction Ping-pong out of range!\n");
    }
    u32 numReduceTo = std::min(1024u, N / 32);
    if (numReduceTo < 32) {
        reduceGlobalKernel<<<{1, arrays}, N>>>(data, data, N);
    } else {
        reduceGlobalKernel<<<{numReduceTo, arrays}, N / numReduceTo>>>(data, buf, N);
        reduceGlobalKernel<<<{1, arrays}, numReduceTo>>>(buf, data, numReduceTo);
    }
}


static const int numQubitsShared = 12;
static const size_t numAmpsShared = 1 << numQubitsShared;

__inline__ __device__ __host__ unsigned zeroBitInsert(unsigned x, unsigned targetBitMask) {
    return ((x & (-targetBitMask)) << 1) | (x & (targetBitMask - 1));
}

__inline__ __device__ __host__ unsigned oneBitInsert(unsigned x, unsigned targetBitMask) {
    return ((x & (-targetBitMask)) << 1) | (x & (targetBitMask - 1)) | targetBitMask;
}

__device__ void findZeroProbOfLowQubit(const qreal *stateVecProb, qreal *partialProb, int targetMask) {
#pragma unroll 4
    for (u32 offset = 0; offset < numAmpsShared / 2; offset += 1024) {
        u32 index = zeroBitInsert(threadIdx.x + offset, targetMask);
        partialProb[threadIdx.x + offset] = stateVecProb[index];
    }
    __syncthreads();
}

__device__ void findOneProbOfLowQubit(const qreal *stateVecProb, qreal *partialProb, int targetMask) {
#pragma unroll 4
    for (u32 offset = 0; offset < numAmpsShared / 2; offset += 1024) {
        u32 index = oneBitInsert(threadIdx.x + offset, targetMask);
        partialProb[threadIdx.x + offset] = stateVecProb[index];
    }
    __syncthreads();
}

__global__ void statevec_findProbabilityOfZeroKernel(const qComplex *vec, int numQubits, qreal *arrayOut) {
    __shared__ qreal stateVecProbShared[numAmpsShared];
    __shared__ qreal probPartialTempShared[numAmpsShared / 2];

#pragma unroll 4
    for (u32 offset = 0; offset < numAmpsShared; offset += 1024) {
        u32 globalIndex = blockIdx.x * numAmpsShared + threadIdx.x;
        qreal real = vec[globalIndex + offset].x;
        qreal imag = vec[globalIndex + offset].y;
        stateVecProbShared[threadIdx.x + offset] = imag * imag + real * real;
    }
    __syncthreads();

    // first calculate zero probability of qubit[numQubitsShared - 1]
    findZeroProbOfLowQubit(stateVecProbShared, probPartialTempShared, 1 << (numQubitsShared - 1));
    qreal zeroPartialProb = reduceAllShared<(numAmpsShared / 2), 1024>(probPartialTempShared);
    findOneProbOfLowQubit(stateVecProbShared, probPartialTempShared, 1 << (numQubitsShared - 1));
    qreal onePartialProb = reduceAllShared<(numAmpsShared / 2), 1024>(probPartialTempShared);
    qreal fullPartialProb = zeroPartialProb + onePartialProb;

    u32 i = threadIdx.x;
    if (i == numQubitsShared - 1) {
        arrayOut[gridDim.x * i + blockIdx.x] = zeroPartialProb;
        arrayOut[gridDim.x * (numQubits) + blockIdx.x] = fullPartialProb; // for multi GPUs
    }
    // High qubits, either full prob or 0
    if (numQubitsShared <= i && i < numQubits) {
        arrayOut[gridDim.x * i + blockIdx.x] = fullPartialProb * ((blockIdx.x >> (i - numQubitsShared)) & 1 ^ 1);
    }
    // Low qubits, half-half
#pragma unroll 12
    for (i = 0; i < numQubitsShared - 1; i++) {
        findZeroProbOfLowQubit(stateVecProbShared, probPartialTempShared, 1 << i);
        qreal prob = reduceAllShared<(numAmpsShared / 2), 1024>(probPartialTempShared);
        if (threadIdx.x == 0) {
            arrayOut[gridDim.x * i + blockIdx.x] = prob;
        }
    }
}

void statevec_calculateProbabilityOfZeroLocal(qreal *probs, qreal *buf, qComplex *stateVec, int numQubits) {
    if (numQubits < 12) {
        printf("%s:%d too few qubits to observe! The result will be wrong.", __func__, __LINE__);
        return;
    }

    u32 numAmps = 1 << numQubits;
    u32 numValuesToReduce = (u32) numAmps / numAmpsShared;

    const size_t threads = 1024;
    // double t1 = getWallTime();
    statevec_findProbabilityOfZeroKernel<<<numValuesToReduce, threads>>>(stateVec, numQubits, buf);
    CUDA_CHECK(cudaDeviceSynchronize());
    // double t2 = getWallTime();
    // printf("findProb stage 1: time elapsed %.6f\n", t2 - t1);

    if (numValuesToReduce < 32) {
        qreal probs_cpu[numQubits + 1][numValuesToReduce];
        CUDA_CHECK(cudaMemcpy(probs_cpu, buf, numValuesToReduce * (numQubits + 1) * sizeof(qreal), cudaMemcpyDefault));
        for (int qubit = 0; qubit < numQubits + 1; qubit++) {
            probs[qubit] = 0;
            for (int i = 0; i < numValuesToReduce; i++) {
                probs[qubit] += probs_cpu[qubit][i];
            }
        }
    } else {
        reducePingpong(buf, buf + numValuesToReduce * (numQubits + 1), numValuesToReduce, numQubits + 1);
        CUDA_CHECK(cudaMemcpy(probs, buf, (numQubits + 1) * sizeof(qreal), cudaMemcpyDefault));
    }
    // double  t3 = getWallTime();
    // printf("findProb stage 2 (reduction): time elapsed %.6f\n", t3 - t2);
}