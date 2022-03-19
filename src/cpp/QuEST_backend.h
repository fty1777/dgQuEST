// Distributed under MIT licence. See https://github.com/fty1777/dgQuEST/blob/main/LICENSE for details

#ifndef QUEST_BACKEND_H
#define QUEST_BACKEND_H

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <map>
#include <mutex>
#include <queue>
#include <set>
#include <stack>
#include <thread>
#include <vector>

#include "../QuEST_internal.h"
#include "QuEST.h"
#include "QuEST_precision.h"

#define SHARED_MEMORY_QUBITS 11
#define SHARED_MEMORY_LENGTH (1 << SHARED_MEMORY_QUBITS)
#define COALESCED_QUBITS     5

typedef unsigned int u32;
typedef unsigned long long int u64;
static constexpr double piOn4() {
    return std::atan(1);
}
static constexpr double pi() {
    return piOn4() * 4.0;
}
#define rsqrt2 0.7071067811865475244008443621
extern std::vector<FILE*> log_files;
#define SLEEP_TIME 5

#define CUDA_CHECK(cmd)                                        \
    do {                                                       \
        cudaError_t e = cmd;                                   \
        if (e != cudaSuccess) {                                \
            printf("Failed: Cuda error %s:%d '%s'\n",          \
                   __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    } while (0)

#define MPI_CHECK(cmd)                               \
    do {                                             \
        int e = cmd;                                 \
        if (e != MPI_SUCCESS) {                      \
            printf("Failed: MPI error %s:%d '%d'\n", \
                   __FILE__, __LINE__, e);           \
            exit(EXIT_FAILURE);                      \
        }                                            \
    } while (0)

/** Information about the environment the program is running in.
In practice, this holds info about MPI ranks and helps to hide MPI initialization code
*/
enum InitFunction {
    INIT_ALL_ZERO,
    INIT_CLASSICAL_STATE,
    INIT_PLUS_STATE,
    INIT_FROM_AMPS
};

inline static const char* getInitFunctionString(InitFunction func) {
    switch (func) {
        case INIT_ALL_ZERO:
            return "INIT_ALL_ZERO";
        case INIT_CLASSICAL_STATE:
            return "INIT_CLASSICAL_STATE";
        case INIT_PLUS_STATE:
            return "INIT_PLUS_STATE";
        case INIT_FROM_AMPS:
            return "INIT_FROM_AMPS";
        default:
            return nullptr;
    }
}

enum GateType {
    // definitive pseudo dependency (no target mask)
    // 8 phase st gates, DO NOT MODIFY the first 8 gates
    OP_ONE = 0,
    OP_T_GATE = 1,
    OP_S_GATE = 2,
    OP_T_GATE_CONJ_NEG = 3,
    OP_PAULI_Z = 4,
    OP_T_GATE_NEG = 5,
    OP_S_GATE_CONJ = 6,
    OP_T_GATE_CONJ = 7,

    OP_PHASE_SHIFT = 8,
    OP_ROTATE_Z_FAKE = 9, // KERNEL ONLY

    // dependency may exist
    OP_ROTATE_Z = 10,
    OP_PAULI_X = 11,
    OP_PAULI_Y = 12,
    OP_PAULI_Y_CONJ = 13,
    OP_HADAMARD = 14,
    OP_ROTATE_X = 15,
    OP_ROTATE_Y = 16,
    OP_COMPACT_UNITARY = 17,
    OP_UNITARY = 18
};

inline static const char* getGateTypeString(GateType name) {
    switch (name) {
        case OP_PAULI_Z:
            return "OP_PAULI_Z";
        case OP_S_GATE:
            return "OP_S_GATE";
        case OP_S_GATE_CONJ:
            return "OP_S_GATE_CONJ";
        case OP_T_GATE:
            return "OP_T_GATE";
        case OP_T_GATE_CONJ:
            return "OP_T_GATE_CONJ";
        case OP_PHASE_SHIFT:
            return "OP_PHASE_SHIFT";
        case OP_ROTATE_Z:
            return "OP_ROTATE_Z";
        case OP_PAULI_X:
            return "OP_PAULI_X";
        case OP_PAULI_Y:
            return "OP_PAULI_Y";
        case OP_PAULI_Y_CONJ:
            return "OP_PAULI_Y_CONJ";
        case OP_HADAMARD:
            return "OP_HADAMARD";
        case OP_ROTATE_X:
            return "OP_ROTATE_X";
        case OP_ROTATE_Y:
            return "OP_ROTATE_Y";
        case OP_COMPACT_UNITARY:
            return "OP_COMPACT_UNITARY";
        case OP_UNITARY:
            return "OP_UNITARY";
        case OP_ROTATE_Z_FAKE:
            return "OP_ROTATE_Z_FAKE";
        case OP_ONE:
            return "OP_ONE";
        case OP_T_GATE_CONJ_NEG:
            return "OP_T_GATE_CONJ_NEG";
        case OP_T_GATE_NEG:
            return "OP_T_GATE_NEG";
    }
    return "[INVALID OP]";
}

union GateOperand {
    qreal angle;
    struct {
        Complex alpha;
        Complex beta;
    }; // compactUnitary
    struct {
        qreal cosAngle;
        qreal sinAngle;
    }; // rotateXYZ
    ComplexMatrix2 unitaryMatrix;
};

struct Gate {
    GateOperand operand;
    GateType type;
    u64 targetMask;
    u64 controlMask;
};

struct QubitMapping {
    // Logical: Qubit sequence in real world
    // Virtual: Qubit sequence in address bits correspondence
    std::vector<int> l2v{}; // logical2virtual
    std::vector<int> v2l{}; // virtual2logical
};

struct FusedGate {
    std::vector<Gate> gates{};
    InitFunction initFunction = INIT_ALL_ZERO;
    union {
        u64 stateIndex;
        qreal probFactor;
    } initParam;

    u64 targetMask = 0;
};

struct SectionLocalView {
    std::vector<FusedGate> fusedGates{};
    std::vector<Gate> gates{};
    InitFunction initFunction = INIT_ALL_ZERO;
    union {
        u64 stateIndex;
        qreal probFactor;
    } initParam;
    u64 targetMask = 0;
};

struct SectionGlobalView {
    explicit SectionGlobalView(int sectionIndex);

    std::map<int, SectionLocalView> localViews{};
    std::vector<Gate> gates{};
    int sectionIndex;
    InitFunction initFunction = INIT_ALL_ZERO;
    union {
        u64 stateIndex;
        qreal probFactor;
    } initParam;

    u64 targetMask = 0;

    QubitMapping mappingCalc;
    QubitMapping mappingComm;
    u64 votedMask = 0;
    u64 victimMask = 0;
};

struct GateOp {
    GateOperand operand;
    GateType type;
    u32 controlMask;
    u32 targetMask;
    u32 targetMaskLocal;
};

struct FusedOp {
    GateOp ops[666];
    int numOperators;
    InitFunction initFunction;
    union {
        u32 stateIndex;
        qreal probFactor;
    } initParam;
    int memsel[SHARED_MEMORY_QUBITS - COALESCED_QUBITS];
    u32 controlMask;
    u32 targetMask;
};

struct PartialGateOp {
    GateOperand operand;
    GateType type;
    u32 controlMask;
    u32 targetMask;
};

struct GateOpGroup {
    u32 targetMaskLocal;
    u32 size;
};

struct PartitionedFusedOperator {
    PartialGateOp pOps[666];
    GateOpGroup groups[333];
    int numOperators;
    int numGroups;
    InitFunction initFunction;
    union {
        u32 stateIndex;
        qreal probFactor;
    } initParam;
    int memsel[SHARED_MEMORY_QUBITS - COALESCED_QUBITS];
    u32 controlMask;
    u32 targetMask;
};


class DAG {
public:
    struct Node {
        explicit Node(const Gate& gate);

        Gate gate;
        std::vector<Node*> prevNodes;
        std::vector<Node*> rearNodes;
        u64 targetMask = 0;
        u64 barrierMask = 0;
        size_t remainPrevNum = 0;
        bool visited = false;
        bool selected = false;
    };

public:
    explicit DAG(int numQubits, bool mergeGates = true);
    ~DAG();

    void addGate(const Gate& gate);
    DAG* setSectionGates(std::vector<Gate>& gates, u64& targetMask, int numQubitsPerTask, int maxGateNum = 100000000);
    bool empty() const { return allNodes.empty(); }
    size_t size() const { return allNodes.size(); }

private:
    void dfs(Node* node, u64& targetMask, int numQubitsPerTask);
    u32 gate_count = 0;

    bool mergeGates;

private:
    std::list<Node> allNodes;
    std::vector<std::stack<Node*>> barrierNodeStack;
    std::vector<Node*> frontNodes;
};

class QuregState {
public:
    explicit QuregState(Qureg& qureg);
    virtual ~QuregState() noexcept;
    void statevec_initClassicalState(long long int stateInd);
    void densmatr_initClassicalState(long long int stateInd);
    void statevec_initPlusState();
    void densmatr_initPlusState();
    virtual void statevec_initStateFromAmps(qreal* reals, qreal* imags) = 0;
    void addGate(const Gate& gate);
    void splitCircuit();
    void applyMapping();
    virtual void switchToNextState();
    void prepareMappings();
    void splitSectionsIntoFusedGates();

public:
    std::deque<SectionGlobalView> sectionGlobalViewList;

    QubitMapping mapping[3];
    int sectionIndex = 0;

    bool probsDirty = true;
    qreal* probsOfZero = nullptr;

    qComplex* deviceStateVecs[4];

public:
    Qureg qureg;
    std::vector<Gate> gates{};
    int numGatesTotal = 0;
    int numGatesCalculated = 0;
    void translateSections();
};

class QuregStateDistributed : public QuregState {
public:
    explicit QuregStateDistributed(Qureg& qureg);
    void statevec_initStateFromAmps(qreal* reals, qreal* imags) override;
    void switchToNextState() override;

    u64 numQubitsPerTask;
    u64 numAmpsPerTask;
    u64 numTasksTotal; // =numPagesPerTask
    u64 numPagesPerTask;
    u64 numQubitsPerPage;
    u64 numAmpsPerPage;
    u64 numTasksThisNode;
    u64 taskBeginOffset;

    int numTasksTable[4];
    int taskBeginOffsetTable[4];
    int numTaskIdQubits;

    std::vector<qreal*> probsOfZero;

    std::vector<qComplex*> pageTable[2]{};

    u64 memPoolNumPagesTotal{0};
    std::atomic<size_t> memPoolAvailHead{0};
    std::atomic<size_t> memPoolAvailTail{0};
    std::vector<qComplex*> memPoolAvailQueue{};
    std::vector<qComplex*> memPoolAllocatedEnvPage{};


    qComplex* mallocPage();
    qComplex* mallocPageBusyWaiting();
    void freePage(qComplex* ptr);
};

class QuESTEnvValue {
public:
    virtual ~QuESTEnvValue() = default;
    virtual void statevec_createQureg(Qureg& qureg, int numQubits) = 0;
    virtual void statevec_destroyQureg(Qureg& qureg) = 0;
    virtual void calculateAllSections(Qureg& qureg) = 0;
    virtual Complex statevec_getAmp(Qureg& qureg, long long int index) = 0;
    virtual qreal statevec_calcProbOfOutcome(Qureg& qureg, int measureQubit, int outcome) = 0;
    virtual void searchTaskPartition(Qureg& qureg) = 0;
    qComplex* mallocEnvPage(u64 pageSizeInAmps);
    void freeEnvPage(u64 pageSizeInAmps, qComplex* ptr);

public:
    int rankId = 0;
    int numRanks = 1;
    int numDevices = 1;

    u64 envMemPoolPagesNumTotal = 0;
    std::set<std::tuple<qComplex*, qComplex*, u64>> envMemPoolBlocks{};
    std::map<u64, std::queue<qComplex*>> envMemPoolAvailQueue{};
    std::mutex envMemPoolAvailQueueMutex;
    std::condition_variable envMemPoolAvailQueueCv;
};

class QuESTEnvValueDistributed : public QuESTEnvValue {
public:
    QuESTEnvValueDistributed(int numDevices, int minK, int maxK);
    QuESTEnvValueDistributed(int numDevices, int minK, int maxK, int designatedK);
    ~QuESTEnvValueDistributed() override;
    void statevec_createQureg(Qureg& qureg, int numQubits) override;
    void statevec_destroyQureg(Qureg& qureg) override;
    Complex statevec_getAmp(Qureg& qureg, long long int index) override;
    void calculateAllSections(Qureg& qureg) override;
    qreal statevec_calcProbOfOutcome(Qureg& qureg, int measureQubit, int outcome) override;
    void searchTaskPartition(Qureg& qureg) override;
    void applyK(Qureg& qureg, int numTaskIdQubits);

    int minK = 0;
    int maxK = 0;
    int designatedK = 0;
    bool searchK = true;

protected:
    qreal* deviceBuffer[4]{};

private:
    void growMemPool(u64 numPages, u64 numAmpsPerPage);
};

void launchKernel(const FusedGate& fusedGate, int numQubitsLogical, int numQubitsKernel, int deviceId, qComplex* vec, bool mergeGates = true, FILE* logFile = nullptr);
void launchKernelForSampling(const FusedGate& fusedGate, int numQubitsLogical, int numQubitsKernel, int deviceId, qComplex* vec);
void calculateSectionGpu(Qureg& qureg, int deviceId, int taskId);
void reorderDataGpu(Qureg& qureg, int deviceId, int taskId);

inline static u64 selectBit(u64 x, int pos) {
    return (x >> pos) & 1;
}

inline static u64 translateIndex(u64 index, const std::vector<int>& mapping, int bitNum) {
    u64 translated = 0;
    for (int i = 0; i < bitNum; i++) {
        translated |= selectBit(index, i) << mapping[i];
    }
    return translated;
}

inline static int countOneUint32(u32 n) {
    const u32 MASK1 = 0x55555555ULL;
    const u32 MASK2 = 0x33333333ULL;
    const u32 MASK4 = 0x0F0F0F0FULL;
    const u32 MASK8 = 0x00FF00FFULL;
    const u32 MASK16 = 0x0000FFFFULL;
    n = ((n >> 1) & MASK1) + (n & MASK1);
    n = ((n >> 2) & MASK2) + (n & MASK2);
    n = ((n >> 4) & MASK4) + (n & MASK4);
    n = ((n >> 8) & MASK8) + (n & MASK8);
    n = (n >> 16) + (n & MASK16);
    return (int) n;
}

inline static int countOneUint64(u64 n) {
    const u64 MASK1 = 0x5555555555555555ULL;
    const u64 MASK2 = 0x3333333333333333ULL;
    const u64 MASK4 = 0x0F0F0F0F0F0F0F0FULL;
    const u64 MASK8 = 0x00FF00FF00FF00FFULL;
    const u64 MASK16 = 0x0000FFFF0000FFFFULL;
    n = ((n >> 1) & MASK1) + (n & MASK1);
    n = ((n >> 2) & MASK2) + (n & MASK2);
    n = ((n >> 4) & MASK4) + (n & MASK4);
    n = ((n >> 8) & MASK8) + (n & MASK8);
    n = ((n >> 16) & MASK16) + (n & MASK16);
    return (int) (n >> 32) + (int) n;
}

void statevec_calculateProbabilityOfZeroLocal(qreal* probs, qreal* buf, qComplex* stateVec, int numQubits);

static ComplexMatrix2 getUnitaryMatrix(const GateOperand& op, GateType operatorName) {
    ComplexMatrix2 mat{};
    switch (operatorName) {
        case OP_ROTATE_Z:
            mat.r0c0 = {cos(op.angle), -sin(op.angle)};
            mat.r1c1 = {cos(op.angle), sin(op.angle)};
            break;
        case OP_PAULI_X:
            mat.r0c1.real = 1;
            mat.r1c0.real = 1;
            break;
        case OP_PAULI_Y:
            mat.r0c1.imag = -1;
            mat.r1c0.imag = +1;
            break;
        case OP_PAULI_Y_CONJ:
            mat.r0c1.imag = +1;
            mat.r1c0.imag = -1;
            break;
        case OP_HADAMARD:
            mat.r0c0.real = +rsqrt2;
            mat.r0c1.real = +rsqrt2;
            mat.r1c0.real = +rsqrt2;
            mat.r1c1.real = -rsqrt2;
            break;
        case OP_ROTATE_X:
            mat.r0c0.real = cos(op.angle);
            mat.r0c1.imag = -sin(op.angle);
            mat.r1c0.imag = -sin(op.angle);
            mat.r1c1.real = cos(op.angle);
            break;
        case OP_ROTATE_Y:
            mat.r0c0.real = cos(op.angle);
            mat.r0c1.real = -sin(op.angle);
            mat.r1c0.real = sin(op.angle);
            mat.r1c1.real = cos(op.angle);
            break;
        case OP_COMPACT_UNITARY:
            mat.r0c0 = op.alpha;
            mat.r0c1 = {-op.beta.real, op.beta.imag};
            mat.r1c0 = op.beta;
            mat.r1c1 = {op.alpha.real, -op.alpha.imag};
            break;
        case OP_UNITARY:
            mat = op.unitaryMatrix;
            break;
        default:
            printf("ERROR! NOT ABLE TO CONVERT TO UNITARY MATRIX! %s:%d\n", __func__, __LINE__);
    }
    return mat;
}

template<class secGVListType>
void deriveMappings(secGVListType& sectionGlobalViewList, int numQubitsInStateVec, int numQubitsPerTask, int numQubitsPerPage) {
    const u64 numAmpsPerPage = 1ULL << numQubitsPerPage;
    const u64 numAmpsPerTask = 1ULL << numQubitsPerTask;

    if (sectionGlobalViewList[0].initFunction != INIT_FROM_AMPS) {
        // When initializing from given state (read from file), the mapping is also fixed. Do not remap this section.
        auto* currSecGV = &sectionGlobalViewList[0];
        currSecGV->mappingCalc.v2l.resize(numQubitsInStateVec);
        currSecGV->mappingCalc.l2v.resize(numQubitsInStateVec);
        int n = numQubitsInStateVec - 1;
        for (int i = numQubitsInStateVec - 1; i >= 0; i--) {
            if (!selectBit(currSecGV->targetMask, i)) {
                currSecGV->mappingCalc.v2l[n] = i;
                currSecGV->mappingCalc.l2v[i] = n--;
            }
        }
        for (int i = numQubitsInStateVec - 1; i >= 0; i--) {
            if (selectBit(currSecGV->targetMask, i)) {
                currSecGV->mappingCalc.v2l[n] = i;
                currSecGV->mappingCalc.l2v[i] = n--;
            }
        }
    }

    for (auto idx = 0; idx < sectionGlobalViewList.size() - 1; idx++) {
        auto* currSecGV = &sectionGlobalViewList[idx];
        currSecGV->mappingComm = currSecGV->mappingCalc;
        int l1 = numQubitsInStateVec - 1;
        int l0, v1, v0;
        u64 votedMask = 0;
        int numVoted = 0;
        auto* nextSecGV = &sectionGlobalViewList[idx + 1];
        for (int i = 0; i < numQubitsInStateVec - numQubitsPerTask; i++) {
            while (selectBit(nextSecGV->targetMask, l1)) {
                l1--;
            }
            v1 = currSecGV->mappingCalc.l2v[l1];
            if (v1 < numQubitsPerTask) {
                votedMask |= 1ULL << v1;
                numVoted++;
            }
            l1--;
        }
        u64 victimMask = numAmpsPerTask - numAmpsPerPage;
        u64 voted_xor_victim = votedMask ^ victimMask;
        votedMask &= voted_xor_victim;
        victimMask &= voted_xor_victim;
        v0 = numQubitsPerPage;
        for (int i = numVoted; i < numQubitsInStateVec - numQubitsPerTask; i++) {
            while (!selectBit(victimMask, v0)) {
                v0++;
            }
            victimMask &= ~(1ULL << v0);
        }
        currSecGV->votedMask = votedMask;
        currSecGV->victimMask = victimMask;
        v1 = numQubitsPerPage - 1;
        v0 = numQubitsInStateVec - 1;
        int num_to_swap = countOneUint64(votedMask);
        for (int i = 0; i < num_to_swap; i++) {
            while (!selectBit(votedMask, v1)) {
                v1--;
            }
            while (!selectBit(victimMask, v0)) {
                v0--;
            }
            l1 = currSecGV->mappingComm.v2l[v1];
            l0 = currSecGV->mappingComm.v2l[v0];
            currSecGV->mappingComm.l2v[l0] = v1;
            currSecGV->mappingComm.l2v[l1] = v0;
            currSecGV->mappingComm.v2l[v0] = l1;
            currSecGV->mappingComm.v2l[v1] = l0;
            v1--;
            v0--;
        }

        // Derive next sec's mappingCalc from 1) curr sec's mappingComm and 2) next sec's targetBits（卡间交换映射），寻找最少交互方式（高位尽量不改变）
        nextSecGV->mappingCalc = currSecGV->mappingComm;
        int vH = numQubitsInStateVec - 1;
        int vL = numQubitsPerTask - 1;
        auto targetMask = nextSecGV->targetMask;
        while (true) {
            while (vH >= numQubitsPerTask && !selectBit(targetMask, nextSecGV->mappingCalc.v2l[vH])) {
                vH--;
            }
            while (vL >= numQubitsPerPage && selectBit(targetMask, nextSecGV->mappingCalc.v2l[vL])) {
                vL--;
            }
            if (vH < numQubitsPerTask || vL < numQubitsPerPage) {
                break;
            }
            int logicalH = nextSecGV->mappingCalc.v2l[vH];
            int logicalL = nextSecGV->mappingCalc.v2l[vL];
            nextSecGV->mappingCalc.l2v[logicalH] = vL;
            nextSecGV->mappingCalc.l2v[logicalL] = vH;
            nextSecGV->mappingCalc.v2l[vH] = logicalL;
            nextSecGV->mappingCalc.v2l[vL] = logicalH;
            vH--;
            vL--;
        }
    }

    sectionGlobalViewList.back().mappingComm = sectionGlobalViewList.back().mappingCalc;
}

void translateSection(SectionGlobalView* secGV, int taskId, int numQubitsInStateVec, int numQubitsPerTask);
void splitSectionLocalViewIntoFusedGates(SectionLocalView* secLV, int numQubitsInStateVec);

#endif // QUEST_BACKEND_H
