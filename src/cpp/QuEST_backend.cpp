// Distributed under MIT licence. See https://github.com/fty1777/dgQuEST/blob/main/LICENSE for details

#include "QuEST_backend.h"

#include <easylogging++.h>
#include <fmt/core.h>
#include <fmt/format.h>

std::vector<FILE*> log_files;

QuregState::QuregState(Qureg& qureg)
    : qureg(qureg) {
    for (int i = 0; i < 3; i++) {
        mapping[i].l2v.resize(qureg.numQubitsInStateVec);
        mapping[i].v2l.resize(qureg.numQubitsInStateVec);
    }
    probsOfZero = new qreal[qureg.numQubitsInStateVec + 1];
}
QuregState::~QuregState() noexcept = default;

void QuregState::statevec_initClassicalState(long long int stateInd) {
    sectionGlobalViewList.clear();
    sectionGlobalViewList.emplace_back(sectionIndex++);
    sectionGlobalViewList.back().initFunction = INIT_CLASSICAL_STATE;
    sectionGlobalViewList.back().initParam.stateIndex = stateInd;
}

void QuregState::densmatr_initClassicalState(long long int stateInd) {
    sectionGlobalViewList.clear();
    sectionGlobalViewList.emplace_back(sectionIndex++);
    sectionGlobalViewList.back().initFunction = INIT_CLASSICAL_STATE;
    sectionGlobalViewList.back().initParam.stateIndex = stateInd << qureg.numQubitsRepresented | stateInd;
}

void QuregState::statevec_initPlusState() {
    sectionGlobalViewList.clear();
    sectionGlobalViewList.emplace_back(sectionIndex++);
    sectionGlobalViewList.back().initFunction = INIT_PLUS_STATE;
    sectionGlobalViewList.back().initParam.probFactor = 1.0 / sqrt((qreal) (1LL << qureg.numQubitsRepresented));
}

void QuregState::densmatr_initPlusState() {
    sectionGlobalViewList.clear();
    sectionGlobalViewList.emplace_back(sectionIndex++);
    sectionGlobalViewList.back().initFunction = INIT_PLUS_STATE;
    sectionGlobalViewList.back().initParam.probFactor = 1.0 / (qreal) (1LL << qureg.numQubitsRepresented);
}

void QuregState::addGate(const Gate& gate) {
    gates.push_back(gate);
}

void QuregState::prepareMappings() {
    auto* state = (QuregStateDistributed*) qureg.state;
    deriveMappings(sectionGlobalViewList, qureg.numQubitsInStateVec, state->numQubitsPerTask, state->numQubitsPerPage);
}

void translateSection(SectionGlobalView* secGV, int taskId, int numQubitsInStateVec, int numQubitsPerTask) {
    auto& secLV = secGV->localViews[taskId];
    // set init type
    secLV.initFunction = secGV->initFunction;
    u64 taskMask = taskId << numQubitsPerTask;
    u64 gpuMask = (1ULL << numQubitsPerTask) - 1;
    switch (secLV.initFunction) {
        case INIT_ALL_ZERO:
            break;
        case INIT_CLASSICAL_STATE:
            secLV.initParam.stateIndex = translateIndex(secGV->initParam.stateIndex, secGV->mappingCalc.l2v, numQubitsInStateVec);
            if ((secLV.initParam.stateIndex & taskMask) != taskMask) {
                secLV.initFunction = INIT_ALL_ZERO;
            }
            break;
        case INIT_PLUS_STATE:
            secLV.initParam.probFactor = secGV->initParam.probFactor;
            break;
        case INIT_FROM_AMPS:
            break;
    }
//    fprintf(log_files[taskId], "%s -> %s\n", getInitFunctionString(secGV->initFunction), getInitFunctionString(secLV.initFunction));

    // gates
    for (auto gate : secGV->gates) {
//        fprintf(log_files[taskId], "%s %09llx %09llx -> ", getGateTypeString(gate.type), gate.targetMask, gate.controlMask);
        gate.targetMask = translateIndex(gate.targetMask, secGV->mappingCalc.l2v, numQubitsInStateVec);
        gate.controlMask = translateIndex(gate.controlMask, secGV->mappingCalc.l2v, numQubitsInStateVec);
        if ((gate.controlMask & (taskMask | gpuMask)) == gate.controlMask) {
            gate.controlMask &= gpuMask;
            if ((gate.targetMask & gpuMask) != gate.targetMask) {
                switch (gate.type) {
                    case OP_ROTATE_Z:
                        gate.type = OP_PHASE_SHIFT;
                        if ((gate.targetMask & taskMask) != gate.targetMask) {
                            gate.operand.angle = -gate.operand.angle;
                        }
                        break;
                    default:
                        LOG(FATAL) << fmt::format("{}: {:x} -> {:x}", getGateTypeString(gate.type), gate.controlMask, gate.targetMask);
                }
                gate.targetMask = 0;
            }
            secLV.gates.push_back(gate);
//            fprintf(log_files[taskId], "%s %08llx %08llx\n", getGateTypeString(gate.type), gate.targetMask, gate.controlMask);
        } else {
//            fprintf(log_files[taskId], "NULL\n");
        }
    }
}

void QuregState::translateSections() {
    auto* state = (QuregStateDistributed*) qureg.state;
    for (int gVIdx = 0; gVIdx < sectionGlobalViewList.size(); gVIdx++) {
        for (int taskId = state->taskBeginOffset, end = state->taskBeginOffset + state->numTasksThisNode; taskId < end; taskId++) {
            ::translateSection(&sectionGlobalViewList[gVIdx], taskId, qureg.numQubitsInStateVec, state->numQubitsPerTask);
        }
    }
}

void splitSectionLocalViewIntoFusedGates(SectionLocalView* secLV, int numQubitsInStateVec) {
    auto& fusedGates = secLV->fusedGates;
    fusedGates.emplace_back();
    fusedGates.back().initFunction = secLV->initFunction;
    fusedGates.back().targetMask = (1 << COALESCED_QUBITS) - 1;
    memcpy(&fusedGates.back().initParam, &secLV->initParam, sizeof(SectionGlobalView::initParam));

    std::shared_ptr<DAG> localDag(new DAG(numQubitsInStateVec));
    for (const auto& gate : secLV->gates) {
        localDag->addGate(gate);
    }
    if (!localDag->empty()) {
        while (true) {
            localDag.reset(localDag->setSectionGates(fusedGates.back().gates, fusedGates.back().targetMask, SHARED_MEMORY_QUBITS, 666));

            if (localDag->empty()) {
                break;
            }
            fusedGates.emplace_back();
            fusedGates.back().initFunction = INIT_FROM_AMPS;
            fusedGates.back().targetMask = (1 << COALESCED_QUBITS) - 1;
        }
    }
}

void QuregState::splitSectionsIntoFusedGates() {
    for (auto & secGV : sectionGlobalViewList) {
        for (auto& it : secGV.localViews) {
            auto& secLV = it.second;
            ::splitSectionLocalViewIntoFusedGates(&secLV, qureg.numQubitsInStateVec);
        }
    }
}

void QuregState::splitCircuit() {
    auto* state = dynamic_cast<QuregStateDistributed*>(qureg.state);
    if (gates.empty()) {
        return;
    }

    auto globalDag = new DAG(qureg.numQubitsInStateVec);
    for (const auto& gate : gates) {
        globalDag->addGate(gate);
    }
    gates.clear();

    if (sectionGlobalViewList.empty()) {
        sectionGlobalViewList.emplace_back(sectionIndex++);
        sectionGlobalViewList.back().initFunction = INIT_FROM_AMPS;
    }
    while (true) {
        auto next_dag = globalDag->setSectionGates(sectionGlobalViewList.back().gates, sectionGlobalViewList.back().targetMask, state->numQubitsPerTask);
        numGatesTotal += sectionGlobalViewList.back().gates.size();
        delete globalDag;
        globalDag = next_dag;
        if (globalDag->empty()) {
            prepareMappings();
            translateSections();
            splitSectionsIntoFusedGates();
            return;
        }
        sectionGlobalViewList.emplace_back(sectionIndex++);
        sectionGlobalViewList.back().initFunction = INIT_FROM_AMPS;
    }
}

void QuregState::applyMapping() {
    auto* currSecGV = &sectionGlobalViewList.front();
    if (qureg.env.value->rankId == 0) {
        auto it = ++sectionGlobalViewList.begin();
        printf("section %d\n", currSecGV->sectionIndex);
        printf("(current) %llx ===> %llx (next)    initFunction: %s    gates: %lu (%d+%lu/%d)\n",
               currSecGV->targetMask, (sectionGlobalViewList.size() <= 1) ? 0 : it->targetMask,
               getInitFunctionString(currSecGV->initFunction), currSecGV->gates.size(),
               numGatesCalculated, currSecGV->gates.size(), numGatesTotal);
        numGatesCalculated += currSecGV->gates.size();
    }

    mapping[0] = currSecGV->mappingCalc;
    mapping[1] = currSecGV->mappingComm;
    if (qureg.env.value->rankId == 0) {
        printf("voted %llx, victim %llx\n", currSecGV->votedMask, currSecGV->victimMask);
    }
    if (sectionGlobalViewList.size() > 1) {
        auto* nextSecGV = &sectionGlobalViewList[1];
        mapping[2] = nextSecGV->mappingCalc;
    } else {
        mapping[2] = mapping[1];
    }

    if (qureg.env.value->rankId == 0) {
        for (int n = 0; n < 3; n++) {
            for (int i = qureg.numQubitsInStateVec - 1; i >= 0; i--) {
                printf("%d ", mapping[n].v2l[i]);
            }
            printf(", ");
            for (int i = qureg.numQubitsInStateVec - 1; i >= 0; i--) {
                printf("%d ", mapping[n].l2v[i]);
            }
            printf("\n");
        }
    }
}

void QuregState::switchToNextState() {
    sectionGlobalViewList.pop_front();
    std::swap(mapping[0], mapping[2]);
}

void QuESTEnvValue::freeEnvPage(u64 pageSizeInAmps, qComplex* ptr) {
    std::unique_lock<std::mutex> lock(envMemPoolAvailQueueMutex);
    envMemPoolAvailQueue[pageSizeInAmps].push(ptr);
    envMemPoolAvailQueueCv.notify_one();
}

qComplex* QuESTEnvValue::mallocEnvPage(u64 pageSizeInAmps) {
    std::unique_lock<std::mutex> lock(envMemPoolAvailQueueMutex);
    auto& q = envMemPoolAvailQueue[pageSizeInAmps];
    while (q.empty()) {
        envMemPoolAvailQueueCv.wait(lock);
    }
    qComplex* ptr = q.front();
    q.pop();
    return ptr;
}

SectionGlobalView::SectionGlobalView(int sectionIndex)
    : sectionIndex{sectionIndex} {}