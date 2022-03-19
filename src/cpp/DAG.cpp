// Distributed under MIT licence. See https://github.com/fty1777/dgQuEST/blob/main/LICENSE for details

#undef DEBUG
#undef CHECK_UNITARY

#include <algorithm>

#include "QuEST_backend.h"

DAG::DAG(int numQubits, bool mergeGates)
    : mergeGates(mergeGates) {
    barrierNodeStack.resize(numQubits);
}

DAG::~DAG() = default;

DAG::Node::Node(const Gate &gate)
    : gate(gate) {
    barrierMask = gate.targetMask | gate.controlMask;
}

void DAG::addGate(const Gate &gate) {
#ifdef DEBUG
    gate_count++;
#endif
    allNodes.emplace_back(gate);
    auto &node = allNodes.back();

    auto reduceGate = [&](const int qubit) -> bool {
        auto *topNode = barrierNodeStack[qubit].top();

        const auto deleteGate = [&] {
#ifdef DEBUG
            printf("deleted gate [%20s] and [%20s] %8llx => %8llx (%d)\n",
                   getOperatorNameString(node.gate.type), getOperatorNameString(topNode->gate.type),
                   node.gate.controlMask, node.gate.targetMask, gate_count);
#endif
            for (auto &stack : barrierNodeStack) {
                if (!stack.empty() && stack.top() == topNode) {
                    stack.pop();
                }
            }
            if (topNode->prevNodes.empty()) {
                frontNodes.erase(std::remove_if(frontNodes.begin(), frontNodes.end(),
                                                [topNode](Node *p) { return p == topNode; }),
                                 frontNodes.end());
            } else {
                for (const auto &prevNode : topNode->prevNodes) {
                    prevNode->rearNodes.erase(std::remove_if(prevNode->rearNodes.begin(), prevNode->rearNodes.end(),
                                                             [topNode](Node *p) { return p == topNode; }),
                                              prevNode->rearNodes.end());
                }
            }
            allNodes.remove_if([topNode](const Node &n) { return &n == topNode; });
            return true;
        };
        const auto overrideGateNoParam = [&node, topNode, qubit](const GateType op) {
#ifdef DEBUG
            printf("merges gate [%s] and [%s] into a [%s:%d] gate\n", getOperatorNameString(node.gate.type),
                   getOperatorNameString(topNode->gate.type), getOperatorNameString(op), qubit);
#endif
            topNode->gate.type = op;
            return true;
        };
        const auto overrideGate = [&node, topNode](const Gate &gate) {
#ifdef DEBUG
            printf("merges gate [%s] and [%s] into a [%s] gate\n", getOperatorNameString(node.gate.type),
                   getOperatorNameString(topNode->gate.type), getOperatorNameString(gate.type));
#endif
            topNode->gate = gate;
            return true;
        };

        const auto getMergedPhaseShift = [](const Gate &anotherGate, const Gate &psGate) -> Gate {
            Gate mergedPSGate = psGate;
            if (anotherGate.type == OP_PHASE_SHIFT) {
                mergedPSGate.operand.angle = psGate.operand.angle + anotherGate.operand.angle;
            } else {
                mergedPSGate.operand.angle = psGate.operand.angle + piOn4() * (int) anotherGate.type;
            }
            return mergedPSGate;
        };
        const auto getMergedUnitary = [](const Gate &anotherGate, const Gate &uGate,
                                         bool uGateOnLeft) -> Gate {
            const auto multUnitaryMatrix = [](ComplexMatrix2 &C, const ComplexMatrix2 &A, const ComplexMatrix2 &B) {
                const auto multComplex = [](Complex x, Complex y) -> Complex {
                    return {x.real * y.real - x.imag * y.imag, x.real * y.imag + x.imag * y.real};
                };
                const auto addComplex = [](Complex x, Complex y) -> Complex {
                    return {x.real + y.real, x.imag + y.imag};
                };
                C.r0c0 = addComplex(multComplex(A.r0c0, B.r0c0), multComplex(A.r0c1, B.r1c0));
                C.r0c1 = addComplex(multComplex(A.r0c0, B.r0c1), multComplex(A.r0c1, B.r1c1));
                C.r1c0 = addComplex(multComplex(A.r1c0, B.r0c0), multComplex(A.r1c1, B.r1c0));
                C.r1c1 = addComplex(multComplex(A.r1c0, B.r0c1), multComplex(A.r1c1, B.r1c1));
            };
            const auto isMatrixUnitary = [](ComplexMatrix2 u) {
                if (absReal(u.r0c0.real * u.r0c0.real + u.r0c0.imag * u.r0c0.imag + u.r1c0.real * u.r1c0.real +
                            u.r1c0.imag * u.r1c0.imag - 1) > 1e-2)
                    return 0;
                if (absReal(u.r0c1.real * u.r0c1.real + u.r0c1.imag * u.r0c1.imag + u.r1c1.real * u.r1c1.real +
                            u.r1c1.imag * u.r1c1.imag - 1) > 1e-2)
                    return 0;
                if (absReal(u.r0c0.real * u.r0c1.real + u.r0c0.imag * u.r0c1.imag + u.r1c0.real * u.r1c1.real +
                            u.r1c0.imag * u.r1c1.imag) > 1e-2)
                    return 0;
                if (absReal(u.r0c1.real * u.r0c0.imag - u.r0c0.real * u.r0c1.imag + u.r1c1.real * u.r1c0.imag -
                            u.r1c0.real * u.r1c1.imag) > 1e-2)
                    return 0;
                return 1;
            };

            Gate mergedUGate = uGate;
            const ComplexMatrix2 mat = getUnitaryMatrix(anotherGate.operand, anotherGate.type);
            const auto printUnitaryMatrix = [](const ComplexMatrix2 &mat) {
                printf("[[%13.8f + %13.8fj, %13.8f + %13.8fj],\n", mat.r0c0.real, mat.r0c0.imag, mat.r0c1.real,
                       mat.r0c1.imag);
                printf(" [%13.8f + %13.8fj, %13.8f + %13.8fj]]\n", mat.r1c0.real, mat.r1c0.imag, mat.r1c1.real,
                       mat.r1c1.imag);
            };
            if (uGateOnLeft) {
                multUnitaryMatrix(mergedUGate.operand.unitaryMatrix, uGate.operand.unitaryMatrix, mat);
            } else {
                multUnitaryMatrix(mergedUGate.operand.unitaryMatrix, mat, uGate.operand.unitaryMatrix);
            }
#ifdef CHECK_UNITARY
            printf("%s\n", getOperatorNameString(anotherGate.type));
            if (!isMatrixUnitary(uGate.operand.unitaryMatrix)) {
                printf("!!!ERROR!!! in merge, uGate is a non-unitary gate\n");
                printUnitaryMatrix(uGate.operand.unitaryMatrix);
                while (true)
                    ;
            }
            if (!isMatrixUnitary(mat)) {
                printf("!!!ERROR!!! in merge, anotherGate is a non-unitary gate\n");
                printUnitaryMatrix(mat);
                while (true)
                    ;
            }
            if (!isMatrixUnitary(mergedUGate.operand.unitaryMatrix)) {
                printf("!!!ERROR!!! in merge, unitary gates merged into a non-unitary gate\n");
                printUnitaryMatrix(mergedUGate.operand.unitaryMatrix);
                while (true)
                    ;
            }
#endif
            return mergedUGate;
        };
        const auto getMergedRotate = [](const Gate &rGate1, const Gate &rGate2) -> Gate {
            Gate mergedRGate = rGate1;
            mergedRGate.operand.angle = rGate1.operand.angle + rGate2.operand.angle;
            return mergedRGate;
        };

        if (node.gate.targetMask == topNode->gate.targetMask && node.gate.controlMask == topNode->gate.controlMask) {
            Gate *gate1, *gate2;
            if (node.gate.type > topNode->gate.type) {
                gate1 = &topNode->gate;
                gate2 = &node.gate;
            } else {
                gate2 = &topNode->gate;
                gate1 = &node.gate;
            }

            if ((int) gate2->type < 8) {
                auto newOp = (GateType) (((int) gate1->type + (int) gate2->type) % 8);
                if (newOp == 0) {
                    return deleteGate();
                } else {
                    return overrideGateNoParam(newOp);
                }
            } else if (gate2->type == OP_PHASE_SHIFT) {
                auto newGate = getMergedPhaseShift(*gate1, *gate2);
                return overrideGate(newGate);
            } else if (gate2->type == OP_UNITARY) { // need test, get wrong answer
                if (gate2 == &topNode->gate) {
                    return overrideGate(getMergedUnitary(*gate1, *gate2, false));
                } else /* gate2 == &node.gate */ {
                    return overrideGate(getMergedUnitary(*gate1, *gate2, true));
                }
            } else {
                if (gate1->type == gate2->type) {
                    if (OP_PAULI_X <= (int) gate1->type && (int) gate1->type <= OP_HADAMARD) {
                        return deleteGate();
                    } else if (gate1->type == OP_ROTATE_X || gate1->type == OP_ROTATE_Y ||
                               gate1->type == OP_ROTATE_Z) {
                        if (gate1->operand.angle == -gate2->operand.angle) {
                            return deleteGate();
                        }
                        auto newGate = getMergedRotate(*gate1, *gate2);
                        return overrideGate(newGate);
                    }
                }
            }
        }
        return false;
    };

    for (auto i = 0; i < barrierNodeStack.size(); i++) {
        if (((1ULL << i) & node.barrierMask) != 0) {
            if (!barrierNodeStack[i].empty()) {
                if (mergeGates) {
                    if (barrierNodeStack[i].top()->rearNodes.empty() && reduceGate(i)) {
                        allNodes.pop_back();
                        return;
                    }
                }
                node.prevNodes.push_back(barrierNodeStack[i].top());
                barrierNodeStack[i].top()->rearNodes.push_back(&node);
            }
            barrierNodeStack[i].push(&node);
        }
    }
    if (node.prevNodes.empty()) {
        frontNodes.push_back(&node);
    }
}

DAG *DAG::setSectionGates(std::vector<Gate> &gates, u64 &targetMask, int numQubitsPerTask, int maxGateNum) {
    /**
     * section赋targetMask初始值0x1f
     */
    for (auto &node : allNodes) {
        if ((int) node.gate.type >= (int) OP_PAULI_X) {
            node.targetMask = node.gate.targetMask;
        } else {
            node.targetMask = 0;
        }
        node.remainPrevNum = node.prevNodes.size();
        node.visited = false;
        node.selected = false;
    }
    for (auto &frontNode : frontNodes) {
        dfs(frontNode, targetMask, numQubitsPerTask);
    }
    //	printf("DAG targetMask = %llx\n", section.targetMask);
    auto next_dag = new DAG((int) barrierNodeStack.size());
    for (auto &node : allNodes) {
        if (node.selected && gates.size() < maxGateNum) {
            gates.push_back(node.gate);
        } else {
            next_dag->addGate(node.gate);
        }
    }
    return next_dag;
}

void DAG::dfs(Node *node, u64 &targetMask, int numQubitsPerTask) {
    node->visited = true;
    if (countOneUint64(targetMask | node->targetMask) <= numQubitsPerTask) {
        node->selected = true;
        targetMask |= node->targetMask;
        for (auto &rearNode : node->rearNodes) {
            if (!rearNode->visited && --rearNode->remainPrevNum == 0) {
                dfs(rearNode, targetMask, numQubitsPerTask);
            }
        }
    }
}
