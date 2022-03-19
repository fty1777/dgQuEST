// Distributed under MIT licence. See https://github.com/fty1777/dgQuEST/blob/main/LICENSE for details

#include "time_predictor.h"

#include <easylogging++.h>
#include <fmt/core.h>
#include <mpi.h>

#include "timer.hpp"

void TimePredictor::sampleAll(int numQubitsMin, int numQubitsMax) {
    if (!samplingEnabled) {
        LOG(ERROR) << fmt::format("Sampling not enabled yet, automatically enabled sampling for qubits {}", numQubitsMax);
        enableSampling(numQubitsMax);
    }
    for (int i = numQubitsMin; i <= numQubitsMax; i++) {
        sample(i);
    }
}

void TimePredictor::sample(int numQubits) {
    printf("[%d] Sampling on numQubits == %d\n", rankId, numQubits);
    if (!samplingEnabled) {
        LOG(ERROR) << fmt::format("Sampling not enabled yet, automatically enabled sampling for qubits {}", numQubits);
        enableSampling(numQubits);
    }
    sampleCalc(numQubits);
    sampleIo(numQubits);
}

void TimePredictor::sampleCalc(int numQubits) {
    processTimeTotal[numQubits] = 0;
    processTimeCount[numQubits] = 0;
    sampleGateTime(numQubits);
    processTime[numQubits] = processTimeTotal[numQubits] / processTimeCount[numQubits];
    LOG(INFO) << fmt::format("Gate agnostic kernel process time ({} qubits): {:.3f} ms", numQubits, processTime[numQubits]);
}

void TimePredictor::sampleIo(int numQubits) {
    sampleH2d(numQubits);
    sampleD2h(numQubits);
    sampleOverlapped(numQubits);
    sampleComm(numQubits);
}

void TimePredictor::sampleH2d(int numQubits) {
    constexpr size_t REPEATS = 8;
    double time_sum = 0;
    for (int i = 0; i < REPEATS; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpyAsync(vec, vecHost, sizeof(qComplex) * (1ULL << numQubits), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(end));
        CUDA_CHECK(cudaEventSynchronize(end));
        float time;
        CUDA_CHECK(cudaEventElapsedTime(&time, start, end));
        time_sum += time;
    }
    bandwidthH2d[numQubits] = (1ULL << numQubits) * sizeof(qComplex) * REPEATS / (double) time_sum;
    LOG(INFO) << fmt::format("Bandwidth host to device (h2d) ({} qubits): {} GB/s", numQubits, bandwidthH2d[numQubits] / (1 << 30) * 1000);
}

void TimePredictor::sampleD2h(int numQubits) {
    constexpr size_t REPEATS = 8;
    double time_sum = 0;
    for (int i = 0; i < REPEATS; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpyAsync(vecHost, vec, sizeof(qComplex) * (1ULL << numQubits), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(end));
        CUDA_CHECK(cudaEventSynchronize(end));
        float time;
        CUDA_CHECK(cudaEventElapsedTime(&time, start, end));
        time_sum += time;
    }
    bandwidthD2h[numQubits] = (1ULL << numQubits) * sizeof(qComplex) * REPEATS / (double) time_sum;
    LOG(INFO) << fmt::format("Bandwidth device to host (d2h) ({} qubits): {} GB/s", numQubits, bandwidthD2h[numQubits] / (1 << 30) * 1000);
}

void TimePredictor::sampleOverlapped(int numQubits) {
    constexpr size_t REPEATS = 8;
    double time_d2h_sum = 0, time_h2d_sum = 0;
    for (int i = 0; i < REPEATS; i++) {
        CUDA_CHECK(cudaEventRecord(start, streams[0]));
        CUDA_CHECK(cudaMemcpyAsync(vecHost, vec, sizeof(qComplex) * (1ULL << numQubits), cudaMemcpyDeviceToHost, streams[0]));
        CUDA_CHECK(cudaEventRecord(end, streams[0]));

        CUDA_CHECK(cudaEventRecord(start2, streams[1]));
        CUDA_CHECK(cudaMemcpyAsync(vec, vecHost, sizeof(qComplex) * (1ULL << numQubits), cudaMemcpyHostToDevice, streams[1]));
        CUDA_CHECK(cudaEventRecord(end2, streams[1]));

        CUDA_CHECK(cudaEventSynchronize(end));
        CUDA_CHECK(cudaEventSynchronize(end2));
        float time_d2h, time_h2d;
        CUDA_CHECK(cudaEventElapsedTime(&time_d2h, start, end));
        CUDA_CHECK(cudaEventElapsedTime(&time_h2d, start2, end2));
        time_d2h_sum += time_d2h;
        time_h2d_sum += time_h2d;
    }
    bandwidthD2hOverlapped[numQubits] = (1ULL << numQubits) * sizeof(qComplex) * REPEATS / (double) time_d2h_sum;
    bandwidthH2dOverlapped[numQubits] = (1ULL << numQubits) * sizeof(qComplex) * REPEATS / (double) time_h2d_sum;
    LOG(INFO) << fmt::format("Bandwidth device to host (d2h, overlapped) ({} qubits): {} GB/s", numQubits, bandwidthD2hOverlapped[numQubits] / (1 << 30) * 1000);
    LOG(INFO) << fmt::format("Bandwidth host to device (h2d, overlapped) ({} qubits): {} GB/s", numQubits, bandwidthH2dOverlapped[numQubits] / (1 << 30) * 1000);
}

void TimePredictor::sampleComm(int numQubits) {

    if (numRanks == 1) {
        bandwidthComm[numQubits] = MAXFLOAT;
    } else {
        constexpr size_t REPEATS = 16;
        const double time_start = getWallTime();
        for (int i = 0; i < REPEATS; i++) {
            if (rankId == 0) {
                MPI_CHECK(MPI_Send(vecHost, (1ULL << numQubits), MPI_C_DOUBLE_COMPLEX, 1, 0, MPI_COMM_WORLD));
            } else if (rankId == 1) {
                MPI_CHECK(MPI_Recv(vecHost, (1ULL << numQubits), MPI_C_DOUBLE_COMPLEX, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            }
            MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        }
        double time = (getWallTime() - time_start) * 1000;
        LOG(WARNING) << fmt::format("[{}] Time comm: {} ms", rankId, time);
        MPI_CHECK(MPI_Bcast(&time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD));
        bandwidthComm[numQubits] = (1ULL << numQubits) * sizeof(qComplex) * REPEATS / time;
    }
    LOG(INFO) << fmt::format("Bandwidth comm ({} qubits): {} GB/s", numQubits, bandwidthComm[numQubits] / (1 << 30) * 1000);
}

void TimePredictor::sampleGateTime(int numQubits) {
    for (GateType type = GateType::OP_T_GATE; type <= GateType::OP_PHASE_SHIFT; *((int*) &type) += 1) {
        std::vector<Gate> gates;
        Gate gate1{0, type, 0, 0x8};
        Gate gate2{0, type, 0, 0x8};
        memset(&gate1.operand, 0, sizeof(GateOperand));
        memset(&gate2.operand, 0, sizeof(GateOperand));

        gates.push_back(gate1);

        // warm up
        executeGateTimeSampling(gates, numQubits);

        const double time_once = executeGateTimeSampling(gates, numQubits);
        for (int i = 0; i < repeatTimes; i += 2) {
            gates.push_back(gate1);
            gates.push_back(gate2);
        }
        const double time_repeat = executeGateTimeSampling(gates, numQubits);
        const double time = (time_repeat - time_once) / repeatTimes;

        singleQubitGateTime[numQubits][type] = time;
        processTimeCount[numQubits] += 1;
        processTimeTotal[numQubits] += time_once - time;

        LOG(INFO) << fmt::format("Time single qubit gate {:18}({} qubits): {:.3f} ms", getGateTypeString(type), numQubits, time);
    }

    for (GateType type = GateType::OP_ROTATE_Z; type <= GateType::OP_UNITARY; *((int*) &type) += 1) {
        std::vector<Gate> gates;
        Gate gate1{0, type, 0x8, 0x0};
        Gate gate2{0, type, 0x8, 0x0};
        memset(&gate1.operand, 0, sizeof(GateOperand));
        memset(&gate2.operand, 0, sizeof(GateOperand));

        gates.push_back(gate1);

        // warm up
        executeGateTimeSampling(gates, numQubits);

        const double time_once = executeGateTimeSampling(gates, numQubits);
        for (int i = 0; i < repeatTimes; i += 2) {
            gates.push_back(gate1);
            gates.push_back(gate2);
        }
        const double time_repeat = executeGateTimeSampling(gates, numQubits);
        const double time = (time_repeat - time_once) / repeatTimes;

        singleQubitGateTime[numQubits][type] = time;
        processTimeCount[numQubits] += 1;
        processTimeTotal[numQubits] += time_once - time;

        LOG(INFO) << fmt::format("Time single qubit gate {:18}({} qubits): {:.3f} ms", getGateTypeString(type), numQubits, time);
    }
}

double TimePredictor::executeGateTimeSampling(const std::vector<Gate>& gates, int numQubits) {
    FusedGate fusedGate;
    fusedGate.initFunction = INIT_FROM_AMPS;
    for (const auto& gate : gates) {
        fusedGate.targetMask |= gate.targetMask;
        fusedGate.gates.push_back(gate);
    }

    CUDA_CHECK(cudaEventRecord(start));
    launchKernelForSampling(fusedGate, numQubits, numQubits, 0, vec);
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float time;
    CUDA_CHECK(cudaEventElapsedTime(&time, start, end));

    return time;
}

double TimePredictor::predictGateTime(const Gate& gate, int numQubits) {
    return singleQubitGateTime[numQubits][gate.type];
}

double TimePredictor::predictFusedGateTime(const FusedGate& fusedGate, int numQubits) {
    double time = processTime[numQubits];
    for (const auto& gate : fusedGate.gates) {
        time += predictGateTime(gate, numQubits);
    }
    return time;
}

double TimePredictor::predictSectionCalcTimePerTaskFromLocalView(const SectionLocalView& secLV, int numQubits) {
    double time = 0;
    for (const auto& fusedGate : secLV.fusedGates) {
        const double t = predictFusedGateTime(fusedGate, numQubits);
        time += t;
        // LOG(INFO) << fmt::format("Fused kernel ({:3} gates) time: {} ms", fusedGate.gates.size(), t);
    }
    return time;
}

double TimePredictor::predictSectionCalcTimePerTask(const SectionGlobalView& secGV, int numQubits) {
    double time = 0;
    for (const auto& it : secGV.localViews) {
        const auto& secLV = it.second;
        time = std::max(time, predictSectionCalcTimePerTaskFromLocalView(secLV, numQubits));
    }
    return time;
}

double TimePredictor::predictPartitionPlan(const std::vector<SectionGlobalView>& sectionGlobalViewList, int numQubits, int K) {
    if (sectionGlobalViewList.empty()) {
        return 0;
    }

    const int numTasksTotal = 1 << K;
    const int numPagesPerTask = 1 << K;

    int numTasksTable[numRanks];
    int taskBeginOffsetTable[numRanks];

    for (int i = 0; i < numRanks; i++) {
        numTasksTable[i] = numTasksTotal / numRanks;
    }
    for (int i = 0; i < numRanks; i++) {
        taskBeginOffsetTable[i] = numTasksTotal / numRanks * i;
    }

    const int numTasksThisNode = numTasksTable[rankId];
    const int taskBeginOffset = taskBeginOffsetTable[rankId];

    double timeTotal = 0;
    const int numQubitsPerTask = numQubits - K;
    const double alpha = (double) numTasksThisNode / numTasksTotal;
    const int a = numTasksThisNode / numDevices;
    const int b = numTasksThisNode - a * numDevices;
    const u64 size = (1ULL << numQubitsPerTask) * sizeof(qComplex);
    const double t_d2h = size / bandwidthD2h[numQubitsPerTask];
    const double t_d2h_ol = size / bandwidthD2hOverlapped[numQubitsPerTask];
    const double t_h2d = size / bandwidthH2d[numQubitsPerTask];
    const double t_send = size / bandwidthComm[numQubitsPerTask];
    const double ptRatio = 1.0 / numPagesPerTask; // page task ratio

    for (int i = 0; i < sectionGlobalViewList.size() && i <= 0; i++) {
        auto& secGV = sectionGlobalViewList[i];
        const double t_calc = predictSectionCalcTimePerTask(secGV, numQubitsPerTask);

        const double time1 = t_calc + alpha * t_d2h + (1 - alpha) * t_send;

        const double lastSectionOffset = i == sectionGlobalViewList.size() - 1 ? (1 - alpha) * (t_d2h - t_send) : 0;

        const double timeSection = time1 + lastSectionOffset;
        // no first h2d because init state is initialized directly in GPU
        if (rankId == 0) {
            LOG(INFO) << fmt::format("[{:3}] K = {}, Section {}, time == {}", rankId, K, secGV.sectionIndex, timeSection);
        }
        double timeSectionMax;
        MPI_CHECK(MPI_Allreduce(&timeSection, &timeSectionMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
        timeTotal += timeSectionMax;
        if (rankId == 0) {
            LOG(INFO) << fmt::format("[ttl] K = {}, Section {}, time == {}", K, secGV.sectionIndex, timeSectionMax);
        }
    }

    for (int i = 1; i < sectionGlobalViewList.size() && i <= 1; i++) {
        auto& secGV = sectionGlobalViewList[i];
        const double t_calc = predictSectionCalcTimePerTask(secGV, numQubitsPerTask);
        const double time1 = t_h2d / numTasksThisNode;
        const double time2 = a * (t_calc + alpha * t_d2h + (1 - alpha) * t_send * numDevices);
        const double time3 = b == 0 ? 0 : t_calc + alpha * t_d2h + (1 - alpha) * t_send * b;

        const double lastSectionOffset = i == sectionGlobalViewList.size() - 1 ? a * (1 - alpha) * (t_d2h - t_send * numDevices) + (1 - alpha) * (t_d2h - t_send * b) : 0;

        const double timeSection = time1 + time2 + time3 + lastSectionOffset;

        if (rankId == 0) {
            LOG(INFO) << fmt::format("[{:3}] K = {}, Section {}, time1 == {}", rankId, K, secGV.sectionIndex, time1);
            LOG(INFO) << fmt::format("[{:3}] K = {}, Section {}, time2 == {}", rankId, K, secGV.sectionIndex, time2);
            LOG(INFO) << fmt::format("[{:3}] K = {}, Section {}, time3 == {}", rankId, K, secGV.sectionIndex, time3);
            LOG(INFO) << fmt::format("[{:3}] K = {}, Section {}, lastSectionOffset == {}", rankId, K, secGV.sectionIndex, lastSectionOffset);
            LOG(INFO) << fmt::format("[{:3}] K = {}, Section {}, time == {}", rankId, K, secGV.sectionIndex, timeSection);
        }
        double timeSectionMax;
        MPI_CHECK(MPI_Allreduce(&timeSection, &timeSectionMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
        timeTotal += timeSectionMax;
        if (rankId == 0) {
            LOG(INFO) << fmt::format("[ttl] K = {}, Section {}, time == {}", K, secGV.sectionIndex, timeSectionMax);
        }
    }


    for (int i = 2; i < sectionGlobalViewList.size(); i++) {
        auto& secGV = sectionGlobalViewList[i];
        const double t_calc = predictSectionCalcTimePerTask(secGV, numQubitsPerTask);
        const double time1 = t_h2d;
        const double time2 = a * (t_calc + (alpha - ptRatio) * t_d2h_ol + (1 - alpha) * t_send * numDevices + ptRatio * (t_d2h + t_h2d));
        const double nonOverlapOffset = (alpha - ptRatio) * (t_d2h - t_d2h_ol);
        const double time3 = b == 0 ? 0 : t_calc + (alpha - ptRatio) * t_d2h_ol + (1 - alpha) * t_send * b + ptRatio * (t_d2h + t_h2d);

        const double lastSectionOffset = i == sectionGlobalViewList.size() - 1 ? a * (1 - alpha) * (t_d2h_ol - t_send * numDevices) + (1 - alpha) * (t_d2h_ol - t_send * b) : 0;

        const double timeSection = time1 + time2 + nonOverlapOffset + time3 + lastSectionOffset;

        if (rankId == 0) {
            LOG(INFO) << fmt::format("[{:3}] K = {}, Section {}, time1 == {}", rankId, K, secGV.sectionIndex, time1);
            LOG(INFO) << fmt::format("[{:3}] K = {}, Section {}, time2 == {}", rankId, K, secGV.sectionIndex, time2);
            LOG(INFO) << fmt::format("[{:3}] K = {}, Section {}, time3 == {}", rankId, K, secGV.sectionIndex, time3);
            LOG(INFO) << fmt::format("[{:3}] K = {}, Section {}, lastSectionOffset == {}", rankId, K, secGV.sectionIndex, lastSectionOffset);
            LOG(INFO) << fmt::format("[{:3}] K = {}, Section {}, time == {}", rankId, K, secGV.sectionIndex, timeSection);
        }
        double timeSectionMax;
        MPI_CHECK(MPI_Allreduce(&timeSection, &timeSectionMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
        timeTotal += timeSectionMax;
        if (rankId == 0) {
            LOG(INFO) << fmt::format("[ttl] K = {}, Section {}, time == {}", K, secGV.sectionIndex, timeSectionMax);
        }
    }

    return timeTotal;
}

std::map<int, double> TimePredictor::predictCircuit(const std::vector<Gate>& gates, int numQubits, int minK, int maxK) {
    std::map<int, double> Tpreds;
    for (int K = minK; K <= maxK; K++) {

        const int numTasksTotal = 1 << K;
        const int numTasksThisNode = numTasksTotal / numRanks;
        const int taskBeginOffset = numTasksTotal / numRanks * rankId;
        const int numQubitsPerTask = numQubits - K;
        const int numQubitsPerPage = numQubits - K * 2;

        DAG* dag = new DAG(numQubits);
        for (const auto& gate : gates) {
            dag->addGate(gate);
        }

        std::vector<SectionGlobalView> secGVList;
        int sectionIndex = 0;
        secGVList.clear();
        secGVList.emplace_back(sectionIndex++);
        secGVList.back().initFunction = INIT_CLASSICAL_STATE;
        secGVList.back().initParam.stateIndex = 0;

        while (true) {
            auto next_dag = dag->setSectionGates(secGVList.back().gates, secGVList.back().targetMask, numQubitsPerTask);
            delete dag;
            dag = next_dag;

            if (dag->empty()) {
                deriveMappings(secGVList, numQubits, numQubitsPerTask, numQubitsPerPage);

                for (auto& secGV : secGVList) {
                    for (int taskId = taskBeginOffset; taskId < taskBeginOffset + numTasksThisNode; taskId++) {
                        ::translateSection(&secGV, taskId, numQubits, numQubitsPerTask);
                    }
                }

                for (auto& secGV : secGVList) {
                    for (auto& it : secGV.localViews) {
                        auto& secLV = it.second;
                        ::splitSectionLocalViewIntoFusedGates(&secLV, numQubits);
                    }
                }

                Tpreds[K] = predictPartitionPlan(secGVList, numQubits, K);
                break;
            }

            secGVList.emplace_back(sectionIndex++);
            secGVList.back().initFunction = INIT_FROM_AMPS;
        }
    }
    return Tpreds;
}

void TimePredictor::save(const std::string& filename) {
    auto f = fopen(filename.c_str(), "w");

    for (const auto& it : singleQubitGateTime) {
        const auto& numQubits = it.first;
        fputs(fmt::format("GateTime {}\n", numQubits).c_str(), f);
        const auto& timeMap = it.second;
        for (const auto& gateTime : timeMap) {
            const auto& gate = gateTime.first;
            const auto& time = gateTime.second;
            fputs(fmt::format("{} {}\n", getGateTypeString(gate), time).c_str(), f);
        }
    }

    for (const auto& it : bandwidthH2d) {
        const auto& numQubits = it.first;
        const auto& time = it.second;
        fputs(fmt::format("BandwidthH2d {} {}\n", numQubits, time).c_str(), f);
    }
    for (const auto& it : bandwidthD2h) {
        const auto& numQubits = it.first;
        const auto& time = it.second;
        fputs(fmt::format("BandwidthD2h {} {}\n", numQubits, time).c_str(), f);
    }
    for (const auto& it : bandwidthD2hOverlapped) {
        const auto& numQubits = it.first;
        const auto& time = it.second;
        fputs(fmt::format("BandwidthD2hOverlapped {} {}\n", numQubits, time).c_str(), f);
    }
    for (const auto& it : bandwidthComm) {
        const auto& numQubits = it.first;
        const auto& time = it.second;
        fputs(fmt::format("BandwidthComm {} {}\n", numQubits, time).c_str(), f);
    }
    for (const auto& it : processTime) {
        const auto& numQubits = it.first;
        const auto& time = it.second;
        fputs(fmt::format("ProcessTime {} {}\n", numQubits, time).c_str(), f);
    }
    fclose(f);
}
void TimePredictor::load(const std::string& filename) {
    auto f = fopen(filename.c_str(), "r");

    char buf[512];
    int numQubits;
    while (fscanf(f, "%s%d", buf, &numQubits) == 2) {
        if (strcmp(buf, "GateTime") == 0) {
            for (GateType type = GateType::OP_T_GATE; type <= GateType::OP_UNITARY; *((int*) &type) += 1) {
                if (type == OP_ROTATE_Z_FAKE)
                    continue;
                float time;
                fscanf(f, "%s%f", buf, &time);
                singleQubitGateTime[numQubits][type] = time;
            }
        } else if (strcmp(buf, "BandwidthH2d") == 0) {
            float time;
            fscanf(f, "%f", &time);
            bandwidthH2d[numQubits] = time;
        } else if (strcmp(buf, "BandwidthD2h") == 0) {
            float time;
            fscanf(f, "%f", &time);
            bandwidthD2h[numQubits] = time;
        } else if (strcmp(buf, "BandwidthD2hOverlapped") == 0) {
            float time;
            fscanf(f, "%f", &time);
            bandwidthD2hOverlapped[numQubits] = time;
        } else if (strcmp(buf, "BandwidthComm") == 0) {
            float time;
            fscanf(f, "%f", &time);
            bandwidthComm[numQubits] = time;
        } else if (strcmp(buf, "ProcessTime") == 0) {
            float time;
            fscanf(f, "%f", &time);
            processTime[numQubits] = time;
        } else {
            LOG(FATAL) << fmt::format("Unrecognized data name: {}", buf);
        }
    }
    fclose(f);
}

TimePredictor::TimePredictor(int samplingMaxNumQubits)
    : TimePredictor() {
    this->enableSampling(samplingMaxNumQubits);
}

TimePredictor::~TimePredictor() {
    if (samplingEnabled) {
        CUDA_CHECK(cudaStreamDestroy(streams[0]));
        CUDA_CHECK(cudaStreamDestroy(streams[1]));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(end));
        CUDA_CHECK(cudaEventDestroy(start2));
        CUDA_CHECK(cudaEventDestroy(end2));
        CUDA_CHECK(cudaFree(vec));
        CUDA_CHECK(cudaFreeHost(vecHost));
    }
}

void TimePredictor::enableSampling(int numQubits) {
    if (samplingEnabled) {
        if (numQubits > samplingMaxNumQubit) {
            CUDA_CHECK(cudaFree(vec));
            CUDA_CHECK(cudaFreeHost(vecHost));
            CUDA_CHECK(cudaMalloc(&vec, (1ULL << numQubits) * sizeof(qComplex)));
            CUDA_CHECK(cudaMallocHost(&vecHost, (1ULL << numQubits) * sizeof(qComplex)));
            samplingMaxNumQubit = numQubits;
        }
    } else {
        CUDA_CHECK(cudaStreamCreate(&streams[0]));
        CUDA_CHECK(cudaStreamCreate(&streams[1]));
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&end));
        CUDA_CHECK(cudaEventCreate(&start2));
        CUDA_CHECK(cudaEventCreate(&end2));
        CUDA_CHECK(cudaMalloc(&vec, (1ULL << numQubits) * sizeof(qComplex)));
        CUDA_CHECK(cudaMallocHost(&vecHost, (1ULL << numQubits) * sizeof(qComplex)));
        samplingMaxNumQubit = numQubits;
        samplingEnabled = true;
    }
}

TimePredictor::TimePredictor() {
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rankId));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numRanks));
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    LOG(WARNING) << fmt::format("[{}] Predictor created with {} GPUs. {} nodes (ranks) in total.", rankId, numDevices, numRanks);
}
