// Distributed under MIT licence. See https://github.com/fty1777/dgQuEST/blob/main/LICENSE for details

#include <cuda_runtime_api.h>
#include <easylogging++.h>
#include <fmt/core.h>
#include <mpi.h>
#include <numa.h>

#include <queue>

#include "QuEST_backend.h"
#include "time_predictor.h"
#include "timer.hpp"

#define DEVICE_BUFFER_QREALS ((1ULL << 19) * 32 * 2) // Too magic, already forgot the reason for using this value

#define OPTIMIZE_ZERO

#define SET_PAGE        0x80000 // MPI tag
#define GPU_DATA_TO_CPU 0x90000 // MPI tag

#define HEADER_BUF_SIZE   4
#define HEADER_SECTION_ID 0
#define HEADER_TASK_ID    1
#define HEADER_PAGE_INDEX 2
#define HEADER_ALL_ZERO   3

inline static int getCpuRank(int taskId, Qureg &qureg) {
    auto *state = dynamic_cast<QuregStateDistributed *>(qureg.state);
    for (int i = qureg.env.value->numRanks - 1; i >= 0; i--) {
        if (taskId >= state->taskBeginOffsetTable[i]) {
            return i;
        }
    }
    return -1;
}

QuESTEnvValueDistributed::QuESTEnvValueDistributed(int numDevices, int minK, int maxK)
    : minK{minK},
      maxK{maxK} {

    this->numDevices = numDevices;
    for (int d = 0; d < numDevices; d++) {
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaMalloc(&deviceBuffer[d], (u64) DEVICE_BUFFER_QREALS * sizeof(qreal)));
    }

    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rankId));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numRanks));
    LOG(INFO) << fmt::format("rankNum: {}", numRanks);
    LOG(INFO) << fmt::format("rank {} initialized", rankId);
}

QuESTEnvValueDistributed::QuESTEnvValueDistributed(int numDevices, int minK, int maxK, int designatedK)
    : QuESTEnvValueDistributed(numDevices, minK, maxK) {
    this->designatedK = designatedK;
    this->searchK = false;
}

void QuESTEnvValueDistributed::growMemPool(u64 numPages, u64 numAmpsPerPage) {
    LOG(INFO) << fmt::format("Growing memory pool for {} more {}-amplitude({} MB) pages", numPages, numAmpsPerPage, numAmpsPerPage * 16 / 1048576);
    u64 growAmps = numPages * numAmpsPerPage;
    qComplex *newMemPoolBase;
    CUDA_CHECK(cudaMallocHost(&newMemPoolBase, (u64) sizeof(qComplex) * growAmps, cudaHostAllocPortable));

    envMemPoolPagesNumTotal += numPages;

    for (int i = 0; i < envMemPoolPagesNumTotal; i++) {
        envMemPoolAvailQueue[numAmpsPerPage].push(newMemPoolBase + i * numAmpsPerPage);
    }

    envMemPoolBlocks.emplace(newMemPoolBase, newMemPoolBase + growAmps, numAmpsPerPage);
}

void QuESTEnvValueDistributed::statevec_createQureg(Qureg &qureg, int numQubits) {
    qureg.numAmpsTotal = 1ULL << qureg.numQubitsInStateVec;
    qureg.env.value = this;

    auto *state = new QuregStateDistributed(qureg);
    qureg.state = state;
    state->qureg.state = state; // IMPORTANT: This is necessary because qureg is passed as value

    LOG(INFO) << fmt::format("A qureg with {} qubits created", numQubits);
}

void QuESTEnvValueDistributed::searchTaskPartition(Qureg &qureg) {
    LOG(INFO) << fmt::format("minK = {}, maxK = {}", minK, maxK);
    auto *predictor = new TimePredictor;

    predictor->load("predict_data.txt");
    auto predResults = predictor->predictCircuit(qureg.state->gates, qureg.numQubitsInStateVec, minK, maxK);
    delete predictor;

    if (rankId == 0) {
        LOG(INFO) << "Pred result: ";
        for (const auto &result : predResults) {
            auto K = result.first;
            auto timeMilliseconds = result.second;
            LOG(INFO) << fmt::format("   K == {}, time == {} s", K, timeMilliseconds / 1000);
        }
    }

    auto minTime = predResults[minK];
    auto minTimeK = minK;
    for (const auto &result : predResults) {
        auto K = result.first;
        auto timeMilliseconds = result.second;
        if (timeMilliseconds < minTime) {
            minTimeK = K;
            minTime = timeMilliseconds;
        }
    }

    if (searchK) {
        LOG(INFO) << fmt::format("[{}] Selected K == {}, time pred == {} s", rankId, minTimeK, minTime / 1000);
        applyK(qureg, minTimeK);
    } else {
        LOG(INFO) << fmt::format("[{}] Designated K == {}, time pred == {} s", rankId, designatedK, predResults[designatedK] / 1000);
        applyK(qureg, designatedK);
    }
}

void QuESTEnvValueDistributed::applyK(Qureg &qureg, int numTaskIdQubits) {
    auto *state = dynamic_cast<QuregStateDistributed *>(qureg.state);
    state->numTaskIdQubits = numTaskIdQubits;

    std::string s;
    s += "\n       Tasks Table: ";
    for (int i = 0; i < numRanks; i++) {
        state->numTasksTable[i] = (1 << state->numTaskIdQubits) / numRanks;
        s += fmt::format("{} ", state->numTasksTable[i]);
    }
    s += "\nTasks Begin Offset: ";
    for (int i = 0; i < numRanks; i++) {
        state->taskBeginOffsetTable[i] = (1 << state->numTaskIdQubits) / numRanks * i;
        s += fmt::format("{} ", state->taskBeginOffsetTable[i]);
    }
    LOG(INFO) << s;

    state->numTasksThisNode = state->numTasksTable[rankId];
    state->taskBeginOffset = state->taskBeginOffsetTable[rankId];

    state->numQubitsPerPage = qureg.numQubitsInStateVec - state->numTaskIdQubits * 2;
    state->numAmpsPerPage = 1ULL << state->numQubitsPerPage;

    state->numQubitsPerTask = qureg.numQubitsInStateVec - state->numTaskIdQubits;
    state->numAmpsPerTask = 1ULL << state->numQubitsPerTask;

    state->numPagesPerTask = 1ULL << state->numTaskIdQubits;
    state->numTasksTotal = 1ULL << state->numTaskIdQubits;

    log_files.resize(state->numTasksTotal);
    for (auto i = state->taskBeginOffset; i < state->taskBeginOffset + state->numTasksThisNode; i++) {
        std::string file_name = fmt::format("rank{}task{:02d}.log", rankId, i);
        log_files[i] = fopen(file_name.c_str(), "w");
    }

    for (int i = 0; i < 2; i++) {
        state->pageTable[i].resize(state->numPagesPerTask * state->numTasksTotal, nullptr);
    }
    state->probsOfZero.resize(state->numTasksThisNode, nullptr);

    const u64 sizePerTask = state->numAmpsPerTask * sizeof(Complex);
    std::string sizeStr;
    if (sizePerTask >> 30) {
        sizeStr = fmt::format("{} GB", sizePerTask >> 30);
    } else {
        sizeStr = fmt::format("{} MB", sizePerTask >> 20);
    }

    for (int d = 0; d < numDevices; d++) {
        CUDA_CHECK(cudaSetDevice(d));
        LOG(INFO) << fmt::format("[device:{:2d}] {} CUDA memory to be allocated.", d, sizeStr);
        CUDA_CHECK(cudaMalloc((void **) &state->deviceStateVecs[d], state->numAmpsPerTask * sizeof(qComplex)));
        CUDA_CHECK(cudaMalloc(&deviceBuffer[d], DEVICE_BUFFER_QREALS * sizeof(qreal)));
    }


    state->memPoolAvailQueue.clear();
    const u64 numPagesStateVector = state->numPagesPerTask * state->numTasksThisNode;
    const u64 numPagesBuffer = numDevices * 2;
    const u64 numPagesTotal = numPagesStateVector + numPagesBuffer;
    LOG(INFO) << fmt::format("Pages: {} ({} + {})", numPagesTotal, numPagesStateVector, numPagesBuffer);

    growMemPool(numPagesTotal, state->numAmpsPerPage);

    for (u64 i = 0; i < numPagesTotal; i++) {
        auto *newEnvPage = mallocEnvPage(state->numAmpsPerPage);
        LOG_IF(newEnvPage == nullptr, FATAL) << fmt::format("Out of memory! {}:{}", __FILE__, __LINE__);
        state->memPoolAllocatedEnvPage.push_back(newEnvPage);
        state->memPoolAvailQueue.push_back(newEnvPage);
    }
    LOG(INFO) << fmt::format("{} env pages used in qureg", state->memPoolAvailQueue.size());
    state->memPoolNumPagesTotal = state->memPoolAvailQueue.size();
    state->memPoolAvailQueue.resize(state->memPoolAvailQueue.size() + 1);

    state->memPoolAvailHead = 0;
    state->memPoolAvailTail = state->memPoolNumPagesTotal;
}

QuESTEnvValueDistributed::~QuESTEnvValueDistributed() {
    int finalized;
    MPI_CHECK(MPI_Finalized(&finalized));
    if (!finalized) {
        MPI_CHECK(MPI_Finalize());
        LOG(INFO) << fmt::format("rank {} finalized\n", rankId);
    } else {
        LOG(ERROR) << "ERROR: Trying to close QuESTEnv multiple times. Ignoring";
    }
}

QuregStateDistributed::QuregStateDistributed(Qureg &qureg)
    : QuregState(qureg) {}

void QuregStateDistributed::statevec_initStateFromAmps(double *reals, double *imags) {
    sectionGlobalViewList.clear();
    sectionGlobalViewList.emplace_back(sectionIndex++);
    sectionGlobalViewList.back().targetMask = (1 << numQubitsPerTask) - 1;
    sectionGlobalViewList.back().initFunction = INIT_FROM_AMPS;

    sectionGlobalViewList.back().mappingCalc.l2v.resize(qureg.numQubitsInStateVec);
    sectionGlobalViewList.back().mappingCalc.v2l.resize(qureg.numQubitsInStateVec);
    for (int i = 0; i < qureg.numQubitsInStateVec; i++) {
        sectionGlobalViewList.back().mappingCalc.l2v[i] = i;
        sectionGlobalViewList.back().mappingCalc.v2l[i] = i;
    }

    for (u64 i = taskBeginOffset; i < taskBeginOffset + numTasksThisNode; i++) {
        for (int j = 0; j < numTasksTotal; j++) {
            u64 pageIndex = i * numTasksTotal + j;
            auto &pageTable = this->pageTable[0];
            if (pageTable[pageIndex] == nullptr) {
                pageTable[pageIndex] = mallocPage();
            }
            qComplex *data = pageTable[pageIndex];
            for (u64 offset = 0, base = pageIndex * numAmpsPerPage; offset < numAmpsPerPage; offset++) {
                data[offset] = {reals[base + offset], imags[base + offset]};
            }
        }
    }
}

void QuregStateDistributed::switchToNextState() {
    QuregState::switchToNextState();
    std::swap(pageTable[0], pageTable[1]);
}

void CUDART_CB freePageCallback(void *data) {
    auto **ptr = (qComplex **) (((void **) data)[0]);
    auto *state = (QuregStateDistributed *) (((void **) data)[1]);
    state->freePage(*ptr);
    *ptr = nullptr;
    free(data);
}

void QuregStateDistributed::freePage(qComplex *ptr) {
    size_t curWriteIdx;
    while (true) {
        curWriteIdx = memPoolAvailTail;
        if (memPoolAvailTail.compare_exchange_strong(curWriteIdx, curWriteIdx + 1)) {
            memPoolAvailQueue[curWriteIdx % memPoolAvailQueue.size()] = ptr;
            break;
        }
    }
}

qComplex *QuregStateDistributed::mallocPage() {
    while (true) {
        size_t curMaxReadIdx = memPoolAvailTail;
        size_t curReadIdx = memPoolAvailHead;
        if (curReadIdx % memPoolAvailQueue.size() == curMaxReadIdx % memPoolAvailQueue.size()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_TIME));
            return nullptr;
        }
        if (memPoolAvailHead.compare_exchange_strong(curReadIdx, curReadIdx + 1)) {
            return memPoolAvailQueue[curReadIdx % memPoolAvailQueue.size()];
        }
    }
}

qComplex *QuregStateDistributed::mallocPageBusyWaiting() {
    qComplex *ptr;
    do {
        ptr = mallocPage();
    } while (ptr == nullptr);
    return ptr;
}

void QuESTEnvValueDistributed::calculateAllSections(Qureg &qureg) {
    double calcTime[numDevices];
    double reorderTime[numDevices];
    for (int i = 0; i < numDevices; i++) {
        calcTime[i] = reorderTime[i] = 0;
    }

    auto state = dynamic_cast<QuregStateDistributed *>(qureg.state);
    if (state->sectionGlobalViewList.empty()) {
        return;
    }
    qureg.state->probsDirty = true;
    std::atomic<int> num_prep(0);
    std::atomic<int> num_calc(0);
    std::atomic<int> num_swap(0);
    std::condition_variable cv_prep;
    std::condition_variable cv_calc;
    std::condition_variable cv_swap;
    std::mutex mu;
    std::atomic<int> common_localTaskId(0);

    auto task_func = [&](int deviceId) {
        CUDA_CHECK(cudaSetDevice(deviceId));
        qComplex *bufferPages[2];
        bufferPages[0] = state->mallocPageBusyWaiting();
        bufferPages[1] = state->mallocPageBusyWaiting();

        cudaStream_t streamPrefetch;
        cudaStream_t streamFilling;
        cudaStream_t streamLocalCopy;
        cudaStreamCreate(&streamPrefetch);
        cudaStreamCreate(&streamFilling);
        cudaStreamCreate(&streamLocalCopy);

        while (!state->sectionGlobalViewList.empty()) {
            {
                std::unique_lock<std::mutex> lk(mu);
                cv_prep.wait(lk, [&]() { return num_prep >= 1; });
                num_prep--;
            }
            cv_prep.notify_one();
            int nextLocalTaskId = common_localTaskId++;

            const auto startFetchingNextPage = [&qureg, deviceId, &nextLocalTaskId, &streamPrefetch, state](int pageId) {
                if (nextLocalTaskId >= state->numTasksThisNode) {
                    return;
                }
                const u64 nextTaskId = nextLocalTaskId + state->taskBeginOffset;
                // copy data from CPU to GPU
                if (state->sectionGlobalViewList.front().initFunction == INIT_FROM_AMPS) {
                    const u64 pageIndex = nextTaskId * state->numTasksTotal + pageId;
                    auto &pageTable = state->pageTable[0];
                    qComplex *const src = pageTable[pageIndex];
                    qComplex *const dst = state->deviceStateVecs[deviceId] + pageId * state->numAmpsPerPage;
#ifdef OPTIMIZE_ZERO
                    if (src == nullptr) {
                        CUDA_CHECK(cudaMemsetAsync(dst, 0, state->numAmpsPerPage * sizeof(qComplex)));
                        return;
                    }
#endif
                    CUDA_CHECK(cudaMemcpyAsync(dst, src, state->numAmpsPerPage * sizeof(qComplex), cudaMemcpyHostToDevice, streamPrefetch));

                    void **ptrArr = (void **) malloc(sizeof(void *) * 2); // will be freed inside callback
                    ptrArr[0] = &pageTable[pageIndex];
                    ptrArr[1] = state;
                    cudaLaunchHostFunc(streamPrefetch, freePageCallback, ptrArr);
                }
            };

            // fetch pages for the first task
            for (int pageId = 0; pageId < state->numTasksTotal; pageId++) {
                startFetchingNextPage(pageId);
            }

            while (nextLocalTaskId < state->numTasksThisNode) {
                const u64 taskId = nextLocalTaskId + state->taskBeginOffset;

                // wait for cudaMemcpyAsync in readNextPage
                CUDA_CHECK(cudaDeviceSynchronize());

                int header_buf[HEADER_BUF_SIZE];
                header_buf[HEADER_SECTION_ID] = state->sectionIndex;
                header_buf[HEADER_TASK_ID] = (int) taskId;

                if (qureg.state->sectionGlobalViewList.front().localViews[taskId].initFunction == INIT_ALL_ZERO) {
                    LOG(INFO) << fmt::format("[{}:{}] Get an all zero task {:3}!", rankId, deviceId, taskId);
                    header_buf[HEADER_ALL_ZERO] = 1;
                } else {
                    header_buf[HEADER_ALL_ZERO] = 0;
                }

                if (qureg.state->sectionGlobalViewList.front().localViews[taskId].initFunction != INIT_ALL_ZERO) {
                    const auto calc_start_time = getWallTime();
                    // calculate
                    calculateSectionGpu(qureg, deviceId, (int) taskId);
                    CUDA_CHECK(cudaDeviceSynchronize());
                    const auto calc_stop_time = getWallTime();
                    const auto calc_time = calc_stop_time - calc_start_time;
                    calcTime[deviceId] += calc_time;

                    const auto reorder_start_time = getWallTime();
                    // reorder
                    reorderDataGpu(qureg, deviceId, (int) taskId);
                    CUDA_CHECK(cudaDeviceSynchronize());
                    const auto reorder_stop_time = getWallTime();
                    const auto reorder_time = reorder_stop_time - reorder_start_time;
                    reorderTime[deviceId] += reorder_time;

                    LOG(INFO) << fmt::format("[{}:{}] Time calculating/reordering task {:3d}: {:12.6f}/{:12.6f}", rankId, deviceId, taskId, calc_time, reorder_time);
                }

                // prefetch next task
                nextLocalTaskId = common_localTaskId++;

                std::queue<std::pair<int, int>> remotePages;
                std::queue<int> localPages;
                std::queue<int> localPagesInStream;
                for (int pageId = 0; pageId < state->numTasksTotal; pageId++) {
                    u64 pageIndex = taskId * state->numTasksTotal + pageId;
                    pageIndex = translateIndex(pageIndex << state->numQubitsPerPage, qureg.state->mapping[1].v2l, qureg.numQubitsInStateVec);
                    pageIndex = translateIndex(pageIndex, qureg.state->mapping[2].l2v, qureg.numQubitsInStateVec) >> state->numQubitsPerPage;
                    int dstRank = getCpuRank(pageIndex / state->numTasksTotal, qureg);
                    if (dstRank == rankId) {
                        localPages.push(pageId);
                    } else {
                        remotePages.push({pageId, dstRank});
                    }
                }

                MPI_Request requestsPage[2]{MPI_REQUEST_NULL, MPI_REQUEST_NULL}; // The message is split into 2 halves,
                                                                                 // which makes the communication faster.
                                                                                 // But we do not know why?
                MPI_Request requestHeader = MPI_REQUEST_NULL;
                int bufferPtr = 0;
                int bufferDstRank[2];
                int bufferPageIndex[2];
                auto &pageTable = state->pageTable[1];

                auto fill = [&] {
                    const auto pageId = remotePages.front().first;
                    const auto dstRank = remotePages.front().second;
                    remotePages.pop();

                    if (!header_buf[HEADER_ALL_ZERO]) {
                        qComplex *const src = state->deviceStateVecs[deviceId] + pageId * state->numAmpsPerPage;
                        qComplex *const dst = bufferPages[bufferPtr];
                        CUDA_CHECK(cudaMemcpyAsync(dst, src, state->numAmpsPerPage * sizeof(qComplex), cudaMemcpyDeviceToHost, streamFilling));
                        CUDA_CHECK(cudaStreamSynchronize(streamFilling));
                    }

                    startFetchingNextPage(pageId);

                    u64 pageIndex = taskId * state->numTasksTotal + pageId;
                    pageIndex = translateIndex(pageIndex << state->numQubitsPerPage, qureg.state->mapping[1].v2l, qureg.numQubitsInStateVec);
                    pageIndex = translateIndex(pageIndex, qureg.state->mapping[2].l2v, qureg.numQubitsInStateVec) >> state->numQubitsPerPage;
                    header_buf[HEADER_PAGE_INDEX] = (int) pageIndex;
                    MPI_CHECK(MPI_Issend(header_buf, HEADER_BUF_SIZE, MPI_INT, dstRank, SET_PAGE, MPI_COMM_WORLD, &requestHeader));
                    bufferPageIndex[bufferPtr] = (int) pageIndex;
                    bufferDstRank[bufferPtr] = dstRank;
                };
                auto startSending = [&] {
                    if (!header_buf[HEADER_ALL_ZERO]) {
                        qComplex *const data = bufferPages[bufferPtr];
                        MPI_CHECK(MPI_Isend(data, state->numAmpsPerPage, MPI_DOUBLE, bufferDstRank[bufferPtr],
                                            GPU_DATA_TO_CPU + (bufferPageIndex[bufferPtr] << 1 | 0), MPI_COMM_WORLD, &requestsPage[0]));
                        MPI_CHECK(MPI_Isend(data + (state->numAmpsPerPage >> 1), state->numAmpsPerPage, MPI_DOUBLE, bufferDstRank[bufferPtr],
                                            GPU_DATA_TO_CPU + (bufferPageIndex[bufferPtr] << 1 | 1), MPI_COMM_WORLD, &requestsPage[1]));
                    }
                };
                auto localTryCopy = [&] {
                    if (cudaStreamQuery(streamLocalCopy) != cudaSuccess)
                        return;
                    while (!localPagesInStream.empty()) {
                        startFetchingNextPage(localPagesInStream.front());
                        localPagesInStream.pop();
                    }

                    if (localPages.empty())
                        return;
                    const auto pageId = localPages.front();
                    localPages.pop();

                    u64 pageIndex = taskId * state->numTasksTotal + pageId;
                    pageIndex = translateIndex(pageIndex << state->numQubitsPerPage, qureg.state->mapping[1].v2l, qureg.numQubitsInStateVec);
                    pageIndex = translateIndex(pageIndex, qureg.state->mapping[2].l2v, qureg.numQubitsInStateVec) >> state->numQubitsPerPage;
#ifdef OPTIMIZE_ZERO
                    if (header_buf[HEADER_ALL_ZERO]) {
                        return;
                    }
#endif
                    pageTable[pageIndex] = state->mallocPage();
                    if (pageTable[pageIndex] == nullptr) {
                        return;
                    }
                    qComplex *const dst = pageTable[pageIndex];

                    if (!header_buf[HEADER_ALL_ZERO]) {
                        qComplex *const src = state->deviceStateVecs[deviceId] + (u64) pageId * state->numAmpsPerPage;
                        CUDA_CHECK(cudaMemcpyAsync(dst, src, (u64) state->numAmpsPerPage * sizeof(qComplex), cudaMemcpyDeviceToHost, streamLocalCopy));
                    } else {
                        memset(dst, 0, (u64) state->numAmpsPerPage * sizeof(qComplex));
                    }
                    localPagesInStream.push(pageId);
                };
                auto waitTillSent = [&] {
                    int flag1, flag2;
                    MPI_CHECK(MPI_Test(&requestHeader, &flag1, MPI_STATUS_IGNORE));
                    MPI_CHECK(MPI_Testall(2, requestsPage, &flag2, MPI_STATUS_IGNORE));
                    while (!flag1 || !flag2) {
                        localTryCopy();
                        std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_TIME));
                        MPI_CHECK(MPI_Test(&requestHeader, &flag1, MPI_STATUS_IGNORE));
                        MPI_CHECK(MPI_Testall(2, requestsPage, &flag2, MPI_STATUS_IGNORE));
                    }
                };

                if (!remotePages.empty()) {
                    while (!remotePages.empty()) {
                        fill();
                        waitTillSent();

                        startSending();
                        bufferPtr ^= 1;
                    }
                    waitTillSent();
                }

                while (!localPages.empty() || !localPagesInStream.empty()) {
                    localTryCopy();
                }
            }

            CUDA_CHECK(cudaDeviceSynchronize());

            if (++num_calc == numDevices) {
                cv_calc.notify_one();
            }

            {
                std::unique_lock<std::mutex> lk(mu);
                cv_swap.wait(lk, [&]() { return num_swap >= 1; });
                num_swap -= 1;
            }
            cv_swap.notify_one();
        }
        cudaStreamDestroy(streamPrefetch);
        cudaStreamDestroy(streamFilling);
        cudaStreamDestroy(streamLocalCopy);
        state->freePage(bufferPages[0]);
        state->freePage(bufferPages[1]);
    };
    std::vector<std::thread> threads(numDevices);
    for (int i = 0; i < numDevices; i++) {
        threads[i] = std::thread(task_func, i);
    }
    int headerBuf[HEADER_BUF_SIZE];
    MPI_Request headerRequest;
    MPI_CHECK(MPI_Recv_init(headerBuf, HEADER_BUF_SIZE, MPI_INT, MPI_ANY_SOURCE, SET_PAGE, MPI_COMM_WORLD, &headerRequest));

    while (!qureg.state->sectionGlobalViewList.empty()) {
        qureg.state->applyMapping();
        num_prep = numDevices;
        cv_prep.notify_one();
        {
            int numPageToReceive = 0;
            for (int taskIdOffset = 0; taskIdOffset < state->numTasksThisNode; taskIdOffset++) {
                u64 taskId = state->taskBeginOffset + taskIdOffset;
                for (int i = 0; i < state->numPagesPerTask; i++) {
                    const u64 dstPageIndex = taskId * state->numPagesPerTask + i;
                    const int dstRank = getCpuRank(dstPageIndex / state->numPagesPerTask, qureg);

                    const u64 srcPageIndex = translateIndex(
                                                     translateIndex(
                                                             dstPageIndex << state->numQubitsPerPage,
                                                             qureg.state->mapping[2].v2l, qureg.numQubitsInStateVec),
                                                     qureg.state->mapping[1].l2v, qureg.numQubitsInStateVec) >>
                                             state->numQubitsPerPage;
                    const int srcRank = getCpuRank(srcPageIndex / state->numPagesPerTask, qureg);

                    if (srcRank != rankId && dstRank == rankId) {
                        numPageToReceive++;
                    }
                }
            }

            for (int _ = 0; _ < numPageToReceive; _++) {
                qComplex *tmp = state->mallocPageBusyWaiting();
                MPI_CHECK(MPI_Start(&headerRequest));
                MPI_Status status;
                MPI_Wait(&headerRequest, &status);

                int pageIndex = headerBuf[HEADER_PAGE_INDEX];
#ifdef OPTIMIZE_ZERO
                if (headerBuf[HEADER_ALL_ZERO]) {
                    state->freePage(tmp);
                    continue;
                }
#endif
                int source = status.MPI_SOURCE;
                auto &pageTable = state->pageTable[1];
                pageTable[pageIndex] = tmp;
                qComplex *data = pageTable[pageIndex];

                if (!headerBuf[HEADER_ALL_ZERO]) {
                    MPI_CHECK(MPI_Recv(data, state->numAmpsPerPage, MPI_DOUBLE, source,
                                       GPU_DATA_TO_CPU + (pageIndex << 1 | 0), MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                    MPI_CHECK(MPI_Recv(data + (state->numAmpsPerPage >> 1), state->numAmpsPerPage, MPI_DOUBLE, source,
                                       GPU_DATA_TO_CPU + (pageIndex << 1 | 1), MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                } else {
                    memset(data, 0, state->numAmpsPerPage * sizeof(qComplex));
                }
            }
        }

        {
            std::unique_lock<std::mutex> lk(mu);
            cv_calc.wait(lk, [&]() { return num_calc == numDevices; });
            num_calc = 0;
        }
        MPI_Barrier(MPI_COMM_WORLD);

#ifdef STUB_PROFILE_ZEROS
        // STUB: profiling zeros
        u64 non_zeros = 0;
        u64 zeros = 0;
        auto &pageTable = dynamic_cast<QuregStateDistributed *>(qureg.state)->pageTable[1];
        for (auto &i : pageTable) {
            if (i) {
                qreal *real = i;
                qreal *imag = real + state->numAmpsPerPage;
                for (int j = 0; j < state->numAmpsPerPage; j++) {
                    if (real[j] != 0 || imag[j] != 0) {
                        non_zeros += 1;
                    } else {
                        zeros += 1;
                    }
                }
            }
        }
        LOG(WARNING) << fmt::format("Non-zeros after section:      {:11d} / {:11d}", non_zeros, qureg.numAmpsTotal);
        LOG(WARNING) << fmt::format("Computed zeros after section: {:11d} / {:11d}", zeros, qureg.numAmpsTotal);
#endif // STUB_PROFILE_ZEROS

        qureg.state->switchToNextState();
        common_localTaskId = 0;
        num_swap = numDevices;
        cv_swap.notify_one();
    }
    for (int i = 0; i < numDevices; i++) {
        threads[i].join();
    }

    for (int i = 0; i < numDevices; i++) {
        LOG(INFO) << fmt::format("Total calcTime [device{}] : {:10.6f}", i, calcTime[i]);
    }
    for (int i = 0; i < numDevices; i++) {
        LOG(INFO) << fmt::format("Total reorderTime [device{}] : {:10.6f}", i, reorderTime[i]);
    }
}

Complex QuESTEnvValueDistributed::statevec_getAmp(Qureg &qureg, long long int index) {
    auto *state = dynamic_cast<QuregStateDistributed *>(qureg.state);
    Complex amp{0., 0.};
    auto virtualIndex = translateIndex(index, qureg.state->mapping[0].l2v, qureg.numQubitsInStateVec);
    int root = getCpuRank((int) (virtualIndex >> state->numQubitsPerTask), qureg);
    auto env = dynamic_cast<QuESTEnvValueDistributed *>(qureg.env.value);
    if (env->rankId == root) {
        int pno = (int) (virtualIndex >> state->numQubitsPerPage);
        u32 offset = virtualIndex & (state->numAmpsPerPage - 1);
        qComplex *ptr = state->pageTable[0][pno] + offset;
        amp = {ptr->x, ptr->y};
    }
    MPI_CHECK(MPI_Bcast(&amp, 2, MPI_DOUBLE, root, MPI_COMM_WORLD));
    return amp;
}

qreal QuESTEnvValueDistributed::statevec_calcProbOfOutcome(Qureg &qureg, const int measureQubit, int outcome) {
    if (qureg.state->probsDirty) {
        std::atomic<int> common_taskId(0);
        auto state = dynamic_cast<QuregStateDistributed *>(qureg.state);
        std::vector<qreal *> &probsOfZero = state->probsOfZero;
        auto task_func = [&](int deviceId) {
            CUDA_CHECK(cudaSetDevice(deviceId));
            while (true) {
                // get next task
                int localTaskId = common_taskId++;
                if (localTaskId >= state->numTasksThisNode) {
                    return;
                }
                u64 taskId = localTaskId + state->taskBeginOffset;

                // receive data
                probsOfZero[localTaskId] = new qreal[qureg.numQubitsInStateVec];
                qComplex *stateVec = state->deviceStateVecs[deviceId];
                for (int i = 0; i < state->numTasksTotal; i++) {
                    u64 pageIndex = taskId * state->numTasksTotal + i;
                    auto &pageTable = state->pageTable[0];
                    qComplex *src = pageTable[pageIndex];
                    qComplex *dst = stateVec + i * state->numAmpsPerPage;
#ifdef OPTIMIZE_ZERO
                    if (src == nullptr) {
                        CUDA_CHECK(cudaMemsetAsync(dst, 0, state->numAmpsPerPage * sizeof(qComplex)));
                        continue;
                    }
#endif
                    CUDA_CHECK(cudaMemcpyAsync(dst, src, state->numAmpsPerPage * sizeof(qComplex), cudaMemcpyDefault));
                }
                CUDA_CHECK(cudaDeviceSynchronize());

                // observe
                statevec_calculateProbabilityOfZeroLocal(probsOfZero[localTaskId], deviceBuffer[deviceId], stateVec, (int) state->numQubitsPerTask);
            }
        };
        std::vector<std::thread> threads(numDevices);
        for (int d = 0; d < numDevices; d++) {
            threads[d] = std::thread(task_func, d);
        }
        for (int d = 0; d < numDevices; d++) {
            threads[d].join();
        }

        for (u32 i = 0; i < state->numQubitsPerTask; i++) {
            qureg.state->probsOfZero[i] = 0;
            for (int localTaskId = 0; localTaskId < state->numTasksThisNode; localTaskId++) {
                qureg.state->probsOfZero[i] += probsOfZero[localTaskId][i];
            }
        }
        for (u32 i = state->numQubitsPerTask; i < qureg.numQubitsInStateVec; i++) {
            qureg.state->probsOfZero[i] = 0;
            u32 taskMask = 1 << (i - state->numQubitsPerTask);
            for (int localTaskId = 0; localTaskId < state->numTasksThisNode; localTaskId++) {
                u64 taskId = localTaskId + state->taskBeginOffset;
                if ((taskId & taskMask) == 0x0) {
                    qureg.state->probsOfZero[i] += probsOfZero[localTaskId][state->numQubitsPerTask];
                }
            }
        }
        for (int localTaskId = 0; localTaskId < state->numTasksThisNode; localTaskId++) {
            delete[] probsOfZero[localTaskId];
        }

        MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, qureg.state->probsOfZero, qureg.numQubitsInStateVec, MPI_QuEST_REAL,
                                MPI_SUM, MPI_COMM_WORLD));

        qureg.state->probsDirty = false;
    }
    qreal outcomeProb = qureg.state->probsOfZero[qureg.state->mapping[0].l2v[measureQubit]];
    if (outcome == 1)
        outcomeProb = 1.0 - outcomeProb;
    return outcomeProb;
}

void QuESTEnvValueDistributed::statevec_destroyQureg(Qureg &qureg) {
    auto state = dynamic_cast<QuregStateDistributed *>(qureg.state);
    for (const auto &pageBase : state->memPoolAllocatedEnvPage) {
        freeEnvPage(state->numAmpsPerPage, pageBase);
    }

    delete qureg.state;
    for (int i = 0; i < numDevices; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaFree(state->deviceStateVecs[i]));
    }
}
