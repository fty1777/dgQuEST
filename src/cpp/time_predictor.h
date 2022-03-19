// Distributed under MIT licence. See https://github.com/fty1777/dgQuEST/blob/main/LICENSE for details

#ifndef DGQUEST_TIME_PREDICTOR_H
#define DGQUEST_TIME_PREDICTOR_H


#include <map>
#include <vector>

#include "QuEST_backend.h"
class TimePredictor {
public:
    TimePredictor();
    explicit TimePredictor(int samplingMaxNumQubits);

    ~TimePredictor();

    void enableSampling(int numQubits);

    void sample(int numQubits);
    void sampleAll(int numQubitsMin, int numQubitsMax);

    void save(const std::string& filename);
    void load(const std::string& filename);

    std::map<int, double> predictCircuit(const std::vector<Gate>& gates, int numQubits, int minK, int maxK);

private:
    std::map<int, std::map<GateType, float>> singleQubitGateTime;
    std::map<int, float> processTimeTotal;
    std::map<int, int> processTimeCount;
    std::map<int, float> processTime;
    std::map<int, float> bandwidthH2d;
    std::map<int, float> bandwidthD2h;
    std::map<int, float> bandwidthH2dOverlapped;
    std::map<int, float> bandwidthD2hOverlapped;
    std::map<int, float> bandwidthComm;

    const static size_t repeatTimes = 256;
    qComplex* vec = nullptr;
    qComplex* vecHost = nullptr;

    cudaEvent_t start, end;
    cudaEvent_t start2, end2;
    cudaStream_t streams[2];

    int numRanks;
    int rankId;
    int numDevices;

    bool samplingEnabled = false;
    int samplingMaxNumQubit = 0;

    void sampleGateTime(int numQubits);
    double executeGateTimeSampling(const std::vector<Gate>& gates, int numQubits);
    void sampleCalc(int numQubits);
    void sampleIo(int numQubits);
    void sampleH2d(int numQubits);
    void sampleD2h(int numQubits);
    void sampleOverlapped(int numQubits);
    void sampleComm(int numQubits);

    double predictGateTime(const Gate& gate, int numQubits = -1);
    double predictFusedGateTime(const FusedGate& fusedGate, int numQubits = -1);
    double predictPartitionPlan(const std::vector<SectionGlobalView>& sectionGlobalViewList, int numQubits, int K);
    double predictSectionCalcTimePerTaskFromLocalView(const SectionLocalView& secLV, int numQubits = -1);
    double predictSectionCalcTimePerTask(const SectionGlobalView& secGV, int numQubits = -1);
};


#endif //DGQUEST_TIME_PREDICTOR_H
