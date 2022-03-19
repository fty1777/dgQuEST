// Distributed under MIT licence. See https://github.com/fty1777/dgQuEST/blob/main/LICENSE for details

#include <mpi.h>

#include "../src/cpp/time_predictor.h"

int main() {
    MPI_CHECK(MPI_Init(nullptr, nullptr));
    TimePredictor predictor(30);
    predictor.sampleAll(26, 30);
    predictor.save("predict_data.txt");
    MPI_CHECK(MPI_Finalize());
    return 0;
}