# dgQuEST

Source code is now available. Build/run guidelines and experimental setup descriptions are still incomplete and we are working in progress.
## Dependencies

- NVIDIA CUDA Toolkit (tested on CUDA 11.1, other versions are not tested)
- MPI implementation (tested on MPICH 4.0.1 and OpenMPI 4.1.2, other versions are not tested)

## Experimental Setup

The evaluation is done in the following environment
- MPI: MPICH 4.0.1
- OS: Ubuntu 20.04 LTS (Linux kernel: 5.4.0-77-generic)
- CPU: Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz
- GPU: NVIDIA Tesla V100 32GB
- InfiniBand NIC: ConnectX-3 40 Gb/sec (4X FDR10)

## How to Build

Clone the repo and build with CMake. Do not forget to clone submodules by adding `--recursive` option to `git clone`, or use `git submodule init` followed by `git submodule update`.

```bash
git clone https://github.com/fty1777/dgQuEST.git
cd dgQuEST
git submodule init
git submodule update

mkdir build
cd build
cmake .. -DCMAKE_CUDA_COMPILER=<path/to/nvcc> -DCMAKE_CUDA_ARCHITECTURES=<arch, e.g. 70 for V100>
make -j
```

To change the user program including the quantum circuit, append `-DUSER_PROGRAM=<path/to/user/program>` option to `cmake` command or modify the `CMakeLists.txt`. After executing `cmake`, you can also modify the generated `CMakeCache.txt` in build directory.

## How to Run

First, execute the sampler once to run the benchmarks for the time predictor:

```bash
cd build
./sampler
```

To change the sampled range of the nubmer of qubits, edit `sampler/main.cpp`. The sampled data are stored in file `predict_data.txt` in text format. The name should not be changed because the simulator will use the file with this fixed filename.

Before running the simulator, a config file is required to be created. The config file must contains 2 to 3 integers, which are the minK, maxK and designatedK, respectively. The minK and maxK determined the searching range of the analytical model in our paper. designatedK is not necessary. If designatedK is given, the simulator will run with this given K instead of the optimum one predicted by the model. If designatedK is not given, the simulator will use the K selected by the prediction model. After creating the config file, run the simulator:

```bash
echo <configs> > QuEST.conf # or `cp <DGQUEST_ROOT>/configs/QuEST.conf .` to create the file
./test
```
