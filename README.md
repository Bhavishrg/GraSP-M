# GraphDB

This directory contains the implementation of the secure graph database protocol.
The protocol is implemented in C++17 and [CMake](https://cmake.org/) is used as the build system.

## External Dependencies
The following libraries need to be installed separately and should be available to the build system and compiler.

- [GMP](https://gmplib.org/)
- [NTL](https://www.shoup.net/ntl/) (11.0.0 or later)
- [Boost](https://www.boost.org/) (1.72.0 or later)
- [Nlohmann JSON](https://github.com/nlohmann/json)
- [EMP Tool](https://github.com/emp-toolkit/emp-tool)

### Docker
All required dependencies to compile and run the project are available through the docker image.
To build and run the docker image, execute the following commands from the root directory of the repository:

```sh
# Build the GraphDB Docker image.
#
# Building the Docker image requires at least 4GB RAM. This needs to be set 
# explicitly in case of Windows and MacOS.
docker build -t graphdb .

# Create and run a container.
#
# This should start the shell from within the container.
docker run -it -v $PWD:/code graphdb

# The following command changes the working directory to the one containing the 
# source code and should be run on the shell started using the previous command.
cd /code
```

## Compilation
The project uses [CMake](https://cmake.org/) for building the source code. 
To compile, run the following commands from the root directory of the repository:

```sh
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..

# The two main targets are 'benchmarks' and 'tests' corresponding to
# binaries used to run benchmarks and unit tests respectively.
make benchmarks
```

## Usage
A short description of the compiled programs is given below.
All of them provide detailed usage description on using the `--help` option.

- `benchmarks/e2e_emgraph`: Benchmark the performance of the end to end emgraph protocol with initialization, preprocessing and online phases.
- `benchmarks/initialization_emgraph`: Benchmark the performance of the initialization phase of the emgraph protocol.
- `benchmarks/initialization_graphiti`: Benchmark the performance of the initialization phase of the graphiti protocol.
- `benchmarks/mpa_emgraph`: Benchmark the performance of the preprocessing and online phases of 1 round of message passing for emgraph.
- `benchmarks/mpa_graphiti`: Benchmark the performance of the preprocessing and online phases of 1 round of message passing for graphiti.
Execute the following commands from the `build` directory created during compilation to run the programs:
```sh
# Benchmark EmGraph MPA.
#
# The command below should be run on n+1 different terminals with $PID set to
# 0, 1, 2, upto n i.e., one instance corresponding to each party.
#
# The -v option can be used to vary the graph size. The -i option can be used to
# vary the number of iterations for message passing. The -l option will later on
# allow to vary the network latency.
#
# The program can be run on different machines by replacing the `--localhost`
# option with '--net-config <net_config.json>' where 'net_config.json' is a
# JSON file containing the IPs of the parties. A template is given in the
# repository root.
./benchmarks/e2e_emgraph -p $party --localhost -l 100.0 -v $vec_size -i 10 -n $players

# Run the graph_analysis script to automatically run the benchmarks
./../graph_analysis.sh
```
