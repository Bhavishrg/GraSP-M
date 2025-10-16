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

## Network Configuration and Socket Buffers

### Socket Buffer Sizes
The benchmarks automatically configure socket buffer sizes to prevent deadlocks with large data transfers. The default is 128 MB per socket buffer, which is sufficient for most use cases.

**Memory Usage Estimates:**
- 2 parties: ~512 MB per process in socket buffers
- 5 parties: ~2 GB per process in socket buffers
- 10 parties: ~4.5 GB per process in socket buffers

For systems with limited memory (< 32 GB), you may want to reduce buffer sizes in the source code. For systems with abundant memory (> 64 GB), larger buffers can improve performance.

### System-Level Buffer Limits
If you encounter socket buffer limitation warnings, you may need to increase system limits:

**Check current limits:**
```sh
cat /proc/sys/net/core/rmem_max  # Receive buffer max
cat /proc/sys/net/core/wmem_max  # Send buffer max
```

**Increase limits (requires root on host machine):**
```sh
# Temporary (until reboot)
sudo sysctl -w net.core.rmem_max=134217728  # 128 MB
sudo sysctl -w net.core.wmem_max=134217728  # 128 MB

# Permanent
echo "net.core.rmem_max = 134217728" | sudo tee -a /etc/sysctl.conf
echo "net.core.wmem_max = 134217728" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

**For Docker containers:**
```sh
# Run Docker with increased buffer limits
docker run -it \
  --sysctl net.core.rmem_max=134217728 \
  --sysctl net.core.wmem_max=134217728 \
  -v $PWD:/code graphdb
```

## Usage
A short description of the compiled programs is given below.
All of them provide detailed usage description on using the `--help` option.

### Available Benchmark Executables

**Circuit Operation Benchmarks:**
- `benchmarks/add`: Benchmark addition and multiplication gates
- `benchmarks/mult`: Benchmark multiplication gate performance
- `benchmarks/equality`: Benchmark equality testing operations
- `benchmarks/reconstruction`: Benchmark secret reconstruction operations

**Graph Protocol Benchmarks:**
- `benchmarks/shuffle`: Benchmark shuffle gate operations
- `benchmarks/compaction`: Benchmark compaction operations (single-threaded)
- `benchmarks/compaction_parallel`: Benchmark compaction operations (parallel)
- `benchmarks/groupindex`: Benchmark group-wise indexing (single-threaded)
- `benchmarks/groupindex_parallel`: Benchmark group-wise indexing (parallel)
- `benchmarks/grouppropagate`: Benchmark group-wise propagation (single-threaded)
- `benchmarks/grouppropagate_parallel`: Benchmark group-wise propagation (parallel)

**Network Benchmarks:**
- `benchmarks/vector_reconstruction`: Benchmark vector creation and reconstruction using direct network send/recv

### Running Benchmarks

Execute the following commands from the `build` directory created during compilation to run the programs:

```sh
# Example: Run compaction benchmark with 2 parties on localhost
# Run this command in separate terminals for each party (pid 1 and 2)

# Party 1
./benchmarks/compaction -p 1 -n 2 --localhost -l 0.5 -v 100000 --num-payloads 1

# Party 2 (in another terminal)
./benchmarks/compaction -p 2 -n 2 --localhost -l 0.5 -v 100000 --num-payloads 1
```

**Common Options:**
- `-p, --pid`: Party ID (required, starts from 1)
- `-n, --num-parties`: Number of parties (required)
- `-v, --vec-size`: Size of input vectors
- `-l, --latency`: Network latency in milliseconds
- `-t, --threads`: Number of threads (default: 6)
- `--localhost`: Run all parties on localhost
- `--net-config`: Path to JSON file with network configuration (alternative to --localhost)
- `-o, --output`: File to save benchmark results
- `--help`: Show detailed help message

**Network Configuration File Format (net_config.json):**
```json
[
  "192.168.1.1",
  "192.168.1.2",
  "192.168.1.3"
]
```
The array should contain IP addresses for party 0 (king party), party 1, party 2, etc.

### Example Benchmark Runs

```sh
# Vector reconstruction with 1 million elements
./benchmarks/vector_reconstruction -p 1 -n 2 --localhost -l 0.5 -v 1000000

# Parallel compaction with 100k elements and 5 payload vectors
./benchmarks/compaction_parallel -p 1 -n 2 --localhost -l 0.5 -v 100000 --num-payloads 5

# Group-wise propagation with output saved to file
./benchmarks/grouppropagate_parallel -p 1 -n 2 --localhost -l 0.5 -v 50000 -o results.json
```

### Legacy Benchmarks

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
