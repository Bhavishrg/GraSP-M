set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/benchmarks)

set(benchbin "")

macro(add_benchmark source_name)
    add_executable(${source_name} ${source_name}.cpp utils.cpp)
    target_link_libraries(${source_name} Boost::system Boost::program_options nlohmann_json::nlohmann_json GraphDB Threads::Threads NTL GMP EMPTool)
    list(APPEND benchbin ${source_name})
endmacro()

add_benchmark(initialization_graphiti)
add_benchmark(initialization_emgraph)
add_benchmark(mpa_graphiti)
add_benchmark(mpa_emgraph)
add_benchmark(e2e_emgraph)
add_benchmark(shuffle)
add_benchmark(test_primitives)
add_benchmark(mult)

add_custom_target(benchmarks)
add_dependencies(benchmarks ${benchbin})