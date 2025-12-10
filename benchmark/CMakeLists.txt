set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/benchmarks)

set(benchbin "")

macro(add_benchmark source_name)
    add_executable(${source_name} ${source_name}.cpp utils.cpp)
    target_link_libraries(${source_name} Boost::system Boost::program_options nlohmann_json::nlohmann_json GraphDB Threads::Threads NTL GMP EMPTool)
    list(APPEND benchbin ${source_name})
endmacro()

# TODO: Add benchmark executables here when malicious implementations are ready
# Example: add_benchmark(permAndSh)

add_benchmark(add)
add_benchmark(permsh)
add_benchmark(amortzdpns)

add_custom_target(benchmarks)
if(benchbin)
    add_dependencies(benchmarks ${benchbin})
endif()