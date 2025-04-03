# Find OpenSSL
find_package(OpenSSL REQUIRED)

# Include directories for headers
include_directories(${OPENSSL_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Define source files
file(GLOB BLOCKCHAIN_SOURCES "src/*.cpp" "src/sha3.cu")
file(GLOB BLOCKCHAIN_HEADERS "src/*.h" "src/*.hpp")

# Check if we found any source files
if("${BLOCKCHAIN_SOURCES}" STREQUAL "")
    message(FATAL_ERROR "No source files found in ${CMAKE_CURRENT_SOURCE_DIR}/src. Please check that files exist.")
endif()

# List the source files we found
message(STATUS "Source files found:")
foreach(SOURCE ${BLOCKCHAIN_SOURCES})
    message(STATUS "  ${SOURCE}")
endforeach()

# Create a library from the blockchain implementation
add_library(blockchain STATIC ${BLOCKCHAIN_SOURCES} ${BLOCKCHAIN_HEADERS})
set_target_properties(blockchain PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link with OpenSSL and CUDA runtime
target_link_libraries(blockchain PRIVATE ${OPENSSL_LIBRARIES} cuda)

# Find main file
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
    set(MAIN_FILE "src/main.cpp")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/Blockchain.cpp")
    set(MAIN_FILE "src/Blockchain.cpp")
else()
    foreach(SOURCE ${BLOCKCHAIN_SOURCES})
        message(STATUS "Checking if ${SOURCE} could be a main file")
    endforeach()
    list(GET BLOCKCHAIN_SOURCES 0 MAIN_FILE)
endif()

# Create executable
message(STATUS "Using main file: ${MAIN_FILE}")
add_executable(BlockchainDemo ${MAIN_FILE})
target_link_libraries(BlockchainDemo PRIVATE blockchain)