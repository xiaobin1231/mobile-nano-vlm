cmake_minimum_required(VERSION 3.10)
project(minimind-v-inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# ---- Detect platform ----
if(ANDROID)
    message(STATUS "Building for Android ${ANDROID_ABI}")
    set(PLATFORM_LIBS log)  # Android log library
else()
    message(STATUS "Building for Linux")
    set(PLATFORM_LIBS pthread dl)
endif()

# ---- MNN paths ----
set(MNN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/MNN")
if(ANDROID)
    set(MNN_BUILD "${MNN_ROOT}/build_android")
else()
    set(MNN_BUILD "${MNN_ROOT}/build")
endif()

set(MNN_INCLUDE_DIRS
    ${MNN_ROOT}/include
    ${MNN_ROOT}/express
    ${MNN_ROOT}/source
    ${MNN_ROOT}/tools
    ${MNN_ROOT}/3rd_party
    ${MNN_ROOT}/3rd_party/flatbuffers/include
    ${MNN_ROOT}/3rd_party/half
    ${MNN_ROOT}/schema/current
)

set(MNN_LLM_SRC "${MNN_ROOT}/transformers/llm/engine")

# ---- stb_image (JPEG/PNG loader) ----
# Download: wget https://raw.githubusercontent.com/nothings/stb/master/stb_image.h -O common/stb_image.h
set(STB_IMAGE_IMPLEMENTATION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/common/stb_image.cpp")

# ---- Sources ----
set(SOURCES
    main.cpp
    preprocess/image_preprocess.cpp
    vision/vision_encoder.cpp
    llm/llm_runtime.cpp
    pipeline/pipeline.cpp
    common/stb_image.cpp
    # MNN-LLM engine
    ${MNN_LLM_SRC}/src/llm.cpp
    ${MNN_LLM_SRC}/src/llmconfig.cpp
    ${MNN_LLM_SRC}/src/sampler.cpp
    ${MNN_LLM_SRC}/src/embedding.cpp
    ${MNN_LLM_SRC}/src/diskembedding.cpp
    ${MNN_LLM_SRC}/src/omni.cpp
    # Tokenizer
    ${MNN_LLM_SRC}/src/tokenizer/tokenizer.cpp
    ${MNN_LLM_SRC}/src/tokenizer/unicode.cpp
    ${MNN_LLM_SRC}/src/tokenizer/unicode_data.cpp
    # Generation
    ${MNN_LLM_SRC}/src/speculative_decoding/generate.cpp
    ${MNN_LLM_SRC}/src/speculative_decoding/lookahead.cpp
    ${MNN_LLM_SRC}/src/speculative_decoding/mtp.cpp
    ${MNN_LLM_SRC}/src/speculative_decoding/eagle.cpp
    ${MNN_LLM_SRC}/src/speculative_decoding/dflash.cpp
)

# ---- Executable ----
add_executable(minimind_cli ${SOURCES})

target_include_directories(minimind_cli PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${MNN_INCLUDE_DIRS}
    ${MNN_LLM_SRC}/include
    ${MNN_LLM_SRC}/src
)

target_link_libraries(minimind_cli
    ${MNN_BUILD}/libMNN.a
    ${PLATFORM_LIBS}
)

target_compile_definitions(minimind_cli PRIVATE
    MNN_USE_METAL=OFF
)

# Android-specific settings
if(ANDROID)
    target_compile_definitions(minimind_cli PRIVATE ANDROID)
    set_target_properties(minimind_cli PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
endif()
