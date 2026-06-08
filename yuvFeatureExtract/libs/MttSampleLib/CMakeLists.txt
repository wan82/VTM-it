# MttSampleLib —— MTT 标签采样的纯函数库（无外部依赖）
add_library( MttSampleLib STATIC
    src/MttSample.cpp
)

target_include_directories( MttSampleLib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features( MttSampleLib PUBLIC cxx_std_17 )
