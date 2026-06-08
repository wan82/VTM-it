# MiniYaml —— 极简 YAML 子集解析库（无外部依赖）
add_library( MiniYaml STATIC
    src/MiniYaml.cpp
)

target_include_directories( MiniYaml PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features( MiniYaml PUBLIC cxx_std_17 )
