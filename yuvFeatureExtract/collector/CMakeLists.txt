# collector —— YUV 特征采集子项目（可执行程序）
add_executable( yuvFeatureCollector
    src/main.cpp
)

target_link_libraries( yuvFeatureCollector PRIVATE
    MiniYaml
    YuvGrayLib
    SpatialFeatureLib
)

target_compile_features( yuvFeatureCollector PRIVATE cxx_std_17 )

# 把示例配置拷到构建输出目录，方便 `make run`
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/config/config.yaml
    ${CMAKE_CURRENT_BINARY_DIR}/config/config.yaml
    COPYONLY
)
