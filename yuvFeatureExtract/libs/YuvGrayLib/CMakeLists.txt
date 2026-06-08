# YuvGrayLib —— 灰度图像矩阵自建函数库（读YUV/位深转换/CTU分割/无用帧检测）
add_library( YuvGrayLib STATIC
    src/GrayFrame.cpp
    src/YuvReader.cpp
    src/CtuSplitter.cpp
    src/FrameFilter.cpp
)

target_include_directories( YuvGrayLib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# 复用 SpatialFeatureLib 的 Mat 类型（sf::Mat）
target_link_libraries( YuvGrayLib PUBLIC SpatialFeatureLib )

target_compile_features( YuvGrayLib PUBLIC cxx_std_17 )
