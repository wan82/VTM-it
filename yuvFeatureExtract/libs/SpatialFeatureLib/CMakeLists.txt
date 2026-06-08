# SpatialFeatureLib —— 空域特征自建函数库
add_library( SpatialFeatureLib STATIC
    src/Moments.cpp
    src/Filters.cpp
    src/VarianceTests.cpp
    src/CtuFeatures.cpp
)

target_include_directories( SpatialFeatureLib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features( SpatialFeatureLib PUBLIC cxx_std_17 )
