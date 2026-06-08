# MttFeatureLib —— MTT 块级特征(②③④)纯函数库
add_library( MttFeatureLib STATIC
    src/MttFeature.cpp
)

target_include_directories( MttFeatureLib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# 依赖 SpatialFeatureLib 的 sf::Mat 类型定义
target_link_libraries( MttFeatureLib PUBLIC SpatialFeatureLib )

target_compile_features( MttFeatureLib PUBLIC cxx_std_17 )
