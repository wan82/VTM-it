# mttFeature —— 给 MTT 标签块配空域特征，产出 TrainingMtt.csv
add_executable( mttFeatureBuilder
    src/main.cpp
)

target_link_libraries( mttFeatureBuilder PRIVATE
    MiniYaml
    YuvGrayLib
    SpatialFeatureLib
    MttFeatureLib
)

target_compile_features( mttFeatureBuilder PRIVATE cxx_std_17 )
