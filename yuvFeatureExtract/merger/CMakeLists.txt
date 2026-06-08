# merger —— 把标签与特征合并成 Training.csv（标准库，无第三方依赖）
add_executable( yuvLabelMerger src/main.cpp )
target_compile_features( yuvLabelMerger PRIVATE cxx_std_17 )
