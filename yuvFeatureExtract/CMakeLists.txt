cmake_minimum_required( VERSION 3.13 )
project( yuvFeatureExtract CXX )

if( NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release )
endif()

set( CMAKE_CXX_STANDARD 17 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
# 导出 compile_commands.json，供 VSCode IntelliSense 使用正确的标准/包含路径
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

# 自建函数库（单独文件夹 libs/）
add_subdirectory( libs )
# 采集子项目（单独文件夹 collector/）
add_subdirectory( collector )
# 合并子项目（标签×特征 → Training.csv）
add_subdirectory( merger )
# MTT 标签编排（解码 binEncode202 → 采样 MTT 划分标签）
add_subdirectory( mttLabel )
