# mttLabel —— MTT 标签编排器（可执行程序）
# 驱动 VTM DecoderApp 解码 binEncode202，按采样导出 MTT 划分标签 CSV。
add_executable( mttLabelBuilder
    src/main.cpp
)

target_link_libraries( mttLabelBuilder PRIVATE
    MiniYaml
    MttSampleLib
)

target_compile_features( mttLabelBuilder PRIVATE cxx_std_17 )

# 把示例配置拷到构建输出目录，方便 `make mtt-label`
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/config/mtt.yaml
    ${CMAKE_CURRENT_BINARY_DIR}/config/mtt.yaml
    COPYONLY
)
