# 本机模型目录

每个已注册模型保留 PT、设备本地 TensorRT engine 和 `selection.json`，三者构成切换所需完整资产；不要将旧模型当作历史垃圾删除。

```text
models/<directory>/
├── model.pt
├── model-<width>x<height>-fp16.engine
└── selection.json
```

`selection.json` 的 `name` 是控制台稳定模型 ID；目录名可以不同。manifest 中路径相对项目根目录，包含 PT/engine SHA-256、`library=runtime/libdesktop_detector.so`、`input_size=[height,width]`、类别映射和精度说明。每台设备 engine 与哈希独立维护。

公开仓库只保留此说明，不包含 PT、engine 或现场 selection。请按根 README 的导出、构建、注册步骤生成本机资产。注册命令自动添加 README 更新日志；新增模型时也应同步维护 README 的模型表。
