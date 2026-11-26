# Maintenance rules

- Keep functions, calls and conditions horizontally compact; avoid one argument per line.
- Every source, configuration-template, deployment or model-catalog change must add a brief dated item to the README 更新日志 in the same commit.
- Never commit live camera/Redis configuration, credentials, private keys, recordings, Arial font files or device binaries.
- Preserve model IDs, class order (person, car, cyclist, truck), wire IDs (0, 2, 1, 7), current calibration and model switching unless explicitly requested.
- Use the model's validated input height/width and verify actual TensorRT bindings. Do not silently change input shape or algorithms during cleanup.
- Run CPU tests and the changelog guard before publishing; Jetson deployment also needs native inference, all assigned camera outputs, previews and Redis freshness checks.
