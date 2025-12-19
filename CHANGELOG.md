# Change Log

All notable changes to the "Lim Code" extension will be documented in this file.

## [1.0.4] - 2025-12-19

### Fixed
- 修复工具执行完成后点击终止按钮无法正常结束的问题（循环开始时检测取消信号后需发送 cancelled 消息给前端）

### Improved
- 优化搜索工具（find_files、search_in_files）忽略问题，添加默认排除模式配置

## [1.0.3] - 2025-12-19

### Added
- 添加了向 AI 发送诊断信息功能

### Fixed
- 修复上下文感知页面保存问题

### Note
- ⚠️ 旧版本使用者建议重置系统提示词以添加诊断信息功能

## [1.0.0] - 2025-12-19

### Added
- 🎉 首次发布
- AI 编程助手核心功能
- 多模态支持
- 对话历史管理
- 多语言支持（中文、英文、日文）
- MCP 服务器集成
- 文件操作工具
- 终端命令执行
- 图像处理功能