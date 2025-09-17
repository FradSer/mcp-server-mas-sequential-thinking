# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2025-09-17

### Added
- Comprehensive TDD test coverage for refactoring and quality improvement
- Default settings and processing strategy enum for enhanced configuration
- Adaptive architecture with cost optimization capabilities
- Comprehensive test infrastructure and unit tests
- Magic number extraction to constants for better maintainability

### Changed
- **BREAKING**: Migration to Agno v2.0 with architectural updates (~10,000x faster agent creation, ~50x less memory usage)
- Upgraded Agno to version 2.0.5 with enhanced agent features
- Reorganized types module and cleaned duplicates for better structure
- Modernized codebase with enhanced type safety and annotations
- Adopted src layout for Python project structure following best practices
- Optimized code structure and performance across modules

### Fixed
- Resolved mypy type checking errors across all modules
- Comprehensive security and quality improvements
- Updated minimum Agno version to 2.0.5 for compatibility

### Documentation
- Updated CLAUDE.md with Agno v2.0 migration details and corrected commands
- Enhanced guidance for src layout and development requirements
- Improved test documentation and GitHub provider information

## [0.4.1] - 2025-08-06

### Fixed
- app_lifespan function signature for FastMCP compatibility

### Changed
- Restructured main.py with modular architecture for better maintainability

## [0.4.0] - 2025-08-06

### Added
- Support for Kimi K2 model via OpenRouter integration
- Enhanced model provider options and configuration flexibility

### Changed
- CHANGELOG.md following Keep a Changelog standards
- Moved changelog from README.md to dedicated CHANGELOG.md file

## [0.3.0] - 2025-08-01

### Added
- Support for Ollama FULL LOCAL (no API key needed, but requires Ollama installed and running locally)
- Local LLM inference capabilities through Ollama integration
- Enhanced model configuration options for local deployment
- MseeP.ai security assessment badge

### Changed
- Restored DeepSeek as default LLM provider
- Improved package naming and configuration
- Updated dependencies to support local inference
- Enhanced agent memory management (disabled for individual agents)

### Fixed
- Package naming issues in configuration
- Dependency conflicts resolved
- Merge conflicts between branches

## [0.2.3] - 2025-04-22

### Changed
- Updated version alignment in project configuration and lock file

## [0.2.2] - 2025-04-10

### Changed
- Default agent model ID for DeepSeek changed from `deepseek-reasoner` to `deepseek-chat`
- Improved model selection recommendations

## [0.2.1] - 2025-04-10

### Changed
- Model selection recommendations updated in documentation
- Enhanced guidance for coordinator vs specialist model selection

## [0.2.0] - 2025-04-06

### Added
- Major refactoring of sequential thinking team structure
- Enhanced coordination logic
- Improved JSON output format
- LLM configuration and model selection enhancements

### Changed
- Agent model IDs updated for better performance
- Project structure improvements

## [0.1.3] - 2025-04-06

### Changed
- Project entry point script from `main:main` to `main:run`
- Updated documentation for improved user guidance
- Cleaned up dependencies in lock file

## [0.1.0] - 2025-04-06

### Added
- Initial project structure and configuration files
- Multi-Agent System (MAS) architecture using Agno framework
- Sequential thinking tool with coordinated specialist agents
- Support for multiple LLM providers (DeepSeek, Groq, OpenRouter)
- Pydantic validation for robust data integrity
- Integration with external tools (Exa for research)
- Structured logging with file and console output
- Support for thought revisions and branching
- MCP server implementation with FastMCP
- Distributed intelligence across specialized agents

[Unreleased]: https://github.com/FradSer/mcp-server-mas-sequential-thinking/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/FradSer/mcp-server-mas-sequential-thinking/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/FradSer/mcp-server-mas-sequential-thinking/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/FradSer/mcp-server-mas-sequential-thinking/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/FradSer/mcp-server-mas-sequential-thinking/compare/v0.2.3...v0.3.0
[0.2.3]: https://github.com/FradSer/mcp-server-mas-sequential-thinking/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/FradSer/mcp-server-mas-sequential-thinking/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/FradSer/mcp-server-mas-sequential-thinking/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/FradSer/mcp-server-mas-sequential-thinking/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/FradSer/mcp-server-mas-sequential-thinking/compare/v0.1.0...v0.1.3
[0.1.0]: https://github.com/FradSer/mcp-server-mas-sequential-thinking/releases/tag/v0.1.0