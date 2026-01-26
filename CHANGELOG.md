# Changelog

All notable changes to ASPIRE will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-01-22

### Added
- `aspire doctor` command for environment diagnostics
- `--version` / `-V` flag to CLI
- Helpful error messages for missing API keys
- SECURITY.md for vulnerability reporting
- Dependabot configuration for automated dependency updates
- Comprehensive CONTRIBUTING.md guide
- Input validation for all teacher implementations
- Expanded test suite with 7 new test files
- CI and PyPI badges to README

### Changed
- CLI now shows help when run without arguments
- Improved error handling across the codebase

### Security
- Added ClaudeTeacherError and OpenAITeacherError for better error handling
- API keys now validated before use with actionable error messages

## [0.1.0] - 2026-01-22

### Added
- Initial release of ASPIRE
- Core training loop with student-critic-teacher architecture
- Teacher implementations:
  - ClaudeTeacher (Anthropic API)
  - OpenAITeacher (OpenAI API)
  - LocalTeacher (local models via transformers)
- Teacher personas:
  - Socratic - teaches through questions
  - Scientific - demands evidence and rigor
  - Creative - encourages novel thinking
  - Adversarial - stress-tests reasoning
  - Compassionate - balances challenge with encouragement
- CompositeTeacher for multi-teacher ensembles
- Critic architectures:
  - CriticHead - lightweight MLP on student hidden states
  - SeparateCritic - independent encoder model
  - SharedEncoderCritic - shared encoder with student
- Loss functions:
  - Critic score prediction loss
  - Critic reasoning alignment loss
  - Student reward loss
  - Contrastive loss (student vs teacher improved)
  - Trajectory improvement loss
- Dialogue generation system with caching
- CLI commands: `train`, `evaluate`, `dialogue`, `teachers`, `init`
- Pydantic-based configuration with YAML support
- Integration modules:
  - Stable Diffusion WebUI Forge (image generation)
  - Isaac Gym/Lab (robotics)
  - Code assistants (code review)
- Full Windows compatibility (RTX 5080/Blackwell support)
- Comprehensive test suite

### Technical Details
- Python 3.10+ required
- PyTorch 2.0+ with CUDA support
- 4-bit and 8-bit quantization support via bitsandbytes
- LoRA fine-tuning via PEFT
- Async teacher API calls

[Unreleased]: https://github.com/mcp-tool-shop/aspire-ai/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/mcp-tool-shop/aspire-ai/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/mcp-tool-shop/aspire-ai/releases/tag/v0.1.0
