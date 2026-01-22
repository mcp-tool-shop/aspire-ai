# ASPIRE Test Coverage Improvement Plan

**Last Updated**: 2026-01-22
**Current Coverage**: 44% (411 tests passing)
**Target Coverage**: 85%+ overall, 100% critical paths

---

## Instructions for Claude

**Goal:** Improve test coverage from 44% to 80%+

**Process:**
1. Pick 20 unchecked items from this list
2. Write the tests
3. Run `pytest tests/ --cov=aspire --cov=integrations -q` to verify
4. Mark completed items with `[x]`
5. Commit changes
6. Repeat until coverage target is met

---

## Windows Compatibility Requirements (CRITICAL)

All tests MUST follow these requirements:

```python
# At top of every test file
import os
os.environ["XFORMERS_DISABLED"] = "1"

# At bottom of every test file
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    pytest.main([__file__, "-v"])
```

### DataLoader Requirements (CRITICAL)
```python
# NEVER use num_workers > 0 on Windows
DataLoader(dataset, batch_size=4, num_workers=0)
```

---

## Priority 1: Core ASPIRE Package (High Impact)

### CLI Tests (`aspire/cli.py` - 0% coverage)
- [x] Test `aspire --version` outputs version
- [x] Test `aspire -V` outputs version
- [x] Test `aspire` with no args shows help
- [x] Test `aspire doctors` checks Python version
- [x] Test `aspire doctor` checks PyTorch
- [x] Test `aspire doctor` checks transformers
- [x] Test `aspire doctor` with ANTHROPIC_API_KEY set
- [x] Test `aspire doctor` with ANTHROPIC_API_KEY missing
- [x] Test `aspire doctor` with OPENAI_API_KEY set
- [x] Test `aspire doctor` disk space check
- [x] Test `aspire init` creates config file
- [x] Test `aspire init --output custom.yaml` uses custom path
- [x] Test `aspire teachers` lists all teachers
- [x] Test `aspire train` with demo prompts (mock trainer)
- [x] Test `aspire train --config` loads config file
- [x] Test `aspire evaluate` with mock checkpoint

### Trainer Tests (`aspire/trainer.py` - 17% coverage)
- [x] Test AspireTrainer initialization with default config
- [x] Test AspireTrainer._init_student with LoRA
- [x] Test AspireTrainer._init_student without LoRA
- [x] Test AspireTrainer._init_student with 4-bit quantization
- [x] Test AspireTrainer._init_student with 8-bit quantization
- [x] Test AspireTrainer._init_critic with "head" architecture
- [x] Test AspireTrainer._init_critic with "separate" architecture
- [x] Test AspireTrainer._init_critic with "shared_encoder" architecture
- [x] Test AspireTrainer._init_teacher with Claude
- [x] Test AspireTrainer._init_teacher with OpenAI
- [x] Test AspireTrainer._init_loss creates AspireLoss
- [x] Test AspireTrainer._init_optimizers creates optimizers
- [x] Test AspireTrainer._save_checkpoint creates files
- [x] Test AspireTrainer.load_checkpoint restores state
- [x] Test AspireDataset.__len__ returns correct length
- [x] Test AspireDataset.__getitem__ returns tokenized data

### Teacher Tests - Claude (`aspire/teachers/claude.py` - 20% coverage)
- [x] Test ClaudeTeacher initialization with API key
- [x] Test ClaudeTeacher raises ClaudeTeacherError without key
- [x] Test ClaudeTeacher.challenge returns TeacherChallenge (mock API)
- [x] Test ClaudeTeacher.challenge with different challenge types (mock API)
- [x] Test ClaudeTeacher.challenge builds history context
- [x] Test ClaudeTeacher.evaluate returns TeacherEvaluation (mock API)
- [x] Test ClaudeTeacher.evaluate with generate_improved=True (mock API)
- [x] Test ClaudeTeacher.evaluate with generate_improved=False (mock API)
- [x] Test ClaudeTeacher._get_challenge_description for all types
- [x] Test ClaudeTeacher handles JSON parse errors gracefully

### Teacher Tests - OpenAI (`aspire/teachers/openai.py` - 15% coverage)
- [x] Test OpenAITeacher initialization with API key
- [x] Test OpenAITeacher raises OpenAITeacherError without key
- [x] Test OpenAITeacher.challenge returns TeacherChallenge (mock API)
- [x] Test OpenAITeacher.challenge with different challenge types (mock API)
- [x] Test OpenAITeacher.evaluate returns TeacherEvaluation (mock API)
- [x] Test OpenAITeacher.evaluate with generate_improved=True (mock API)
- [x] Test OpenAITeacher handles JSON parse errors gracefully

### Teacher Tests - Local (`aspire/teachers/local.py` - 11% coverage)
- [x] Test LocalTeacher initialization with model path
- [x] Test LocalTeacher.challenge returns TeacherChallenge (mock model)
- [x] Test LocalTeacher.evaluate returns TeacherEvaluation (mock model)
- [x] Test LocalTeacher._generate_text produces output (mock model)
- [x] Test LocalTeacher with custom tokenizer

### Student Loss Tests (`aspire/losses/student.py` - 84% coverage)
- [x] Test RewardLoss with edge case scores (0, 10)
- [x] Test ContrastiveLoss with identical embeddings
- [x] Test TrajectoryLoss with declining scores
- [x] Test CoherenceLoss with uniform logits
- [x] Test KLDivergenceLoss with temperature scaling
- [x] Test StudentLoss weight configuration (combined loss tests)

---

## Priority 2: Integration Code (Medium Impact)

### Code Data (`integrations/code/data.py` - 22% coverage)
- [x] Test CodeSample creation with all fields
- [x] Test CodeSample default language
- [x] Test CodeReviewPair creation and serialization
- [x] Test GitHubRepoCollector initialization
- [x] Test GitHubRepoCollector.clone_repo (mock git)
- [x] Test GitHubRepoCollector.collect_files filters by language
- [x] Test GitHubRepoCollector.collect_files skips test directories
- [x] Test GitHubRepoCollector.collect_files respects line limits
- [x] Test generate_training_pairs creates pairs
- [x] Test save/load training data round-trip
- [x] Test create_balanced_dataset
- [x] Test CodeReviewDataset critic mode
- [x] Test CodeReviewDataset student mode
- [x] Test StreamingCodeDataset

### Code Trainer (`integrations/code/trainer.py` - 11% coverage)
- [ ] Test CodeCriticTrainer initialization
- [ ] Test CodeCriticTrainer.train_step computes loss
- [ ] Test CodeCriticTrainer.evaluate returns metrics
- [ ] Test CodeCriticTrainer.save_checkpoint
- [ ] Test CodeCriticTrainer.load_checkpoint

### Code Teacher (`integrations/code/code_teacher.py` - 63% coverage)
- [ ] Test CorrectnessChecker with valid async code
- [ ] Test StyleGuide with long lines
- [ ] Test SecurityAuditor with pickle.loads
- [ ] Test ArchitectureReviewer with deeply nested code
- [ ] Test PerformanceAnalyst with inefficient patterns
- [ ] Test CodeTeacher with "rotate" strategy
- [ ] Test CodeTeacher with "debate" strategy
- [ ] Test CodeTeacher.get_improvement_suggestions

---

## Priority 3: Optional Integrations (Lower Priority)

### Forge Integration (`integrations/forge/` - 0% coverage)
- [ ] Test VisionTeacher initialization (mock API)
- [ ] Test VisionTeacher.critique_image (mock API)
- [ ] Test ImageCritic.score_image
- [ ] Test ImageCritic with different architectures

### Isaac Integration (`integrations/isaac/` - 0% coverage)
- [ ] Test MotionTeacher initialization
- [ ] Test MotionTeacher.evaluate_trajectory
- [ ] Test TrajectoryCritic.score_motion
- [ ] Test IsaacWrapper environment setup (mock Isaac)

---

## Test Utilities to Create

### Fixtures Needed (add to `tests/conftest.py`)
- [x] `mock_anthropic_client` - Mock Anthropic API responses
- [x] `mock_openai_client` - Mock OpenAI API responses
- [x] `mock_student_model` - Mock HuggingFace model
- [x] `mock_tokenizer` - Mock tokenizer
- [x] `temp_config_file` - Temporary YAML config
- [x] `temp_checkpoint_dir` - Temporary checkpoint directory

### Test File Structure
```
tests/
├── conftest.py              # Shared fixtures
├── test_cli.py              # NEW - CLI tests
├── test_trainer_unit.py     # NEW - Trainer unit tests (mocked)
├── test_teachers_claude.py  # NEW - Claude teacher tests (mocked)
├── test_teachers_openai.py  # NEW - OpenAI teacher tests (mocked)
├── test_teachers_local.py   # NEW - Local teacher tests (mocked)
├── test_code_data.py        # NEW - Code data tests
├── test_code_trainer.py     # NEW - Code trainer tests
└── ... existing tests ...
```

---

## Progress Tracking

| Date | Tests Added | Coverage | Notes |
|------|-------------|----------|-------|
| 2026-01-22 | 411 | 44% | Initial baseline |
| 2026-01-22 | +60 | ~50%+ | Added Claude, OpenAI, Local teacher tests; trainer checkpoint tests; CLI tests already exist |
| 2026-01-22 | +38 | TBD | Added trainer edge cases (22 tests), dialogue edge cases (16 tests) |
| 2026-01-22 | +28 | TBD | Added Windows compatibility tests (28 tests) |
| 2026-01-22 | +40 | 659 tests | Added 8-bit quantization test, Code Data tests (38 tests), trainer tests |

---

## Commands

```bash
# Run all tests with coverage
pytest tests/ --cov=aspire --cov=integrations --cov-report=term -q

# Run specific test file
pytest tests/test_cli.py -v

# Run with HTML report
pytest tests/ --cov=aspire --cov=integrations --cov-report=html

# Skip slow tests
pytest tests/ -m "not slow" --cov=aspire -q
```

---

## Priority 4: Edge Cases & Stability (NEW SECTION)

### Loss Function Edge Cases (`tests/test_losses_edge_cases.py` - NEW)
- [ ] Test critic_score_loss with negative values
- [ ] Test critic_score_loss with NaN handling
- [ ] Test critic_score_loss with infinity handling
- [ ] Test critic_reasoning_loss with zero-norm embeddings
- [ ] Test critic_reasoning_loss with high dimensions (4096+)
- [ ] Test critic_contrastive_loss with identical samples
- [ ] Test reward_loss with scores outside [0, 10] range
- [ ] Test trajectory_loss with empty list
- [ ] Test trajectory_loss with single batch item
- [ ] Test coherence_loss with all padding tokens
- [ ] Test kl_loss with various temperature values
- [ ] Test student_loss with all weights = 0
- [ ] Test combined_loss gradient flow
- [ ] Test loss device consistency (CPU/CUDA)
- [ ] Test loss dtype consistency (float16/32/64)

### Critic Model Edge Cases (`tests/test_critic_edge_cases.py` - NEW)
- [ ] Test critic_head with empty hidden states
- [ ] Test critic_head with single token sequence
- [ ] Test critic_head with very long sequence (8192)
- [ ] Test critic_head with all-zeros attention mask
- [ ] Test critic_head mixed precision (FP16/BF16)
- [ ] Test separate_critic with empty text
- [ ] Test separate_critic with very long text
- [ ] Test shared_critic with student in train mode
- [ ] Test shared_critic adapter gradient flow
- [ ] Test multi_head_critic with single head

### Trainer Edge Cases (`tests/test_trainer_edge_cases.py` - ✅ IMPLEMENTED)
- [x] Test trainer with empty prompts list
- [x] Test trainer with single prompt
- [x] Test trainer with batch size > dataset size
- [x] Test trainer with zero epochs
- [x] Test trainer checkpoint resume
- [x] Test trainer with missing checkpoint files
- [x] Test trainer mixed precision BF16
- [x] Test trainer mixed precision FP16
- [x] Test trainer gradient accumulation edge cases
- [x] Test trainer learning rate scheduler warmup

### Dialogue Edge Cases (`tests/test_dialogue_edge_cases.py` - ✅ IMPLEMENTED)
- [x] Test dialogue with empty prompt
- [x] Test dialogue with very long prompt
- [x] Test dialogue with unicode content
- [x] Test dialogue cache corruption recovery
- [x] Test dialogue concurrent cache access
- [x] Test dialogue generator with zero turns
- [x] Test dialogue max length truncation (via formatter many turns)
- [x] Test dialogue formatter all formats
- [x] Test dialogue manager (cache tests)
- [x] Test dialogue history JSON serialization

### Windows Compatibility Tests (`tests/test_windows_compatibility.py` - ✅ IMPLEMENTED)
- [x] Test DataLoader num_workers=0 enforced
- [x] Test xformers disabled check
- [x] Test freeze_support in main
- [x] Test multiprocessing spawn method
- [x] Test path handling with backslashes
- [x] Test temp file cleanup
- [x] Test long path support (>260 chars)
- [x] Test CUDA memory management
- [x] Test async event loop on Windows
- [x] Test signal handling differences
- [x] Test model loading compatibility (bonus)
- [x] Test mixed precision autocast (bonus)

---

## Notes

- **Mock external APIs** - Don't make real API calls in tests
- **Mock model loading** - Use lightweight mocks instead of loading real models
- **Use `@pytest.mark.asyncio`** for async tests
- **Keep tests fast** - Target < 30 seconds for full suite
- **Windows compatible** - Use `freeze_support()` where needed
- **Follow existing patterns** - Check `test_critics.py` and `test_dialogue.py` for examples
