# SparkLM

SparkLM is a fundamentals-first project to build a small decoder-only language model in PyTorch, starting with a character tokenizer and gradually progressing toward a locally runnable, domain-adapted model.

The project is designed for understanding rather than speed. Core components—tokenisation, causal self-attention, Transformer blocks, training, generation, and evaluation—will be implemented and tested explicitly before being replaced or accelerated with library implementations.

## Project path

```text
raw text
  → character tokenizer
  → tiny next-token model
  → manual causal self-attention
  → decoder-only Transformer
  → SparkLM-Tiny
  → SparkLM-125M
  → evaluation and error analysis
  → domain adaptation and instruction tuning
  → LoRA, distillation, and quantisation
  → local inference on DGX Spark
```

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the full milestones, experiments, acceptance criteria, and suggested repository structure.

## Starting point

Begin with Phase 0:

1. Record the machine, Python, PyTorch, CUDA, GPU, storage, and memory details in `essentials/`.
2. Create `LEARNING_LOG.md` and record the project rules, first observations, and data-provenance decisions.
3. Add a deterministic smoke test before implementing the tokenizer.
4. Keep experiments small, reproducible, and documented before increasing model size.

## Principles

- Understand the simple implementation before optimising it.
- Test tensor shapes, numerical behaviour, and causal masking directly.
- Keep failed experiments and explain what they taught you.
- Never commit confidential, personal, or unlicensed data.
- Use coding assistants as tutors, reviewers, and debugging partners; write the core learning-path implementation yourself.

## Planned layout

The repository will grow toward the structure described in the implementation plan, including `src/sparklm/`, `tests/`, `configs/`, `data/`, `experiments/`, `scripts/`, and `checkpoints/`.

## Status

Phase 0 — repository and experiment foundations.
