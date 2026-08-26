# AGENTS.md

## Project overview

SparkLM is a fundamentals-first project for building a decoder-only language model in PyTorch, progressing from a character tokenizer and tiny next-token model to Transformer training, evaluation, domain adaptation, instruction tuning, LoRA, distillation, quantisation, and local inference on NVIDIA DGX Spark.

Read [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) before making substantive changes. It is the source of truth for milestones, acceptance criteria, and the intended repository structure.

## Current stage

The project is beginning Phase 0: environment, scope, and experiment discipline. Prioritise:

- documenting the Python, PyTorch, CUDA, GPU, storage, and memory environment;
- establishing deterministic seeds and a one-command device smoke test;
- documenting dataset sources, licences, dates, preprocessing, and privacy decisions;
- creating the learning log and reproducible experiment conventions.

Do not jump to the 125M model or production abstractions before the smaller learning stages are correct, tested, documented, and understood.

## Core implementation rules

- Implement the learning-path tokenizer, batching, attention, Transformer block, model forward pass, generation loop, training loop, checkpoint metadata, and evaluation logic explicitly.
- Do not use `torch.nn.Transformer` or a pretrained causal-language-model implementation as the primary implementation during the early phases.
- PyTorch may provide tensors, autograd, optimisers, standard layers such as `Linear` and `LayerNorm`, device management, data loading, and later performance references.
- Keep tensor shapes and intermediate values inspectable. Prefer small, clear implementations over premature generalisation.
- Compare manual implementations with trusted PyTorch references only after the manual version has direct correctness tests.
- Preserve the character-tokenizer path as a reference baseline after introducing subword tokenisation.

## Testing and verification

Every meaningful change should include proportionate verification:

- run the relevant unit tests;
- add deterministic shape, numerical, or behavioural tests for new model components;
- test causal masking directly, including that future positions cannot affect earlier outputs;
- verify tokenizer round trips and input/target one-position alignment;
- verify tiny-batch overfitting and checkpoint save/load when those features are introduced;
- run formatting, linting, or type checks when the repository has configured tools.

Before increasing scale, confirm that the smaller implementation has tests, documentation, reproducible configuration, and a recorded experiment result.

## Data and artifact safety

- Never commit confidential, personal, tenant-identifying, or unlicensed data.
- Keep raw, interim, and processed datasets local or externally mounted as configured by `.gitignore`.
- Do not commit checkpoints, model weights, large generated outputs, or secrets.
- Every dataset used in an experiment needs a source, licence or usage note, date, privacy decision, and preprocessing description.
- Keep small experiment manifests, summaries, and learning notes under version control when they do not contain sensitive information.

## Experiments and reproducibility

Record the question, hypothesis, configuration, dataset version, code version, seed, device, result, and conclusion for meaningful experiments. Keep failed or unstable runs when they provide useful evidence.

Model comparisons should use the same evaluation data, prompt suite, and clearly documented settings. Report more than loss when relevant: perplexity, parameter count, memory, throughput, latency, model size, and qualitative failure modes.

## Coding-agent behaviour

- Keep answers concise and direct; avoid fluff unless the user asks for depth.
- Inspect the existing files and implementation plan before editing.
- Make focused changes that match the current milestone; do not scaffold unrelated phases without a clear request.
- Explain assumptions when requirements are ambiguous, especially around datasets, model architecture, and experiment configuration.
- Do not silently replace a manual learning implementation with a library abstraction.
- Prefer adding a small test or learning note when a change introduces a new concept or invariant.
- Keep user changes intact and avoid destructive repository commands.
- After editing, report changed files and the verification performed.

## Useful repository references

- [README.md](README.md) — project orientation and starting point.
- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — milestones and acceptance criteria.
- [essentials/README.md](essentials/README.md) — expected Phase 0 documentation.
