# Project Rules

## Learning path

- Understand each component before moving on.
- Write the core code manually for tokenizer, batching, attention, Transformer block, generation, training, and evaluation.
- Use PyTorch for tensors, autograd, optimizers, standard layers, and device handling.
- Avoid `nn.Transformer` and pretrained causal language-model wrappers during the early phases.

## Working style

- Prefer small, readable implementations.
- Keep tensor shapes and intermediate values inspectable.
- Add a deterministic test for each important new concept.
- Record failed experiments when they teach something useful.

## Reproducibility

- Fix seeds for reproducible tests and experiments.
- Save code version, dataset version, configuration, seed, device, result, and conclusion.
- Keep checkpoints and generated artifacts out of git.

## Data safety

- Never commit sensitive or unlicensed data.
- Document dataset provenance and preprocessing.
- Keep raw and intermediate datasets local.

## Communication

- Prefer concise answers unless more detail is requested.
- Ask for clarification when assumptions matter.
