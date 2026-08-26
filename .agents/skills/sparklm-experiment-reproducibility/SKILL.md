---
name: sparklm-experiment-reproducibility
description: Design, run, compare, and document reproducible SparkLM experiments, training runs, evaluations, checkpoints, and ablations. Use when adding configs, training or evaluation commands, experiment manifests, loss curves, model comparisons, or run summaries.
---

# SparkLM Experiment Reproducibility

Read `AGENTS.md` and `IMPLEMENTATION_PLAN.md`. Treat every meaningful experiment as a recorded claim with enough information for another agent to reproduce and interpret it.

## Before a run

- State the question and hypothesis.
- Identify the exact code version, configuration, dataset version, split, preprocessing, seed, device, model, and tokenizer.
- Confirm that train, validation, test, and prompt-suite boundaries are defined before sampling.
- Estimate memory, sequence length, effective batch size, checkpoint size, and expected runtime.
- Check that private data and generated artifacts remain outside version control.

## During and after a run

- Log training and validation loss, perplexity where applicable, learning rate, tokens processed, throughput, memory, and checkpoint locations.
- Save a manifest with code version, data version, hyperparameters, seed, device, and metrics.
- Keep failed or unstable runs when they explain a system behaviour.
- Evaluate comparable models on the same held-out data and fixed prompt suite.
- Report parameter count, model size, latency or throughput, memory, and qualitative failure modes when relevant—not only loss.
- Write the conclusion, limitations, and next decision in the learning log or run summary.

## Comparison rules

- Change one important variable at a time for an ablation unless the experiment explicitly studies interactions.
- Compare effective batch size when gradient accumulation differs.
- Record whether dropout, mixed precision, optimised attention, or other performance features changed.
- Use repeated seeds or uncertainty context where practical.
- Never claim improvement from training loss alone; check held-out behaviour and qualitative samples.
