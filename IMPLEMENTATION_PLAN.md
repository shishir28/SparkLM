# SparkLM Implementation Plan

## Building a Small Language Model from First Principles on NVIDIA DGX Spark

SparkLM is a slow, fundamentals-first learning project: build a decoder-only Transformer yourself, understand every major component, and only then scale, adapt, compress, and serve it. The project is intentionally designed for learning rather than speed. A milestone may take several weeks; the quality of understanding matters more than the calendar.

The end-to-end path is:

```text
Raw text
  → character tokenizer
  → tiny next-token model
  → manual causal self-attention
  → decoder-only Transformer
  → SparkLM-Tiny
  → SparkLM-125M pretraining
  → evaluation and error analysis
  → property-management domain adaptation
  → instruction tuning
  → LoRA experiments
  → distillation
  → quantisation
  → local inference on DGX Spark
```

## 1. Project objectives

By the end of the project, you should be able to:

1. Explain next-token prediction, tokenisation, embeddings, positional information, causal masking, attention, residual connections, normalisation, logits, loss, optimisation, and generation in your own words.
2. Implement a small decoder-only Transformer in PyTorch without using `nn.Transformer` or a pretrained causal language-model implementation for the core learning path.
3. Train and debug progressively larger models, from a tiny character model to an approximately 125M-parameter model.
4. Build a reproducible data pipeline, checkpointing system, evaluation harness, and experiment log.
5. Adapt the base model to property-management material without confusing domain knowledge with instruction-following ability.
6. Compare full fine-tuning with LoRA and understand when each is appropriate.
7. Distil and quantise a model, measure the quality and performance trade-offs, and run it locally.
8. Read training curves and generated samples critically instead of treating a lower loss as the whole definition of success.

## 2. Working principles

### Fundamentals before optimisation

The first implementation should favour visibility over speed. Write the matrix operations explicitly, inspect tensor shapes, and keep intermediate values observable. Once the implementation is correct, compare it with optimised PyTorch operations.

### One source of truth for understanding

Do not hide the core model behind a framework abstraction during the learning stages. You may use PyTorch for tensors, automatic differentiation, optimisers, checkpointing, and device management, but the attention and Transformer logic should remain yours until the relevant milestone is complete.

### Tests before scale

A tiny model with strong tests is more valuable than a large model with uncertain correctness. Every major component should have a small deterministic test and at least one numerical or behavioural invariant.

### Experiments are part of the implementation

Record the question, hypothesis, configuration, result, and conclusion for each meaningful run. Keep failed experiments: they are evidence about the system and useful learning material.

### Use coding agents sparingly

The core implementation is deliberately manual. Use ChatGPT or another assistant as a tutor, reviewer, explainer, or debugging partner—not as the author of the attention, Transformer block, training loop, tokenizer integration, generation loop, or evaluation logic. If you ask for help, first describe what you believe is happening and show your own attempt.

### Scale only after understanding

Do not move to a larger model because the smaller model is inconvenient. Move when the current model is correct, tested, documented, and understood well enough that you can predict what scaling will change.

## 3. Suggested pace

The nominal schedule is 10–12 weeks, but it is intentionally open-ended. A reasonable rhythm is:

- 3–5 focused sessions per week
- 60–120 minutes per session
- one written learning note per milestone
- one reproducible experiment before moving on

If a phase takes two or three weeks, that is normal. The project is complete when the acceptance criteria are met, not when the calendar says it should be.

## 4. Phased milestones

### Phase 0 — Environment, scope, and experiment discipline

**Suggested time:** 2–4 sessions

**Objectives**

- Confirm the DGX Spark environment, CUDA/PyTorch installation, GPU visibility, available storage, and basic memory limits.
- Create the repository and establish reproducible configuration and logging conventions.
- Define the first corpus and a policy for licensing, privacy, and data provenance.

**Deliverables**

- Working Python environment and a documented setup procedure.
- A minimal device diagnostic script.
- Seed handling and a configuration file format.
- A short `LEARNING_LOG.md` describing the project rules and first observations.

**Acceptance criteria**

- A one-command smoke test reports Python, PyTorch, CUDA, device name, and tensor-device status.
- The same seed produces the same tiny test result on repeated runs.
- Every dataset used later has a source, licence/usage note, date, and preprocessing description.

### Phase 1 — Character tokenizer and data fundamentals

**Suggested time:** 1 week

**Objectives**

- Learn the complete text-to-tensor path using the simplest useful tokenizer.
- Build vocabulary creation, encoding, decoding, dataset splitting, and fixed-length batching.

**Implementation tasks**

- Normalise text conservatively and document every transformation.
- Create a sorted character vocabulary with explicit unknown handling if needed.
- Implement `encode(text)` and `decode(ids)`.
- Create non-overlapping train/validation/test splits before sampling sequences.
- Implement random batches of input sequences and one-position-shifted targets.

**Experiments**

- Verify `decode(encode(text)) == text` for representative inputs.
- Print a batch and manually confirm that targets are the next character.
- Compare several context lengths and inspect how much context the task requires.

**Acceptance criteria**

- Round-trip tokenisation works for the supported character set.
- Batch shapes and target alignment are tested.
- A trivial frequency baseline is implemented and measured.

### Phase 2 — Tiny next-token model

**Suggested time:** 1 week

**Objectives**

- Understand logits, cross-entropy, teacher forcing, optimisation, and sampling before introducing attention.

**Implementation tasks**

- Implement a token embedding lookup.
- Add a simple linear or small multilayer next-token predictor.
- Write the training loop, validation loop, gradient reset, optimiser step, and checkpoint save/load.
- Implement generation with temperature and a maximum length.

**Experiments**

- Overfit a very small batch; the loss should become extremely low.
- Compare random, greedy, and temperature-controlled generation.
- Plot training and validation loss.

**Acceptance criteria**

- The model can overfit a tiny batch deliberately.
- Checkpoint reload produces the same evaluation result.
- Generated text demonstrates that the model has learned the local training distribution, even if it is not coherent.

### Phase 3 — Manual causal self-attention

**Suggested time:** 1–2 weeks

**Objectives**

- Derive and implement scaled dot-product attention from first principles.

**Implementation tasks**

- Create query, key, and value projections.
- Compute `QKᵀ / sqrt(d)`.
- Apply a lower-triangular causal mask before softmax.
- Multiply attention weights by values.
- Add dropout only after the unregularised version works.
- Inspect attention weights and tensor shapes at every stage.

**Experiments**

- Confirm that a position cannot attend to future positions.
- Test a hand-built example where the expected attention pattern is obvious.
- Compare your implementation with `torch.nn.functional.scaled_dot_product_attention` on identical inputs.
- Check forward-output and gradient differences within an appropriate numerical tolerance.

**Acceptance criteria**

- The causal mask is tested directly, not inferred only from generated text.
- Manual attention agrees numerically with the reference implementation.
- You can explain why scaling by `sqrt(d)` stabilises softmax inputs.

### Phase 4 — Decoder-only Transformer block

**Suggested time:** 1 week

**Objectives**

- Assemble attention into a modern decoder block and understand information flow.

**Implementation tasks**

- Add multi-head attention by splitting and recombining head dimensions.
- Add a feed-forward network, typically an expansion followed by a non-linearity and projection.
- Add residual connections and layer normalisation.
- Compare pre-normalisation and post-normalisation conceptually; use one consistently and document the choice.
- Add positional embeddings or a clearly documented alternative.

**Acceptance criteria**

- Each subcomponent has shape tests.
- The block preserves batch and sequence dimensions.
- A short note explains the role of every residual path and normalisation layer.
- The block can replace the Phase 2 predictor without breaking the training loop.

### Phase 5 — SparkLM-Tiny

**Suggested time:** 1 week

**Objectives**

- Build the first complete language model with enough capacity to show meaningful behaviour while remaining easy to debug.

**Suggested starting configuration**

```text
vocabulary: character-level initially
context length: 128–256
embedding width: 128–256
layers: 4–6
heads: 4–8
```

The exact configuration is less important than documenting parameter count, memory use, training tokens, and throughput.

**Deliverables**

- A complete model class with parameter counting.
- Training and generation commands.
- Checkpoints and loss curves.
- A small qualitative sample report.

**Acceptance criteria**

- The model overfits a tiny corpus and learns a held-out corpus better than the baseline.
- Training can resume from a checkpoint.
- A run manifest records code version, data version, hyperparameters, seed, device, and metrics.

### Phase 6 — Tokenisation upgrade and SparkLM-125M design

**Suggested time:** 1 week

**Objectives**

- Understand why subword tokenisation is useful before committing compute to the larger model.
- Design the 125M model from measured memory and throughput rather than from a label alone.

**Implementation tasks**

- Keep the character tokenizer as a reference baseline.
- Add or integrate a well-documented subword tokenizer only after the character path is stable.
- Measure average tokens per character/word, vocabulary size, sequence packing efficiency, and unknown handling.
- Calculate parameter count and estimate activation/checkpoint memory.

**Acceptance criteria**

- Tokenizer choice is justified with measurements.
- The 125M configuration fits the DGX Spark memory budget with room for the data pipeline and optimiser state.
- A dry run completes forward, backward, checkpoint, and generation steps at the planned sequence length.

### Phase 7 — Pretraining SparkLM-125M

**Suggested time:** 2–3 weeks or longer

**Objectives**

- Run a disciplined pretraining experiment and learn how compute, data, batch size, learning rate, and context length interact.

**Implementation tasks**

- Finalise corpus filtering, deduplication strategy, and train/validation/test boundaries.
- Add gradient accumulation, mixed precision where understood, gradient clipping, learning-rate scheduling, and periodic evaluation.
- Log tokens processed, loss, perplexity, learning rate, throughput, memory, and checkpoint location.
- Save enough metadata to reproduce or compare each run.

**Experiments**

- Learning-rate range test or several small-scale learning-rate trials.
- Context-length comparison.
- Batch-size/gradient-accumulation comparison at similar effective batch size.
- Ablation of dropout or weight decay.
- Compare manual attention against optimised attention after correctness is established.

**Acceptance criteria**

- The run is stable and can resume after interruption.
- Validation loss improves over the baseline and does not merely memorise a tiny sample.
- Generated text shows increasing structure across checkpoints.
- At least one failed or unstable run is analysed in the learning log.

### Phase 8 — Evaluation and error analysis

**Suggested time:** 1 week

**Objectives**

- Treat evaluation as a first-class engineering and learning activity.

**Implementation tasks**

- Implement held-out next-token loss and perplexity carefully.
- Create a fixed prompt suite covering short continuation, long-context continuation, formatting, numbers, names, and domain-like text.
- Add checks for repetition, truncation, invalid characters/tokens, and sensitivity to temperature.
- Separate automatic metrics from qualitative review.

**Acceptance criteria**

- Every model comparison uses the same evaluation data and prompt suite.
- Results include uncertainty or at least repeated-seed context where practical.
- You can identify several concrete failure modes and connect them to likely causes.

### Phase 9 — Property-management domain adaptation

**Suggested time:** 1–2 weeks

**Objectives**

- Adapt the base model to property-management language while preserving general language ability as much as possible.

**Implementation tasks**

- Assemble lawful, representative domain text: policies, maintenance terminology, notices, procedures, public guidance, and synthetic examples where appropriate.
- Remove confidential, personal, and tenant-identifying information.
- Measure domain-token coverage and compare domain versus general validation loss.
- Start with continued pretraining on domain text, using a conservative learning rate and a held-out general-text check.

**Experiments**

- Domain-only adaptation versus mixed domain/general batches.
- Short versus longer adaptation duration.
- Compare catastrophic forgetting on the general validation set.

**Acceptance criteria**

- The adapted model improves on held-out domain text.
- It does not show unacceptable degradation on the general reference set.
- Data handling, privacy, and limitations are documented.

### Phase 10 — Instruction tuning and LoRA

**Suggested time:** 1–2 weeks

**Objectives**

- Teach the model to follow requests and compare full fine-tuning with parameter-efficient adaptation.

**Implementation tasks**

- Define a small, high-quality instruction format with explicit prompt, context, response, and end-of-sequence handling.
- Create training, validation, and challenge sets.
- Begin with supervised instruction tuning on the full model or a small controlled run.
- Implement or integrate LoRA only after understanding the base fine-tuning path.
- Compare trainable parameter count, loss, quality, memory, and training time.

**Acceptance criteria**

- The model distinguishes prompt/context/answer boundaries correctly.
- Instruction-following improves on held-out tasks, not only training examples.
- LoRA results are reported against a comparable full fine-tuning baseline.
- The limitations of small-data instruction tuning are explicit.

### Phase 11 — Distillation

**Suggested time:** 1 week

**Objectives**

- Understand how a stronger teacher can transfer useful behaviour to a smaller student.

**Implementation tasks**

- Choose a teacher and define whether supervision uses hard labels, soft logits, generated responses, or a combination.
- Build a reproducible distillation dataset or on-the-fly teacher pipeline.
- Implement temperature-scaled soft-target loss and combine it with ordinary next-token loss.
- Compare a distilled student with an independently trained student of the same size.

**Acceptance criteria**

- The student is evaluated on identical held-out tests.
- Quality, latency, memory, and model size are compared.
- You can explain what information distillation transfers and what it cannot guarantee.

### Phase 12 — Quantisation and local inference

**Suggested time:** 1 week

**Objectives**

- Produce a practical local model and measure the real trade-offs of compression.

**Implementation tasks**

- Establish a full-precision reference model and benchmark first.
- Try an appropriate post-training quantisation path, then compare with quantisation-aware or weight-only approaches if useful.
- Measure model size, load time, tokens per second, memory use, perplexity, and prompt-suite quality.
- Build a simple local inference interface with configurable temperature, top-k/top-p if implemented, maximum tokens, and reproducible seed.

**Acceptance criteria**

- The quantised model runs locally and produces valid output.
- Differences from the reference model are measured rather than assumed.
- The inference instructions work from a clean environment using documented commands.

## 5. Suggested repository structure

```text
SparkLM/
├── README.md
├── IMPLEMENTATION_PLAN.md
├── LEARNING_LOG.md
├── pyproject.toml
├── configs/
│   ├── tiny.yaml
│   ├── sparklm_tiny.yaml
│   └── sparklm_125m.yaml
├── data/
│   ├── README.md
│   ├── raw/                 # ignored; never commit sensitive data
│   ├── interim/
│   └── processed/
├── notebooks/               # focused explorations, not production training
├── src/sparklm/
│   ├── tokenizers/
│   ├── data.py
│   ├── attention.py
│   ├── layers.py
│   ├── model.py
│   ├── generation.py
│   ├── train.py
│   ├── evaluate.py
│   ├── adaptation.py
│   ├── lora.py
│   ├── distillation.py
│   └── quantization.py
├── tests/
│   ├── test_tokenizer.py
│   ├── test_data.py
│   ├── test_attention.py
│   ├── test_model_shapes.py
│   ├── test_generation.py
│   └── test_checkpointing.py
├── experiments/
│   ├── README.md
│   └── runs/                # manifests and summaries; large artifacts elsewhere
├── checkpoints/             # ignored or externally mounted
└── scripts/
    ├── smoke_test.py
    ├── count_parameters.py
    ├── train.py
    └── generate.py
```

Keep source code small and legible. Prefer one clear implementation to a general framework. Add abstraction only when a repeated experiment proves that it is needed.

## 6. What to write manually and what libraries may handle

### Write manually during the learning path

- Character tokenizer and encode/decode logic.
- Sequence creation, target shifting, and batch inspection.
- Attention score calculation, causal masking, softmax flow, and value aggregation.
- Multi-head splitting and recombination.
- Transformer block composition, residual paths, and normalisation placement.
- Model forward pass and logits interpretation.
- Generation loop and sampling decisions.
- Training and validation loops.
- Checkpoint metadata and experiment summaries.
- Evaluation prompt suite and error analysis.

### Appropriate library responsibilities

- Tensor operations, automatic differentiation, optimisers, schedulers, and standard layers such as `LayerNorm` and `Linear`.
- CUDA/device management, mixed-precision primitives, data loading, and distributed or accelerated execution after the single-device path is understood.
- Plotting, logging, experiment tracking, and filesystem-safe checkpoint storage.
- A mature subword tokenizer after the character tokenizer is understood and retained as a baseline.
- Reference implementations for numerical comparisons and production-quality serving.
- LoRA, distillation, and quantisation libraries after you have implemented a minimal conceptual version or written a clear explanation of what the library is doing.

Do not use `nn.Transformer` or `AutoModelForCausalLM` as the primary implementation in the early phases. Use them later as reference points, compatibility targets, or performance comparisons.

## 7. Definition of done

SparkLM is in a strong first release when:

- The complete model path is understandable from raw text to generated text.
- Core components have tests and explanatory notes.
- SparkLM-Tiny and SparkLM-125M have reproducible training manifests.
- Pretraining, domain adaptation, instruction tuning, LoRA, distillation, and quantisation each have a documented comparison or a clearly stated reason for deferral.
- Evaluation includes both held-out loss/perplexity and a fixed qualitative prompt suite.
- The final local inference path is repeatable and its trade-offs are measured.
- The repository explains what was learned, what failed, and what remains uncertain.

The most important output is not only a model file. It is a durable understanding of why the model works, how to tell when it does not, and which engineering choices changed the result.
