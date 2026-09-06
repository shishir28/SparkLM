# SparkLM Implementation Plan

## Learn to Build a Decoder-Only Language Model from Scratch

This is a learning plan, not a race to produce the largest possible model. It follows
Sebastian Raschka's *Build a Large Language Model (From Scratch)* in order, while
adding the tests, experiment records, data controls, and DGX Spark measurements
needed to make the work reproducible.

The starting assumption is deliberately strict: **nothing is complete yet**. Existing
repository files may be studied or replaced, but a milestone counts only after its
concepts have been implemented, tested, explained, and recorded during this learning
journey.

The core path is:

```text
environment and PyTorch foundations
  -> language-model concepts
  -> character-tokenizer baseline
  -> BPE and sliding-window data pipeline
  -> simple self-attention
  -> causal multi-head attention
  -> GPT-style decoder-only model
  -> pretraining, evaluation, and generation
  -> classification fine-tuning
  -> instruction fine-tuning
  -> LoRA
  -> optional SparkLM extensions
```

## 1. Outcomes

By the end of the core path, you should be able to:

1. Explain how next-token prediction turns unlabeled text into a training task.
2. Trace text through tokenization, token IDs, embeddings, Transformer blocks,
   logits, loss, sampling, and decoded output.
3. Implement a GPT-style decoder-only model in PyTorch without using
   `torch.nn.Transformer` or a pretrained causal-language-model wrapper for the core
   implementation.
4. Derive and implement scaled dot-product attention, causal masking, and multi-head
   attention with inspectable tensor shapes.
5. Pretrain a small model that you own, evaluate it on held-out data, save and resume
   it, and explain its failure modes.
6. Load a compatible reference checkpoint as a separate interoperability exercise.
7. Fine-tune a pretrained model for classification and instruction following.
8. Implement a minimal LoRA path and compare it fairly with full fine-tuning.
9. Reproduce every meaningful result from a recorded configuration, dataset version,
   code version, seed, and device description.

The goal is durable understanding. A large checkpoint without correctness evidence
does not satisfy this plan.

## 2. How to Use the Book

Read the book sequentially because each chapter depends on the preceding chapters.
For each section, use this loop:

1. **Predict:** write down expected inputs, outputs, and tensor shapes before coding.
2. **Implement:** write the smallest clear version yourself.
3. **Inspect:** print or debug intermediate tensors on a tiny deterministic example.
4. **Test:** check shapes, numerical results, gradients, and behavioral invariants.
5. **Compare:** only then compare with the book code or a trusted PyTorch reference.
6. **Explain:** record what the component does, why it is needed, and one way it can
   fail in `LEARNING_LOG.md`.

Do not copy the book's implementation as the first step. The book is the guide and
reference; the SparkLM code should be your own inspectable implementation. Do not
copy prose, figures, or substantial book content into this repository.

### Book-to-project map

| Book material | SparkLM phase | Result |
| --- | --- | --- |
| Appendix A, as needed | Phase 0 | PyTorch and device readiness |
| Chapter 1 | Phase 1 | Written mental model and scope |
| Chapter 2 | Phases 2-3 | Character baseline, BPE, batches, embeddings |
| Chapter 3, sections 3.1-3.4 | Phase 4 | Simple and trainable self-attention |
| Chapter 3, sections 3.5-3.6 | Phase 5 | Causal and multi-head attention |
| Chapter 4 | Phase 6 | Complete GPT-style model and generation |
| Chapter 5 | Phases 7-8 | Pretraining, evaluation, sampling, checkpoints |
| Appendix D | Phase 7, after baseline | Warmup, cosine decay, clipping |
| Chapter 6 | Phase 9 | Classification fine-tuning |
| Chapter 7 | Phase 10 | Instruction fine-tuning and evaluation |
| Appendix E | Phase 11 | LoRA implementation and comparison |

Classification is included because it teaches how a pretrained backbone can be
adapted by changing its objective and output head. It is not a prerequisite for
instruction tuning, but it should be completed once as part of the book path.

## 3. Non-Negotiable Rules

### Implement the learning path explicitly

Write the character tokenizer, input/target construction, attention calculations,
multi-head reshape and recombination, Transformer block, model forward pass,
generation loop, training loop, checkpoint metadata, and evaluation logic yourself.
PyTorch may provide tensors, autograd, optimizers, `Linear`, `Embedding`,
`LayerNorm`, data loading, and device management.

### Establish correctness before optimization

Start with tiny CPU-friendly tensors and no dropout. Add batching, dropout, mixed
precision, fused attention, compilation, or other performance features only after the
simple path has direct tests. Keep the simple implementation available as a reference.

### Separate the two model tracks

- **Ownership track:** initialize SparkLM weights randomly and pretrain the model on a
  lawful corpus. This is the proof that you built and trained your own language model.
- **Reference track:** map an openly available GPT-2-compatible checkpoint into your
  architecture. Use it to test compatibility and to run meaningful fine-tuning without
  pretending those weights were pretrained by this project.

Never mix results from these tracks without labeling them.

### Treat data as part of the implementation

Before any dataset is used, record its name, source, license or usage permission,
acquisition date, content hash or version, privacy decision, split method, and every
preprocessing transformation in `essentials/DATA_POLICY.md` or a dataset manifest.
Do not use the book PDF itself as training data.

### Treat experiments as claims

Every meaningful run must state its question and hypothesis before execution, then
record the code version, config, data version, split, tokenizer, seed, device, metrics,
result, limitations, and next decision. Keep useful failed runs.

### Scale only through exit gates

Do not move to the next phase merely because the code runs. Complete the phase's exit
gate, learning note, and tests first. Do not start the optional 124M-scale work until a
smaller model is correct and reproducible.

## 4. Core Learning Path

### Phase 0 - Foundations from Zero

**Book:** Appendix A as needed; skim the book's setup and code conventions.

**Learn**

- Python classes, imports, virtual environments, and command-line execution.
- Tensor creation, shapes, dtypes, indexing, broadcasting, matrix multiplication,
  gradients, modules, optimizers, datasets, data loaders, and train/eval modes.
- The difference between CPU execution and CUDA execution.
- Reproducibility limits: fixed seeds improve repeatability but do not make all GPU
  operations bitwise deterministic.

**Build**

- A documented Python environment with explicit dependencies.
- A one-command smoke test reporting Python, PyTorch, CUDA, GPU, memory, storage,
  and successful tensor placement.
- A single seed helper covering Python and PyTorch CPU/CUDA RNGs.
- Initial data, artifact, experiment, and privacy conventions.
- A fast test command that works from the repository root.

**Verify**

- Recreating the environment from the documented steps succeeds.
- Repeated seeded CPU examples match exactly.
- Repeated seeded GPU examples are checked and any nondeterminism is documented.
- Tests, checkpoints, datasets, and generated artifacts have deliberate locations;
  large or sensitive artifacts are ignored by Git.

**Learning artifact**

Record the environment, the first smoke-test result, what a gradient represents, and
the shapes produced by one small matrix multiplication.

**Exit gate**

The smoke test and unit-test command pass from a clean shell, the environment is
documented, and the data policy contains a complete template for future datasets.

### Phase 1 - Understand the LLM Development Stages

**Book:** Chapter 1, "Understanding large language models."

**Learn**

- What an LLM is and what next-token prediction actually optimizes.
- The distinction between architecture construction, pretraining, and fine-tuning.
- Why attention and decoder-only Transformers are suited to autoregressive text.
- Why a small educational pretraining run is different from frontier-model training.

**Build**

- No model code yet.
- Write a one-page explanation, in your own words, of the flow from raw text to a
  generated token.
- Define the ownership-track target: a small model that can be pretrained locally.
- Define success metrics for the first corpus: held-out loss, perplexity, parameter
  count, memory, throughput, and qualitative samples.

**Verify**

- Explain the roles of data, parameters, loss, gradients, and sampling without looking
  at the book.
- Explain why pretraining and instruction fine-tuning are different objectives.

**Exit gate**

The learning note is complete and the first corpus is proposed, but no data is acquired
until its provenance and license are recorded.

### Phase 2 - Character Tokenizer and Next-Token Data Baseline

**Book:** Chapter 2, sections 2.1-2.4 and 2.6. This character path is a SparkLM
prerequisite added to make every transformation visible before BPE.

**Learn**

- Tokens, vocabularies, token IDs, unknown tokens, document boundaries, and special
  tokens.
- How a context window creates an input sequence and a target shifted by one token.
- How stride changes overlap and the number of training examples.

**Build**

- A sorted, deterministic character vocabulary.
- `encode(text) -> list[int]` and `decode(ids) -> str`.
- Explicit handling for unsupported characters and end-of-text boundaries.
- Train/validation/test splitting before sliding-window sampling.
- A dataset and batcher that return `(batch, sequence)` input and target tensors.
- A unigram or bigram baseline for context.

**Verify**

- Representative supported text satisfies `decode(encode(text)) == text`.
- Vocabulary IDs are stable across runs.
- Every target token is the next input token, including batch boundaries.
- Splits do not overlap and windows do not cross split or document boundaries unless
  that behavior is explicitly intended and tested.
- Invalid IDs and unsupported characters have tested behavior.

**Learning artifact**

Draw one sequence of characters, IDs, input windows, and shifted targets. Explain
context length and stride.

**Exit gate**

Tokenizer round trips, split integrity, batch shapes, and one-position target alignment
have deterministic tests. The baseline metric is recorded.

### Phase 3 - BPE, Embeddings, and Positional Information

**Book:** Finish Chapter 2, especially sections 2.5-2.8.

**Learn**

- Why subword tokenization balances word and character vocabularies.
- How byte pair encoding handles unfamiliar words.
- The roles of token embeddings and positional embeddings.
- Why embedding lookup and a one-hot vector multiplied by a weight matrix are
  equivalent operations.

**Build**

- A tiny educational BPE merge exercise on a toy vocabulary.
- Integration with a mature GPT-2-compatible BPE tokenizer for the book path.
- Special-token policy with explicit allowed and disallowed cases.
- Sliding-window loaders for BPE token IDs.
- Token and absolute positional embedding layers whose output shapes are explicit.
- Keep the Phase 2 character tokenizer working as a permanent baseline.

**Verify**

- BPE encode/decode round trips work for normal text and representative edge cases.
- Special tokens cannot be inserted accidentally.
- Character and BPE loaders obey the same target-alignment contract.
- Embedding output has shape `(batch, sequence, embedding_dimension)`.
- Position IDs reset and truncate exactly as documented.

**Experiment**

On the same lawful sample, compare vocabulary size, tokens per character, tokens per
word, sequence lengths, and round-trip behavior for character and BPE tokenization.

**Exit gate**

The BPE choice is justified by measurements, both tokenizer paths pass tests, and
embedding shapes are documented and tested.

### Phase 4 - Self-Attention from First Principles

**Book:** Chapter 3, sections 3.1-3.4.

**Learn**

- Why fixed-context representations lose information over long sequences.
- How dot products produce attention scores and softmax produces normalized weights.
- How queries, keys, and values create trainable context vectors.
- Why scores are scaled by the square root of the key dimension.

**Build**

- A loop-based attention calculation for one query.
- A matrix-based unbatched self-attention calculation for all tokens.
- A trainable self-attention module with separate Q, K, and V projections.
- A batched version with named shape assertions at important boundaries.
- Do not add causal masking, multiple heads, or dropout yet.

**Verify**

- Attention rows sum to one within a stated tolerance.
- Loop and matrix implementations agree on a hand-calculated example.
- Batched and unbatched results agree for the same example.
- Forward values and gradients agree with a trusted reference or an independently
  expressed calculation within documented tolerances.

**Learning artifact**

Annotate the shapes in `Q @ K.transpose(-2, -1)`, softmax, and `weights @ V`.
Explain the meanings of query, key, and value in your own words.

**Exit gate**

The unmasked implementation has shape, value, normalization, and gradient tests, and
every matrix dimension can be explained without consulting the code.

### Phase 5 - Causal and Multi-Head Attention

**Book:** Chapter 3, sections 3.5-3.6.

**Learn**

- Why autoregressive training must prevent access to future tokens.
- Why masking is applied to scores before softmax.
- What separate attention heads can represent and how head dimensions are combined.
- The difference between attention dropout and residual dropout.

**Build**

- A causal mask registered as a non-parameter buffer.
- Causal self-attention without dropout first, then with configurable dropout.
- A pedagogical multi-head wrapper using independent heads.
- An efficient multi-head implementation using combined projections, reshape,
  transpose, attention, recombination, and output projection.
- Keep the pedagogical implementation as a numerical reference.

**Verify**

- Future attention probabilities are zero.
- Changing a future token cannot change an earlier output in evaluation mode.
- All masked attention rows still sum to one.
- Pedagogical and efficient multi-head implementations agree when their weights are
  copied into the same layout.
- Invalid head counts and dimensions fail clearly.
- Forward outputs and gradients agree with PyTorch scaled-dot-product attention within
  a documented tolerance after the manual implementation passes direct tests.

**Exit gate**

Causality is proven by a direct influence test, not inferred from generated text, and
the multi-head split/recombine path has deterministic numerical and gradient tests.

### Phase 6 - Assemble a GPT-Style Decoder-Only Model

**Book:** Chapter 4.

**Learn**

- Layer normalization, GELU, feed-forward expansion, residual connections, and
  pre-normalization.
- How repeated Transformer blocks preserve sequence shape.
- How the language-model head maps hidden states to vocabulary logits.
- Why an untrained model generates valid-shaped but meaningless output.

**Build**

- Layer normalization as a small educational implementation, then compare it with
  `torch.nn.LayerNorm`.
- The GELU approximation used by the GPT-style path.
- A feed-forward network and residual connection experiments.
- A pre-normalized Transformer block.
- A complete configurable `GPTModel`: token embeddings, positional embeddings,
  embedding dropout, repeated blocks, final normalization, and output head.
- Parameter counting and estimated parameter-storage size.
- A greedy autoregressive generation loop that crops context correctly.

**Verify**

- Each module has deterministic shape and finite-value tests.
- Layer normalization gives the expected mean and variance within tolerance.
- Residual connections improve gradient flow in a controlled deep-network example.
- The full model maps `(B, T)` token IDs to `(B, T, vocabulary_size)` logits.
- Generation respects prompt preservation, context length, batch dimension, and
  maximum-new-token limits.
- An untrained model completes forward, backward, and generation passes on CPU and
  CUDA where available.

**Experiment**

Instantiate at least two small configurations and compare parameter count, forward
memory, and latency. A GPT-2 124M-style configuration may be instantiated for shape
and memory planning, but it is not yet a required training target.

**Exit gate**

The model passes component and integration tests, and a learning note traces one token
sequence through every shape from IDs to logits.

### Phase 7 - Pretrain SparkLM-Tiny

**Book:** Chapter 5, sections 5.1-5.4. Use Appendix D only after the baseline loop is
working and tested.

**Learn**

- Cross-entropy over flattened token positions, held-out loss, and perplexity.
- Training versus evaluation mode and the effect of dropout.
- Optimizer steps, epoch/token accounting, validation intervals, and overfitting.
- Checkpoint contents and the difference between resuming training and loading only
  model weights.

**Build**

- Loss calculation for a batch and a bounded data loader.
- A transparent AdamW training loop with periodic validation.
- Deliberate tiny-batch overfitting.
- Atomic checkpoints containing model state, optimizer state, step/epoch, config,
  tokenizer identity, seed/RNG state, and dataset/code metadata.
- Exact resume support.
- Loss-curve and generated-sample recording.
- After the baseline is correct: learning-rate warmup, cosine decay, and gradient
  clipping as isolated additions.

**Verify**

- Cross-entropy agrees with a hand-checked tiny example.
- A tiny batch can be overfit to very low loss.
- Evaluation does not compute gradients and uses evaluation mode.
- Checkpoint reload preserves logits and held-out loss.
- An interrupted run resumed from a checkpoint follows the documented reproducibility
  expectations.
- Train, validation, and test data remain disjoint.

**Required experiment**

Pretrain a genuinely small ownership-track model first. Record:

- question and hypothesis;
- model and tokenizer configuration;
- parameter count and tokens processed;
- training and validation loss, plus perplexity where meaningful;
- learning rate, throughput, peak memory, and elapsed time;
- fixed-prompt samples from several checkpoints;
- observed failure modes and conclusion.

**Exit gate**

SparkLM-Tiny beats the Phase 2 baseline on held-out data, produces progressively more
structured samples, resumes from a checkpoint, and has a reproducible run manifest.

### Phase 8 - Generation and Reference-Weight Compatibility

**Book:** Chapter 5, especially sections 5.3 and 5.5.

**Learn**

- Greedy decoding, probabilistic sampling, temperature, and top-k filtering.
- Why generation quality cannot be inferred from training loss alone.
- How external checkpoint tensor names and layouts map into your implementation.

**Build**

- Seeded multinomial sampling with temperature.
- Top-k filtering, with greedy decoding as a clearly separate mode.
- Input validation for invalid temperature, context, and generation limits.
- A fixed qualitative prompt suite and structured sample output.
- A documented adapter that loads one GPT-2-compatible reference checkpoint into the
  SparkLM architecture, with no silent reshaping.

**Verify**

- Greedy generation is deterministic.
- Seeded sampling is repeatable under the documented environment.
- Temperature and top-k behavior is tested on controlled logits.
- Every checkpoint assignment validates tensor shape and dtype.
- Reference weights produce stable expected logits or generated text for a fixed input.
- Ownership-track and reference-track results are clearly labeled.

**Exit gate**

Generation modes pass controlled tests, the prompt suite is versioned, and reference
weights load through an explicit verified mapping. Downloading reference weights is
optional and must have source and license metadata.

### Phase 9 - Classification Fine-Tuning

**Book:** Chapter 6.

**Learn**

- The difference between classification and generative objectives.
- Dataset balance, stratified splitting, padding/truncation, frozen layers, and which
  representation feeds a classifier.
- Loss versus accuracy and why held-out evaluation matters.

**Build**

- A provenance-reviewed labeled text dataset pipeline.
- Deterministic train/validation/test splits and data loaders.
- Replacement of the language-model head with a classification head.
- A staged fine-tuning policy: head only, selected final layers, then full fine-tuning
  only if justified.
- Classification loss, accuracy, confusion matrix, and prediction helper.

**Verify**

- Split class distributions and non-overlap are checked.
- Padding and truncation behavior is tested.
- Frozen parameters receive no gradients or updates.
- Accuracy calculation matches a hand-checked example.
- Saving and loading preserves predictions.

**Experiment**

Compare at least two freezing strategies using the same split and seeds. Report
trainable parameters, validation/test accuracy, time, memory, and common errors.

**Exit gate**

The classifier beats a majority-class baseline on held-out data, the comparison is
recorded, and errors are inspected rather than represented by accuracy alone.

### Phase 10 - Instruction Fine-Tuning

**Book:** Chapter 7.

**Learn**

- Instruction, optional input/context, and response formatting.
- Padding, end-of-text handling, and loss masking for variable-length examples.
- Why supervised instruction fine-tuning changes behavior rather than adding reliable
  factual knowledge.
- The limits and risks of model-based evaluation.

**Build**

- A small, lawful, versioned instruction dataset with train/validation/test boundaries.
- A deterministic prompt formatter.
- A custom collate function that pads sequences, shifts targets, and masks padding
  positions with an ignored loss index.
- Optional response-only loss masking as a documented experiment.
- Fine-tuning from a labeled pretrained checkpoint.
- Response extraction and storage in a reviewable structured format.
- A fixed rubric covering instruction adherence, correctness, relevance, formatting,
  unsupported claims, and refusal behavior where applicable.

**Verify**

- Formatting is tested with and without optional input.
- Target shifting and padding masks are inspected token by token.
- Padding beyond the first end token contributes no loss.
- Generated responses stop and are decoded as intended.
- The test set is not used to select hyperparameters.

**Experiment**

Compare the pretrained and instruction-tuned checkpoints on the exact same held-out
prompts. Use human review as the primary small-scale check. Any automated judge must
record the judge model, prompt, settings, failure modes, and score uncertainty.

**Exit gate**

Held-out instruction following improves under the fixed rubric, dataset and masking
tests pass, and limitations are written down explicitly.

### Phase 11 - LoRA

**Book:** Appendix E.

**Learn**

- How a low-rank product approximates a weight update.
- The meanings of rank and alpha, zero-impact initialization, and frozen base weights.
- Why fewer trainable parameters do not automatically guarantee lower latency.

**Build**

- A minimal `LoRALayer` with A and B matrices and explicit scaling.
- A wrapper for selected linear layers that adds the LoRA update to the frozen base
  output.
- Target-module selection rather than unexamined global replacement.
- Save/load of adapter-only weights with base-checkpoint identity.

**Verify**

- With B initialized to zero, adding LoRA does not change model outputs.
- Only intended adapter parameters require gradients and update.
- Rank, alpha, parameter count, device, and dtype behavior are tested.
- Merged or adapter-loaded output matches the live adapter path if merging is added.

**Experiment**

Compare LoRA with full fine-tuning on the same task, base checkpoint, data split,
effective batch size, evaluation suite, and seed. Report trainable parameters, peak
memory, wall time, checkpoint size, held-out quality, and failure modes.

**Exit gate**

LoRA's behavior and trainable-parameter reduction are proven by tests, and the fair
comparison is recorded without claiming improvement from training loss alone.

## 5. Optional SparkLM Extensions

These phases go beyond the book. Begin them only after Phase 11, or explicitly record
why a book-path phase is being deferred.

### Phase 12 - Domain-Adaptive Pretraining

- Select a lawful, privacy-reviewed domain corpus.
- Continue next-token pretraining with a conservative learning rate.
- Compare domain-only batches with a domain/general mixture.
- Measure held-out domain improvement and general-language forgetting.
- Do not confuse better domain continuation with instruction-following ability.

**Exit gate:** domain held-out metrics improve without unacceptable degradation on the
fixed general evaluation set, and all provenance and privacy decisions are documented.

### Phase 13 - Design and Train a 124M-Class SparkLM

- Use the tested architecture to propose a GPT-2-small-like configuration around 124M
  parameters; derive the exact count rather than naming it approximately.
- Estimate parameter, gradient, optimizer, activation, and checkpoint memory.
- Complete forward, backward, optimizer, checkpoint, resume, and generation dry runs
  at the proposed context length.
- Measure DGX Spark throughput and memory before choosing batch size or accumulation.
- Scale data and training tokens only after a small proxy run validates the setup.
- Add mixed precision or optimized attention one change at a time, comparing each with
  the simple tested path.

**Exit gate:** the run is stable and resumable, has improving held-out metrics, and has
a complete manifest and honest compute/data limitations. Training this model is an
optional compute project, not proof of understanding by itself.

### Phase 14 - Distillation, Quantisation, and Local Inference

- Establish an uncompressed student/reference baseline first.
- For distillation, define the teacher, data source, hard/soft targets, temperature, and
  combined objective; compare with a same-size independently trained student.
- For quantisation, measure full-precision and quantised model size, load time, peak
  memory, tokens per second, held-out loss, and prompt-suite behavior.
- Provide a local inference command with explicit checkpoint, tokenizer, seed,
  temperature, top-k, context limit, and maximum-new-token settings.
- Keep serving code outside the core model so optimization does not obscure learning
  implementations.

**Exit gate:** compressed models are compared on identical data and prompts, local
inference is repeatable, and quality/performance trade-offs are measured rather than
assumed.

## 6. Testing Strategy

Use four levels of verification:

1. **Unit tests:** tokenization, windows, embeddings, masks, attention, normalization,
   sampling, loss masking, and checkpoint metadata.
2. **Numerical tests:** hand calculations and trusted-reference comparisons for
   attention, gradients, normalization, loss, and sampling probabilities.
3. **Behavioral tests:** future-token isolation, tiny-batch overfitting, deterministic
   generation, exact checkpoint reload, and freeze/LoRA update boundaries.
4. **Experiment tests:** fixed data splits, manifests, fixed prompt suites, comparable
   settings, and explicit conclusions.

Every bug in a core invariant should produce a regression test before it is considered
fixed. Tests should remain small enough to run routinely; long GPU experiments are
recorded separately rather than hidden inside the unit-test suite.

## 7. Experiment Record Contract

Each meaningful run should create a small version-controlled manifest or summary
containing:

```yaml
question: "What are we trying to learn?"
hypothesis: "What result do we expect, and why?"
code_version: "git commit or explicit dirty-worktree note"
config: "path plus resolved values"
dataset: "name, version/hash, split, preprocessing manifest"
tokenizer: "type, vocabulary/version, special-token policy"
seed: 42
device: "CPU/GPU name, relevant software versions"
precision: "float32, bfloat16, etc."
model: "architecture and exact parameter count"
training: "optimizer, LR schedule, batch/accumulation, tokens, steps"
metrics: "train/validation loss, perplexity, task metrics"
performance: "elapsed time, tokens/s, peak memory, checkpoint size"
artifacts: "checkpoint and sample locations, kept outside Git if large"
result: "observations including failures"
conclusion: "whether the hypothesis was supported"
next_decision: "what changes or remains fixed"
```

For comparisons, keep evaluation data, prompt suite, decoding settings, and important
training variables fixed. Change one important variable at a time unless the experiment
explicitly studies an interaction.

## 8. Target Repository Shape

Create directories only when their phase begins; do not scaffold all future work at
once.

```text
SparkLM/
|-- AGENTS.md
|-- README.md
|-- IMPLEMENTATION_PLAN.md
|-- LEARNING_LOG.md
|-- pyproject.toml
|-- essentials/
|   |-- README.md
|   |-- ENVIRONMENT.md
|   |-- SMOKE_TEST.md
|   |-- DATA_POLICY.md
|   `-- PROJECT_RULES.md
|-- configs/
|-- data/
|   |-- README.md
|   |-- raw/                  # ignored
|   |-- interim/              # ignored
|   `-- processed/            # ignored unless tiny and licensed
|-- experiments/
|   |-- README.md
|   `-- runs/                 # small manifests and summaries only
|-- src/sparklm/
|   |-- tokenizers/
|   |-- data.py
|   |-- attention.py
|   |-- layers.py
|   |-- model.py
|   |-- generation.py
|   |-- training.py
|   |-- evaluation.py
|   `-- lora.py
|-- tests/
|-- scripts/
`-- checkpoints/             # ignored or externally mounted
```

Keep notebooks for disposable exploration only. Move behavior that matters into small
source modules with tests and commands.

## 9. Core Definition of Done

The book-aligned SparkLM learning path is complete when:

- every Phase 0-11 exit gate is met or a deferral has a written technical reason;
- the ownership-track model was initialized and pretrained by this project;
- raw text can be traced through tokenization, batching, embeddings, attention,
  Transformer blocks, logits, loss, optimization, checkpointing, and generation;
- causal masking and core tensor operations have direct numerical and gradient tests;
- a tiny batch overfits, a held-out set improves, and a checkpoint resumes correctly;
- classification and instruction fine-tuning have held-out evaluations;
- LoRA has a fair full-fine-tuning comparison;
- datasets, experiments, failures, and limitations are documented;
- you can explain the implementation without relying on a framework abstraction or
  merely repeating the book's code.

## 10. Immediate Starting Checklist

Start with Phase 0 only:

1. Read Chapter 1's roadmap and Appendix A sections needed for unfamiliar PyTorch
   concepts.
2. Re-run and understand the device smoke test; do not accept an old result as proof.
3. Recreate the environment notes and deterministic seed experiment yourself.
4. Run the current tests only as a diagnostic baseline.
5. Record what existing code is present, then replace or validate it phase by phase.
6. Complete the Phase 0 exit gate before writing tokenizer or model code.

There is intentionally no deadline. Move forward when the evidence says the current
phase is understood and correct.
