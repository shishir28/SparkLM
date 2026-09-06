# SparkLM Implementation Plan

## Code Along With *Build a Large Language Model (From Scratch)*

This repository is a hands-on companion to Sebastian Raschka's *Build a Large Language Model (From Scratch)*. The book's teaching order is the implementation order. SparkLM does not add a competing curriculum, skip ahead to a larger model, or treat old repository code as completed work.

The starting assumption is: **begin from scratch**. Existing code may be read for context, deleted, or replaced. A chapter is complete only after its implementation has been rebuilt, exercised, tested, and explained by the learner.

The required path is exactly:

```text
setup
  -> Chapter 1: understand LLMs
  -> Chapter 2: prepare text data
  -> Chapter 3: implement attention
  -> Chapter 4: implement GPT and generate text
  -> Chapter 5: pretrain, evaluate, sample, and load GPT-2 weights
  -> Chapter 6: fine-tune for classification
  -> Chapter 7: fine-tune to follow instructions
  -> Appendix D: improve the training loop
  -> Appendix E: implement LoRA
```

Domain adaptation, training a separate 124M SparkLM at scale, distillation, quantisation, serving, and production architecture are **not part of this plan**. They may be considered in a separate post-book roadmap only after this plan is complete.

## 1. How to Work Through Each Chapter

For every numbered book section:

1. Read the section without coding.
2. Write the expected inputs, outputs, and tensor shapes in `LEARNING_LOG.md`.
3. Reproduce the section's small example interactively.
4. Write that section's code in a small, inspectable module.
5. Complete the section's exercises before checking Appendix C.
6. Add focused tests for the behavior taught by the section.
7. Compare results with the book and record any deliberate difference.
8. Record a checkpoint before moving to the next section.

The aim is understanding, not transcription. The implementation may use different names or file boundaries, but it must preserve the equations, behavior, and learning sequence taught in the book. Do not copy the book's prose or figures into this project.

### What "from scratch" means

For the core learning implementation:

- write tokenization examples, input/target construction, attention, causal masking, multi-head recombination, Transformer blocks, the GPT forward pass, generation, training, evaluation, and weight assignment explicitly;
- use PyTorch for tensors, autograd, optimizers, `Linear`, `Embedding`, `LayerNorm`, data loading, and device placement;
- do not use `torch.nn.Transformer`, a pretrained causal-LM class, or a training framework in place of the book implementation;
- use `tiktoken` when Chapter 2 introduces GPT-2 byte pair encoding;
- load OpenAI GPT-2 weights only when Chapter 5 reaches that step.

### Supporting engineering checks

Tests, reproducibility records, and data provenance support the book exercises; they do not create extra curriculum phases. At minimum:

- fix and record random seeds;
- keep tiny examples CPU-runnable;
- assert important tensor shapes;
- test input/target shifting, causal isolation, and tokenizer round trips;
- keep data and checkpoints out of Git unless a small artifact is explicitly safe;
- record dataset source, licence or permitted use, acquisition date, hash/version, preprocessing, and privacy decision before use;
- record meaningful training runs with code version, config, seed, device, metrics, and conclusion.

## 2. Setup - Before Chapter 1

This is preparation for the book, not a separate model-building phase.

### Read

- The preface and "About this book."
- Appendix A only to fill gaps in Python or PyTorch knowledge.
- The book's installation and repository notes.

### Code

- Create a local Python environment.
- Install the book-path dependencies, beginning with PyTorch and `tiktoken`.
- Add one command that reports Python, PyTorch, CUDA, GPU, and tensor device.
- Add a shared seed helper for Python and PyTorch.
- Establish `src/`, `tests/`, `data/`, `checkpoints/`, and `experiments/` only as each becomes necessary.
- Ensure local datasets, downloaded weights, checkpoints, and secrets are ignored.

### Check

- Create tensors on CPU and CUDA when CUDA is available.
- Run a forward calculation, compute a scalar loss, call `backward()`, and observe a parameter gradient.
- Repeated seeded CPU examples produce the same result.
- The test command runs from the repository root.

### Evidence

- `essentials/ENVIRONMENT.md` records the actual environment.
- `essentials/SMOKE_TEST.md` records the command and observed result.
- `essentials/DATA_POLICY.md` contains a usable dataset record template.
- `LEARNING_LOG.md` explains tensors, parameters, gradients, and device placement in the learner's own words.

## 3. Chapter 1 - Understanding Large Language Models

Follow sections 1.1 through 1.7 in order.

### 1.1-1.3: LLMs, applications, and development stages

Read what an LLM predicts, where LLMs are used, and the difference between pretraining and fine-tuning. Draw the book's three broad development stages in your own notation. No model implementation is required yet.

### 1.4-1.6: Transformer and GPT architecture

Explain attention at a conceptual level, why GPT is decoder-only and autoregressive, and how next-token prediction creates labels from raw text. Identify tokenization, embeddings, attention blocks, logits, and token selection in the end-to-end path.

### 1.7: Book roadmap

Write a one-page map of the implementation that Chapters 2-7 will construct. State which work is pretraining, classification fine-tuning, and instruction fine-tuning.

### Chapter checkpoint

- Explain the complete text-to-generated-text flow without consulting the book.
- Explain why a model can learn from unlabeled text.
- Explain why fine-tuning is not the same as pretraining.
- Do not start attention or model code during this chapter.

## 4. Chapter 2 - Working with Text Data

Use the chapter's Edith Wharton short story, "The Verdict," for the exercises. Record its source and public-domain status before adding or downloading the text. Do not use the book PDF as training data.

### 2.1: Understand word embeddings

- Read why text must become numeric vectors.
- Distinguish token embeddings from contextual representations.
- Code a small embedding lookup and inspect its shape.

### 2.2: Tokenize text

- Load and inspect "The Verdict."
- Implement the chapter's regular-expression word and punctuation split.
- Inspect token counts and edge cases such as punctuation and whitespace.

### 2.3: Convert tokens to token IDs

- Build a sorted vocabulary from the text.
- Implement the book's first simple tokenizer with `encode` and `decode`.
- Test ID stability and a supported-text round trip.

### 2.4: Add special context tokens

- Add unknown and end-of-text tokens as taught.
- Implement the second simple tokenizer.
- Demonstrate how independent documents are separated.
- Test known text, unknown words, special tokens, and invalid IDs.

### 2.5: Use byte pair encoding

- Learn the BPE idea and complete Exercise 2.1.
- Use the GPT-2 tokenizer through `tiktoken`, as the book does.
- Compare simple-tokenizer and BPE behavior for known and unknown words.
- Test representative BPE encode/decode round trips.

The repository's character tokenizer may remain as a small compatibility baseline because the project rules require preserving it. It is not a Chapter 2 milestone, must not replace the book's simple tokenizers or BPE work, and must not delay the book sequence.

### 2.6: Sample data with a sliding window

- Construct input sequences and targets shifted by exactly one token.
- Implement the book-equivalent `GPTDatasetV1` and `create_dataloader_v1`.
- Inspect batches at several context lengths and strides.
- Complete Exercise 2.2.

Required tests:

- input and target shapes match;
- `target[:, :-1] == input[:, 1:]` for contiguous examples;
- windows have the expected start offsets;
- short inputs and invalid window settings fail clearly.

### 2.7-2.8: Token and positional embeddings

- Create a token embedding layer.
- Show why embedding lookup matches selection from an embedding matrix.
- Create absolute positional embeddings.
- Add token and positional embeddings to produce tensors shaped `(batch, sequence, embedding_dimension)`.
- Inspect the final embedded batch that Chapter 3 will consume.

### Chapter checkpoint

- Run one command that takes raw text through BPE, sliding windows, and embeddings.
- Explain every shape and the one-position target shift.
- Pass tokenizer, data-window, and embedding tests.
- Record solutions to the chapter exercises without copying Appendix C.

## 5. Chapter 3 - Coding Attention Mechanisms

Follow the chapter's progression. Do not begin with an optimized attention API.

### 3.1-3.2: Long-sequence problem and attention motivation

- Explain the limitation of a fixed encoder representation.
- Explain how attention exposes relevant input positions to the current query.
- Record the meaning of attention scores, weights, and context vectors.

### 3.3: Simple self-attention without trainable weights

- Calculate one context vector step by step using the book's tiny tensor.
- Normalize attention scores and verify that weights sum to one.
- Extend the calculation to all input tokens, first explicitly and then with matrix multiplication.
- Verify the loop and matrix results numerically.

### 3.4: Self-attention with trainable weights

- Create query, key, and value weight matrices explicitly.
- Derive the shapes of Q, K, V, scores, weights, and context vectors.
- Apply scaled dot-product attention.
- Implement both book variants: explicit parameter matrices and a compact class using `Linear` layers.
- Complete Exercise 3.1 by aligning weights and comparing results.

### 3.5: Causal attention

- Apply the causal mask before softmax.
- Add attention-weight dropout only after the unregularized path works.
- Implement the compact batched causal-attention class.
- Register the reusable causal mask as a buffer.

Required causal tests:

- future attention probabilities are zero;
- attention rows sum to one;
- changing a future token cannot change an earlier output in evaluation mode;
- invalid context length fails clearly.

### 3.6: Multi-head attention

- First stack independent causal-attention heads as the book demonstrates.
- Then implement multi-head attention with split projection weights, reshape, transpose, head-wise attention, recombination, and output projection.
- Complete the chapter exercises on head count, dimensions, and parameter sharing.
- Compare the two forms after assigning equivalent weights where practical.

### Chapter checkpoint

- Trace and explain every attention tensor shape.
- Pass numerical, shape, normalization, gradient, and causal-isolation tests.
- Keep the manual attention path available when later optimizations are explored.
- Do not proceed until causal behavior is directly proven by a test.

## 6. Chapter 4 - Implementing a GPT Model from Scratch

Use the book's GPT-2 small/124M configuration to learn architecture and tensor shapes. Instantiating it does not mean this project has pretrained 124 million parameters.

### 4.1: Code the architecture scaffold

- Define the book-equivalent configuration dictionary.
- Implement the dummy GPT model used to expose the data flow.
- Pass token IDs through token embeddings, positional embeddings, placeholder blocks, normalization, and the output head.
- Confirm logits have shape `(batch, sequence, vocabulary_size)`.

### 4.2: Layer normalization

- Calculate mean and variance on a small example.
- Implement the chapter's layer-normalization module.
- Compare with a trusted PyTorch calculation within a stated tolerance.

### 4.3: GELU and feed-forward network

- Implement the GELU approximation used in the book.
- Plot or tabulate GELU against ReLU for understanding.
- Implement the feed-forward network with its expansion and contraction layers.
- Test that sequence and embedding dimensions are preserved.

### 4.4: Shortcut connections

- Reproduce the deep-network gradient example with and without shortcuts.
- Inspect gradient magnitudes rather than merely asserting that execution succeeds.
- Explain how residual paths help gradient flow.

### 4.5: Transformer block

- Combine pre-normalization, causal multi-head attention, dropout, residual paths, and the feed-forward network.
- Verify input and output shapes are identical.
- Run deterministic forward and backward tests with dropout disabled.

### 4.6: GPT model

- Replace dummy blocks with repeated Transformer blocks.
- Add final normalization and vocabulary projection.
- Count parameters exactly and explain the embedding/output-head parameter treatment.
- Estimate parameter memory for the selected dtype.

### 4.7: Generate text

- Implement the book's simple greedy generation loop.
- Crop input context to the supported context length.
- Convert between text and token IDs.
- Test prompt preservation, output length, batch handling, and context cropping.

### Chapter checkpoint

- Run raw prompt -> token IDs -> logits -> generated IDs -> decoded text.
- Complete a forward and backward pass on a small configuration.
- Explain why an untrained model produces structurally valid but meaningless output.
- Pass component and full-model integration tests.

## 7. Chapter 5 - Pretraining on Unlabeled Data

This chapter has two distinct outcomes: train the implementation on the chapter-sized corpus, then load published GPT-2 weights into the same architecture. Label the two weight sources clearly.

### 5.1: Evaluate generative text models

- Reuse the Chapter 4 generator.
- Convert next-token logits and targets into cross-entropy loss.
- Implement loss calculation for a batch and a data loader.
- Split "The Verdict" into training and validation portions as the book demonstrates.
- Calculate training and validation loss before training.

Required tests:

- flattened logits and targets contain the expected token positions;
- loss matches a hand-checked tiny example;
- evaluation runs in evaluation mode without gradients;
- training and validation windows do not cross the chosen split.

### 5.2: Train an LLM

- Implement the book's simple training loop in an inspectable form.
- Track tokens seen, training loss, validation loss, and generated samples.
- Run a tiny-batch overfit check before the chapter-sized run.
- Train on "The Verdict" using the chapter configuration that is practical on the available device; record any deliberate reduction.
- Plot or store the loss curves and inspect sample text during training.

### 5.3: Control decoding randomness

- Implement temperature scaling.
- Implement top-k filtering.
- Extend generation with seeded multinomial sampling and optional end-token stopping.
- Test greedy determinism, seeded repeatability, controlled-logit top-k behavior, and invalid temperature handling.

### 5.4: Load and save model weights

- Save and load model and optimizer state dictionaries.
- Include the model config, tokenizer identity, seed, training position, and dataset metadata in project checkpoints.
- Verify that reloaded weights reproduce logits and evaluation loss.

### 5.5: Load pretrained OpenAI weights

- Acquire GPT-2 weights only from the source used or referenced by the book, recording provenance and licence/usage notes.
- Implement explicit parameter assignment into the Chapter 4 GPT model.
- Validate every source and destination tensor shape.
- Generate with loaded GPT-2 weights and keep these results separate from the weights trained on "The Verdict."

### Chapter checkpoint

- The small from-scratch run lowers training loss and has recorded validation loss.
- A tiny batch can be deliberately overfit.
- Sampling controls behave as intended.
- A saved model reloads without changing its output.
- GPT-2 weights load through explicit, validated assignments.
- The run manifest records config, data, seed, device, losses, time, and conclusion.

## 8. Chapter 6 - Fine-Tuning for Classification

Follow sections 6.1 through 6.8 using the SMS spam classification task from the book. Record the dataset source, licence/usage conditions, version, and split construction.

### 6.1-6.3: Task, dataset, and loaders

- Distinguish classification fine-tuning from instruction fine-tuning.
- Obtain the SMS spam data through the book's documented path.
- Balance classes exactly as the chapter describes.
- Create deterministic train, validation, and test splits.
- Implement the dataset and loaders, including tokenization, padding, and truncation.
- Test split non-overlap, class counts, sequence lengths, and labels.

### 6.4-6.5: Pretrained model and classification head

- Initialize the Chapter 4 architecture with Chapter 5 pretrained GPT-2 weights.
- Replace the output head with the classification head.
- Freeze and selectively unfreeze parameters in the sequence taught by the book.
- Verify exactly which parameters require gradients.

### 6.6-6.7: Metrics and fine-tuning

- Implement classification loss and accuracy.
- Verify metrics with a hand-checked example.
- Fine-tune using the book's training process.
- Track training/validation loss and accuracy; evaluate the test split only at the appropriate final point.

### 6.8: Spam classifier

- Implement a helper that classifies new text consistently with training-time padding and truncation.
- Save and reload the fine-tuned model.
- Inspect representative correct and incorrect predictions.

### Chapter checkpoint

- The classifier beats the relevant simple baseline on held-out data.
- Frozen weights remain unchanged.
- Saved and reloaded models give the same predictions.
- Results include split details, accuracy, loss, runtime, and observed errors.

## 9. Chapter 7 - Fine-Tuning to Follow Instructions

Follow sections 7.1 through 7.9 using the instruction dataset supplied for the book. Record its source and usage conditions before use.

### 7.1-7.2: Instruction task and dataset

- Learn the difference between instruction, optional input/context, and response.
- Load and inspect the book's `instruction-data.json`-style records.
- Implement the chapter's prompt-formatting function.
- Create deterministic train, validation, and test splits.
- Test formatting with and without an optional input field.

### 7.3-7.4: Batches and data loaders

- Implement the custom collate function in the same progression as the book.
- Add end-of-text tokens, padding, target shifting, truncation, and ignored loss positions one step at a time.
- Inspect at least one batch token by token.
- Verify that padding positions excluded by the design contribute no loss.

### 7.5: Load a pretrained LLM

- Load the pretrained GPT-2 model size selected by the book, using the explicit Chapter 5 assignment path.
- The book uses GPT-2 medium/355M for its main run; use it when resources permit.
- If a smaller GPT-2 size is used, record the resource reason and do not present the result as an exact reproduction.

### 7.6: Fine-tune on instruction data

- Evaluate initial training and validation loss.
- Fine-tune with the chapter's next-token objective and masking behavior.
- Record loss, tokens seen, device, peak memory, runtime, and checkpoints.
- Keep the test split out of hyperparameter decisions.

### 7.7: Extract and save responses

- Generate responses for the test records.
- Extract the response portion consistently.
- Save structured responses separately from source data and model weights.
- Manually inspect a representative sample and record failure patterns.

### 7.8-7.9: Evaluate and conclude

- Follow the book's evaluation approach, including its caveats.
- If using a model-based judge, record the judge model, prompt, settings, and known limitations; do not treat a judge score as objective truth.
- Compare the base and instruction-tuned checkpoints on identical held-out prompts.
- Summarize what changed, what did not, and the limitations of supervised instruction fine-tuning.

### Chapter checkpoint

- Formatting, collation, target shifting, and loss masking have direct tests.
- The model completes a recorded fine-tuning run and generates saved test responses.
- Base and fine-tuned models are compared on the same prompts.
- The learner can explain every part of the instruction-data batch and loss.

## 10. Appendices

### Appendix A - Introduction to PyTorch

Use this before or during Chapters 2-5 as needed. Complete unfamiliar topics before depending on them: tensors, modules, autograd, optimizers, datasets/data loaders, and GPU execution. It is reference material, not a gate that must be reread in full by an experienced PyTorch user.

### Appendix B - References and further reading

Consult it when a chapter concept needs another explanation. Record useful references in the learning log rather than expanding the required implementation scope.

### Appendix C - Exercise solutions

Attempt each exercise first. Record the attempt, then use Appendix C to check the result. Correct misunderstandings and add a regression test when the mistake concerned program behavior.

### Appendix D - Improve the training loop

Only after the Chapter 5 baseline works:

- add linear learning-rate warmup;
- add cosine decay;
- add gradient clipping;
- compare the enhanced run with the baseline using the same model, data, seed, and evaluation method;
- change and verify one feature at a time.

Completion requires recorded learning-rate behavior, gradient norms, losses, and a conclusion about the comparison.

### Appendix E - Implement LoRA

Only after Chapter 7:

- implement the low-rank A and B matrices and scaling;
- wrap selected linear layers while freezing base parameters;
- verify zero-impact initialization;
- verify only intended LoRA parameters update;
- compare trainable parameter counts and fine-tuning results with the Chapter 7 path;
- save adapter weights with an identifier for the required base checkpoint.

Completion requires tests for initialization, gradients, parameter freezing, and adapter save/load behavior.

## 11. Repository Evolution

Create files when the corresponding chapter reaches them. Do not scaffold later chapters in advance.

```text
SparkLM/
|-- AGENTS.md
|-- README.md
|-- IMPLEMENTATION_PLAN.md
|-- LEARNING_LOG.md
|-- pyproject.toml
|-- essentials/                  # setup records and data policy
|-- data/                        # local/ignored except approved metadata
|-- experiments/                 # manifests and small result summaries
|-- checkpoints/                 # ignored
|-- src/sparklm/
|   |-- tokenization.py          # Chapter 2
|   |-- data.py                  # Chapters 2, 6, and 7
|   |-- attention.py             # Chapter 3
|   |-- layers.py                # Chapter 4
|   |-- model.py                 # Chapter 4
|   |-- generation.py            # Chapters 4 and 5
|   |-- training.py              # Chapter 5
|   |-- gpt2_weights.py          # Chapter 5
|   |-- classification.py        # Chapter 6
|   |-- instruction_tuning.py    # Chapter 7
|   `-- lora.py                  # Appendix E
|-- tests/
`-- scripts/                     # chapter entry points as needed
```

Existing files do not receive credit automatically. When a chapter reaches an existing implementation, either rebuild it or validate it line by line against the book, then add the chapter's tests and learning note.

## 12. Experiment Record

Each meaningful training or comparison run should record:

```yaml
book_section: "5.2"
question: "What is this run testing?"
hypothesis: "What result is expected, and why?"
code_version: "commit or explicit dirty-worktree note"
config: "resolved model and training settings"
dataset: "source, version/hash, split, and preprocessing record"
tokenizer: "name/version and special-token behavior"
seed: 42
device: "hardware, PyTorch, and CUDA details"
precision: "float32/bfloat16/etc."
model: "architecture, weight source, and exact parameter count"
training: "optimizer, LR, batch size, tokens, and steps"
metrics: "training/validation loss and relevant task metrics"
performance: "runtime, throughput, and peak memory"
artifacts: "local checkpoint and result paths"
result: "observations and failures"
conclusion: "whether the hypothesis was supported"
```

## 13. Completion Criteria

The book path is complete only when:

- Chapters 1-7 have been followed in order and their checkpoints pass;
- the learner completed the exercises before consulting solutions;
- text can be traced through tokenization, batching, embeddings, attention, GPT blocks, logits, loss, optimization, and generation;
- the Chapter 5 model was trained on the small chapter corpus and separately loaded with published GPT-2 weights;
- classification and instruction fine-tuning were completed and evaluated;
- Appendix D improvements and Appendix E LoRA were implemented after the core path;
- direct tests prove target alignment, causal masking, tensor shapes, gradients, checkpoint reload, and instruction loss masking;
- datasets and experiments have reproducible records;
- the learner can explain the implementation without relying on a framework wrapper or reciting code from the book.

## 14. Immediate Next Step

Start at setup and Chapter 1 only:

1. Treat the current repository as unvalidated starting material.
2. Verify the Python/PyTorch/device setup and record the result.
3. Read Chapter 1 sections 1.1-1.7.
4. Write the Chapter 1 mental model in `LEARNING_LOG.md`.
5. Complete the Chapter 1 checkpoint.
6. Begin Chapter 2 only after that checkpoint is understood.

There is no deadline and no requirement to preserve the old implementation order. Progress is measured by completing the book's sequence with working code, tests, and written understanding.
