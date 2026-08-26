---
name: sparklm-transformer-correctness
description: Implement and verify SparkLM's tokenizer, causal self-attention, multi-head attention, Transformer blocks, logits, and generation from first principles. Use when adding or reviewing core model components, tensor operations, causal masking, numerical reference comparisons, or shape and gradient tests.
---

# SparkLM Transformer Correctness

Read `AGENTS.md` and the relevant phase in `IMPLEMENTATION_PLAN.md` before editing core model code. Keep the learning implementation explicit and inspectable.

## Implementation order

1. Establish tensor shapes and a minimal deterministic example.
2. Implement the simplest correct operation: token lookup, shifted targets, Q/K/V projections, scaled scores, causal mask, softmax, value aggregation, then multi-head recombination.
3. Add residual connections, normalisation, feed-forward layers, positional information, logits, and generation one concept at a time.
4. Add dropout or performance optimisations only after the unregularised path is correct.
5. Compare against a trusted PyTorch reference only after direct tests exist for the manual implementation.

## Required checks

- Assert batch, sequence, vocabulary, embedding, head, and projection dimensions at meaningful boundaries.
- Test tokenizer round trips and input/target alignment.
- Test that the causal mask prevents future positions from affecting earlier outputs.
- Test attention outputs and gradients against a reference within an explicitly chosen tolerance.
- Test a tiny corpus or batch can overfit deliberately.
- Test generation respects maximum length, temperature, and reproducible seeds.
- Test checkpoint reload preserves evaluation results when checkpointing is implemented.

## Restrictions

- Do not use `torch.nn.Transformer` or `AutoModelForCausalLM` as the primary early implementation.
- Do not hide the attention equations behind an abstraction before the equations and intermediate tensors are documented.
- Keep the character tokenizer as a working baseline after adding a subword tokenizer.
- Explain any numerical tolerance, masking convention, normalisation placement, or architectural deviation in a learning note.
