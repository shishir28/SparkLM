---
name: sparklm-phase-zero
description: Establish SparkLM's Phase 0 foundations, including environment diagnostics, deterministic execution, data provenance, project rules, and the first smoke test. Use when starting the repository, documenting the DGX Spark environment, adding setup scripts, or preparing the project before model implementation.
---

# SparkLM Phase 0

Read `AGENTS.md`, `IMPLEMENTATION_PLAN.md`, and the files under `essentials/` before changing project foundations.

## Workflow

1. Inspect the current repository and preserve existing user changes.
2. Document Python, PyTorch, CUDA, GPU name and visibility, storage, memory, and relevant package versions in `essentials/ENVIRONMENT.md`.
3. Add or update a one-command smoke test that reports Python, PyTorch, CUDA, device name, and tensor-device status.
4. Make the smoke test deterministic and record the seed and observed result in `essentials/SMOKE_TEST.md`.
5. Document dataset source, licence or usage permission, date, privacy decision, and preprocessing in `essentials/DATA_POLICY.md` before introducing data.
6. Record project constraints and reproducibility conventions in `essentials/PROJECT_RULES.md` or `LEARNING_LOG.md`.

## Guardrails

- Do not download, commit, or expose private, tenant-identifying, unlicensed, or secret data.
- Keep generated data, checkpoints, model weights, and local environments covered by `.gitignore`.
- Prefer a small standard-library or PyTorch diagnostic over a framework-heavy setup layer.
- Verify the command from the repository root and report the exact verification performed.
