# Main #583 Narrow setup-python Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Safely adopt `actions/setup-python` 7.1.0 in the seven workflows whose existing verification contracts permit the pin change, while preserving all artifact-action pins and all frozen/exact-source workflows.

**Architecture:** Start from current `main` at `f3adf2d86bd72f885bdba5d955b6ff73cee95a34`. Apply one mechanical SHA substitution in exactly seven workflow files. Guard the scope with an exact-pin control before and after editing, parse the YAML, run targeted and full Python verification in normal and optimized modes, and inspect the final diff before commit and Draft PR creation.

**Tech Stack:** YAML workflows, Python unittest, Ruby Psych, git

**Spec:** main#583/6389346888

## Global constraints

- Preserve every `actions/upload-artifact` and `actions/download-artifact` pin.
- Do not modify `.github/workflows/architecture-runtime-acquisition.yml`, `.github/workflows/reactor-v0-controls.yml`, or `.github/workflows/source-census.yml`.
- Do not modify the Dependabot branch or original PR head.
- Create only a Draft successor PR; do not mark Ready, rerun checks, waive gates, acquire the MAIN-LOCK, merge, deploy, or alter scientific status.

## Task 1: Apply the bounded setup-python pin update

**Files:**

- Modify: `.github/workflows/current-site.yml`
- Modify: `.github/workflows/formal-verification.yml`
- Modify: `.github/workflows/math-hypergraph.yml`
- Modify: `.github/workflows/navigation.yml`
- Modify: `.github/workflows/periodic-h0-pilot.yml`
- Modify: `.github/workflows/public-shop.yml`
- Modify: `.github/workflows/universality-controls.yml`
- Add: `docs/superpowers/plans/2026-10-24-main583-narrow-setup-python.md`

- [ ] Run an exact-pin control that expects the new setup-python SHA once in each included workflow, rejects the old SHA there, and confirms the excluded workflows and all artifact-action pins remain unchanged. Confirm the control fails before editing.
- [ ] Replace only `actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97` with the canonical v7.1.0 commit `actions/setup-python@6d4fd22b20465ccc0f7354476588e8df7c9ca0e0` in the seven included workflows.
- [ ] Re-run the exact-pin control and confirm it passes.
- [ ] Parse the changed workflows with Ruby Psych.
- [ ] Run the targeted verification suite in normal and optimized Python modes.
- [ ] Run the full Python test suite in normal and optimized modes.
- [ ] Run `git diff --check` and inspect the complete path/stat/content diff for scope and pin integrity.
- [ ] Commit the verified change on `codex/main583-setup-python-only-20261024`.

## Review Focus

- Any accidental `upload-artifact` or `download-artifact` change.
- Any change to the three explicitly excluded frozen/exact-source workflows.
- Incorrect old/new setup-python pin counts or an omitted included workflow.
- Invalid YAML after the SHA substitution.
- A verification result reported from the wrong source or without normal and optimized-mode coverage.
