# HRFunc Refactoring Plan — Index

**Status:** This document was decomposed on 2026-04-14 into three cohesive plans. Use the documents below instead of this one.

---

## Active Plans

- **[phase_breakdown.md](phase_breakdown.md)** — Release roadmap. High-level v1.2.0 → v2.0.0 strategy, branch dependency chains, status of completed work, review protocol, release testing gates.

- **[plan_v1_2_0.md](plan_v1_2_0.md)** — Detailed correctness-release plan. Branch-by-branch fixes scoped to 1.2.0 (crashes, silent wrong results, input validation, packaging). Target: PyPI release.

- **[plan_v2_0_0.md](plan_v2_0_0.md)** — Detailed architectural-refactor plan. Branch-by-branch work scoped to 2.0.0 (composition refactor, performance, type hints, logging, test suite restructure). Ships after 1.2.0 validates in production.

---

## Supporting Documents

- **[../internal/known_issues.md](../internal/known_issues.md)** — Authoritative issue catalogue with file:line refs. Contains KI-001..KI-034, NE-001..NE-007, ND-001..ND-006 plus newer findings (S1-S4, H1-H4, M1-M6, L1-L3) from the 2026-04-14 parallel review.

- **[../internal/architecture.md](../internal/architecture.md)** — Module map, class responsibilities, execution paths, data pipeline.

---

## Why This Decomposition?

The original monolithic `refactoring_plan.md` was 957 lines of code snippets frozen at the 2026-04-13 state of the codebase. Since then:

- Four branches have landed on origin (phase1a, phase1bc, circular-imports-phase2, estimate-activity-threading)
- Parallel agent reviews uncovered new issue classes (scientific correctness, state lifecycle, input validation) not in the original plan
- Denny resolved several open questions (timeout=30s, poly order=3, S2/S3 are intentional a.u. conventions)
- Release strategy was refined to split correctness (1.2.0) from architectural refactor (2.0.0)

Rather than keep editing a single 1000-line document that mixes completed and pending work, the plan is now split by release target with each document focused on one release's scope.

---

## Historical Note

The original `refactoring_plan.md` content (including detailed code snippets for Phases 1 through 12) is preserved in git history at commit `bb388a5` (`fix: phase 1b+1c — pipeline correctness, hash stability, context scoring`) and earlier. Pull it from git if you need the old snippet-level detail.