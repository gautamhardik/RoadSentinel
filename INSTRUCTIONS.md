# Project Operating Instructions & Guidelines

> **IMPORTANT**: These instructions must be reviewed before planning and executing actions in this project. Keep this file updated with active constraints, preferences, and workflows.

---

## 1. File Lifecycle & Cleanup
- **Intermediate File Removal**: Always delete/clean up intermediate developed files, temporary artifacts, debug dumps, and one-off helper scripts once their intended task or usage has completed (e.g. temporary download zips once extracted, one-off inspection scripts, temporary debug media).

---

## 2. Code & Architecture Standards
- Adhere to the **Ponytail** principle: prioritize minimal diffs, zero unnecessary abstractions, and YAGNI (You Ain't Gonna Need It).
- Never introduce bloatware or unrequested dependencies when existing tools, stdlib, or native features suffice.

---

## 3. Workflow & Tooling Preferences
- **Notebooks over Scripts**: Prefer Jupyter Notebooks (`.ipynb`) over standalone `.py` scripts for experiments, data exploration, visualizations, model inference, and analysis pipelines whenever interactive execution, rich outputs, or step-by-step transparency is beneficial.

---

## 4. Reporting & Walkthroughs
- **Post-Task Walkthrough**: After every major task or phase is completed, provide a detailed walkthrough at the end of the response summarizing exact outputs, created artifacts, validation results, and next actions.

---

## 5. Competition Context & Rules
- **Stack Target**: OpenCV 5 + AWS (Graviton / COOL acceleration).
- Maintain reproducible setups, pinned dependencies, and clean directory structure.

---
*(Add new instructions below as requirements evolve)*


