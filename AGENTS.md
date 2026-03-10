# AGENTS.md
Guidelines for Codex when assisting with this repository.

---

# Project Mission

Build a **domain-specific chatbot for coordinated supply chain optimization** designed for **classroom use by advanced undergraduate students**.

The chatbot should support **progressive inquiry** similar to ChatGPT but specialized for the coordinated supply chain framework.

Capabilities include:

• interpreting supply chain problems from natural language  
• extracting entities (suppliers, consumers, nodes, technologies)  
• constructing a mathematical model  
• generating LaTeX formulations  
• solving the model using Pyomo  
• identifying missing parameters  
• checking theoretical assumptions  
• performing what-if analysis  

The system should **not rely on model fine-tuning**.

Instead it should use:

• structured problem schemas  
• rule-based logic  
• retrieval of knowledge sources  
• deterministic code generation  
• optimization solvers

---

# Development Philosophy

This system must follow a **hybrid architecture**:

Language interface → structured model → solver → explanation.

The LLM is **not the mathematical authority**.

All final mathematical results must come from:

• explicit model generation  
• deterministic validation  
• numerical solving.

---

# Architecture Overview

The project contains the following logical modules:

Dialogue Layer  
Handles user conversation and intent classification.

State Manager  
Maintains the evolving supply chain problem state.

Domain Parser  
Extracts structured information from user input.

Schema Module  
Defines canonical supply chain model structures.

Model Builder  
Creates the mathematical model from the schema.

Latex Generator  
Produces formal mathematical formulations.

Solver Engine  
Instantiates and solves the optimization model.

Theorem Checker  
Evaluates whether theoretical assumptions apply.

Scenario Engine  
Runs what-if analyses.

---

# Implementation Order

Codex should follow this implementation sequence:

1. Repository structure
2. Canonical schema
3. Problem state manager
4. LaTeX generator
5. Pyomo model builder
6. Solver interface
7. Theorem applicability checker
8. Scenario analysis module
9. Colab demonstration notebook

Do not skip steps.

---

# File Editing Policy

Codex may modify:

src/
tests/
notebooks/

Codex should NOT rewrite:

docs/project_guide.md
AGENTS.md
README.md

unless explicitly instructed.

---

# Mathematical Safety Rules

Codex must follow these rules:

1. Do not fabricate parameter values.
2. Do not claim a model was solved unless the solver ran.
3. Do not claim theorem applicability without checking assumptions.
4. Always distinguish between:

• user-provided data  
• assumed values  
• sourced values

5. If parameters are missing, request them instead of guessing.

---

# Code Style Guidelines

Python version: 3.10+

Use:

• type hints when appropriate
• modular design
• clear docstrings
• deterministic transformations

Avoid:

• hard-coding domain assumptions
• mixing symbolic and numerical logic
• large monolithic functions

---

# Testing Policy

Each core module should include unit tests.

Add tests for:

schema validation  
model generation  
solver execution  
theorem checks

---

# Preferred Libraries

Python ecosystem:

pyomo  
pandas  
numpy  
pydantic  
networkx  
matplotlib  

Notebook development should be **Colab compatible**.

---

# Expected Outputs

The system must support:

• mathematical formulation in LaTeX  
• symbolic model representation  
• solver results  
• explanation of results  
• theorem applicability statements  
• scenario comparison

---

# Classroom Mode

The chatbot should support:

Hint Mode  
Guided Mode  
Full Solution Mode

Do not always give the final answer immediately.

---

# Instructions for Codex

When implementing code:

• propose modular architecture  
• prefer small testable components  
• explain design decisions when helpful  
• maintain readability for teaching purposes  

Favor **clarity and correctness over clever shortcuts**.

---

# End of AGENTS.md

Describe the agents (roles, responsibilities, coordination) for the chatbot here.
