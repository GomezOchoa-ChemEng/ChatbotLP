# Coordinated Supply Chain Optimization Chatbot

## Project Overview

This project builds a **domain-specific chatbot for coordinated supply chain optimization**.

The system allows users to progressively formulate and solve supply chain models through natural language interaction.

The chatbot is designed for **classroom use** in advanced undergraduate courses.

The system can:

• interpret supply chain problem descriptions  
• extract model entities  
• generate mathematical formulations  
• produce LaTeX equations  
• solve optimization models  
• explain results  
• analyze theoretical assumptions  
• run what-if scenarios

The architecture combines:

natural language interaction  
structured schemas  
optimization modeling  
symbolic reasoning.

---

# Key Features

### Progressive Inquiry

Users can interact step-by-step:

Example:

User: identify the stakeholders  
User: define the sets  
User: write the objective  
User: generate LaTeX  
User: solve the model

The chatbot remembers the evolving problem state.

---

### Mathematical Model Generation

The system can automatically generate optimization formulations including:

• decision variables  
• objective function  
• balance constraints  
• capacity constraints  
• domain restrictions

Output formats include:

LaTeX  
Pyomo code  
structured JSON representation.

---

### Optimization Solver

Once parameters are available, the system can:

• instantiate the optimization model  
• call a solver  
• return optimal allocations  
• explain economic interpretations.

---

### Missing Parameter Detection

The system identifies when parameters are missing and prompts the user for additional information.

It avoids guessing values.

---

### Theorem Applicability Checks

The system can evaluate whether theoretical results from the coordinated supply chain framework apply to the current model instance.

---

### What-If Analysis

Users can run scenario analyses by modifying parameters such as:

transport costs  
capacity limits  
technology availability.

---

# Project Architecture

The system contains the following components:

Dialogue Layer  
Handles conversation and intent detection.

State Manager  
Stores the evolving supply chain model.

Domain Parser  
Extracts structured information from user input.

Schema Module  
Defines canonical supply chain structures.

Latex Generator  
Creates mathematical formulations.

Model Builder  
Builds the optimization model.

Solver Engine  
Runs numerical optimization.

Theorem Checker  
Evaluates theoretical assumptions.

Scenario Engine  
Runs sensitivity and what-if analysis.

---

# Repository Structure

