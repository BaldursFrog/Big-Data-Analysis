# AI Agents and AutoGPT Experiment

## Overview

This laboratory work explores the concept of **AI Agents** and their capabilities compared to traditional Large Language Models (LLMs). The project investigates how autonomous agents such as **AutoGPT** can plan, execute, and evaluate complex tasks through iterative workflows.

The work includes:

* analysis of the AI Agent workflow
* comparison between traditional LLMs and AI Agents
* simulation of AutoGPT performing a data analysis task
* practical experiment using the Titanic dataset

The goal of this lab is to understand how **agent-based systems extend the capabilities of standard language models** by introducing planning, memory, tool usage, and self-reflection.

---

# Laboratory Tasks

The assignment consists of four main tasks. 

### Task 1 — Mind Map of Lesson Content

A conceptual **mind map** describing the main ideas of the lesson:

Key topics:

* AI Agent workflow
* task generation
* task execution
* self-reflection
* comparison between traditional LLMs and agents
* AutoGPT experiments

The mind map demonstrates how these concepts are interconnected and how AI agents enable complex autonomous problem solving.

---

# Task 2 — AI Agent Workflow

The workflow of an AI Agent consists of three core stages:

1. **Task Generation**
2. **Task Execution**
3. **Self-Reflection**

### AutoGPT Workflow Diagram

```
User Goal
   ↓
Task Generation
   ↓
Task Execution
   ↓
Self-Reflection
   ↓
Goal Achieved?
   ↓
Repeat if necessary
```

### Stage Description

**Task Generation**

The agent analyzes the user’s objective and decomposes it into smaller tasks.

Example steps:

* analyze user goal
* identify required operations
* prioritize subtasks
* create task queue

---

**Task Execution**

The agent executes the generated tasks.

Typical operations:

* selecting the next task
* choosing tools or algorithms
* executing data processing or analysis
* storing intermediate results

---

**Self-Reflection**

The agent evaluates the results of completed tasks.

Functions:

* quality assessment
* error detection
* strategy improvement
* task list update

This loop allows the system to **adapt and improve its performance autonomously**.

---

# Task 3 — Traditional LLMs vs AI Agents

Traditional LLMs and AI Agents differ significantly in how they complete tasks.

| Feature     | Traditional LLM   | AI Agent             |
| ----------- | ----------------- | -------------------- |
| Interaction | Single response   | Multi-step workflow  |
| Memory      | Limited context   | Persistent memory    |
| Planning    | None              | Autonomous planning  |
| Tool usage  | No external tools | Uses APIs and tools  |
| Autonomy    | Passive           | Autonomous execution |

### Advantages of AI Agents

AI agents perform better in scenarios such as:

* complex data analysis
* research automation
* workflow automation
* software development tasks
* multi-step problem solving

These systems can **plan tasks, execute actions, and evaluate results without constant human input**.

---

# Task 4 — AutoGPT Experiment

## Objective

Simulate how **AutoGPT performs a data analysis task** using the Titanic dataset.

Goal provided to AutoGPT:

> "Please analyze this dataset and generate a report containing key statistical information and visualizations."

---

# Dataset

The experiment uses a **synthetic Titanic dataset** containing passenger information such as:

* passenger ID
* survival status
* passenger class
* gender
* age
* ticket fare
* embarkation port

The dataset contains approximately **891 passengers and multiple features** for analysis. 

---

# Data Analysis Process

The simulated AutoGPT workflow performs the following steps:

### 1 Data Loading

The dataset is generated and loaded using Python and pandas.

### 2 Data Cleaning

Preprocessing operations include:

* filling missing values
* feature engineering
* creation of new variables

Examples:

* AgeGroup
* FamilySize
* HasCabin

### 3 Statistical Analysis

Key statistics calculated:

* overall survival rate
* survival by gender
* survival by passenger class
* survival by age group
* survival by embarkation port

### 4 Data Visualization

The system automatically generates visualizations including:

* survival rate charts
* passenger class comparisons
* age distributions
* correlation heatmaps
* family size analysis

The visualizations are saved as:

```
titanic_analysis_visualizations.png
titanic_detailed_analysis.png
```

### 5 Report Generation

The program automatically generates a structured report:

```
titanic_analysis_report.md
```

The report summarizes:

* dataset overview
* key statistical findings
* survival patterns
* conclusions and recommendations

---

# Implementation

The experiment was implemented in **Python** using the following libraries:

* pandas
* numpy
* matplotlib
* seaborn

The main analysis script is:

```
titanic_analysis.py
```

It performs the full workflow:

```
Dataset creation
→ data exploration
→ preprocessing
→ statistical analysis
→ visualization
→ report generation
```

---

# Example Findings

The analysis identified several key patterns:

* female passengers had significantly higher survival rates
* first-class passengers survived more frequently
* passengers with higher fares had better survival chances
* small family groups had slightly higher survival rates

These results reflect historical evacuation priorities during the Titanic disaster.

---

# Conclusion

This laboratory work demonstrates how **AI Agents extend the capabilities of traditional LLM systems** by enabling autonomous task execution.

The AutoGPT simulation showed that agents can:

* decompose complex tasks
* perform multi-step analysis
* generate visualizations
* produce structured reports

Although AI agents significantly improve efficiency, human analysts still play an important role in interpretation, contextual understanding, and validation of results.
