# Topic 9: Recommendation System Assignment  
## Comparative Study and Optimization of Traditional Collaborative Filtering vs. LLM-Enhanced Recommendation Systems

---

## 📌 Submission Requirements

- **Part 2:**  
  Submit an experimental report (following the provided experimental report template) + source code.

- **Other Parts:**  
  Submit a written answer report.

- **Word Count Requirement:**  
  No less than **2000 words**.

- **Deadline:**  
  Two weeks from the assignment release date.

---

# Part 1: Theoretical Analysis (40 points)

## 1. Comparative Analysis (20 points)

### 1.1 Comparative Table Analysis

Summarize and compare **Traditional Collaborative Filtering (CF)** and **LLM-Enhanced Recommendation Systems** across the following four aspects:

- Data Utilization Methods  
- Recommendation Logic  
- Interpretability  
- Cold-Start Problem  

The comparison must be presented in a **table format**, with detailed explanation.

---

### 1.2 LLM Paradigms vs Traditional CF

Based on the three LLM recommendation paradigms introduced in the PPT:

1. Embedding Generation  
2. Semantic Tag Generation  
3. End-to-End Generation  

Explain how each paradigm compensates for the shortcomings of traditional Collaborative Filtering.

Your explanation should clearly connect:

- The weaknesses of CF  
- The mechanism of each LLM paradigm  
- The improvement achieved  

---

## 2. Similarity Calculation and Matrix Factorization (20 points)

### 2.1 Similarity Measurement Methods

Explain the applicable scenarios and limitations of the following similarity measures:

- Jaccard Similarity  
- Cosine Similarity  
- Pearson Correlation  

For each method:

- Provide the formula  
- Describe when it is suitable  
- Discuss its limitations  

---

### 2.2 Role of SVD in Recommendation Systems

- Briefly explain the concept of **Singular Value Decomposition (SVD)**.  
- Describe its role in recommendation systems.  
- Explain how it helps alleviate the **data sparsity problem**.  

---

# Part 2: Algorithm Implementation & Programming (30 points)

## 1. User-Based Collaborative Filtering (15 points)

Implement a **User-Based Collaborative Filtering Algorithm** with the following requirements:

- Use the **MovieLens-100K dataset**.  
- Use **Cosine Similarity** to compute user similarity.  
- Recommend **Top-5 movies** for a specified user.  
- Output the recommendation results.  
- Briefly explain the reasoning behind the recommendations.

---

## 2. LLM-Based User Profile Tag Generation (15 points)

Use any open-source LLM (e.g., ChatGLM, LLaMA, etc.) to generate user interest descriptions based on historical behavior.

### Requirements:

- Input: User movie rating records  
- Output: User interest profile (e.g., "This user prefers sci-fi and suspense movies.")  

You may use:

- Few-shot prompting  
- Instruction tuning  
- Prompt engineering strategies  

Provide:

- Prompt design  
- Model usage explanation  
- Example output  

---

# Part 3: Case Design and Analysis (20 points)

Assume you are designing a recommendation system for **"Taobao"**.

---

## 1. Cold-Start Scenario Design (10 points)

Design a recommendation process specifically for **new users**, incorporating **two or more LLM training/usage methods**, such as:

- Prompting  
- Instruction tuning  
- Fine-tuning  
- Embedding-based retrieval  

Explain how you would utilize:

- User registration information  
- Social network data  
- Browsing logs  

to initialize recommendations.

---

## 2. Interpretability Enhancement Scheme (10 points)

Design a recommendation explanation generation module using an LLM.

Example explanation:

> "We recommend 'The Wandering Earth' because you like sci-fi movies and have given them high ratings."

Compare:

- LLM-based personalized explanations  
- Traditional CF explanations (e.g., "Users similar to you also liked...")  

Discuss:

- Advantages  
- Limitations  
- Scalability considerations  

---

# Part 4: Frontier Thinking Questions (10 points)

1. Read the two papers mentioned at the end of the PPT (or relevant survey papers), and answer:

- Can LLMs completely replace traditional Collaborative Filtering? Why or why not?  
- What is the future trend of recommendation systems?

Choose and justify one of the following perspectives:

- "LLM as the main component, CF as auxiliary"  
- "CF as the main component, LLM-enhanced"  

Your answer should demonstrate critical thinking and understanding of system trade-offs.

---

# ⭐ Bonus Question (Optional – Extra 10 points)

Build an **end-to-end movie recommendation system** using any LLM (e.g., GPT, Wenxin Yiyan, etc.).

### Requirements:

- Input: User historical movie-watching record (text format)  
- Output: Recommended movie list + reasoning for each recommendation  

Submit:

- Source code  
- Screenshots of runtime output  
- Brief explanation of system workflow  

---

## 📊 Evaluation Overview

| Part | Content | Points |
|------|---------|--------|
| Part 1 | Theoretical Analysis | 40 |
| Part 2 | Algorithm Implementation | 30 |
| Part 3 | Case Design & Analysis | 20 |
| Part 4 | Frontier Thinking | 10 |
| Bonus | End-to-End LLM System | +10 |

---

## 🎯 Learning Objectives

- Understand traditional recommendation algorithms  
- Analyze the integration of LLMs in recommendation systems  
- Implement collaborative filtering  
- Explore hybrid recommendation architectures  
- Evaluate future trends in AI-driven recommendation systems  

---
