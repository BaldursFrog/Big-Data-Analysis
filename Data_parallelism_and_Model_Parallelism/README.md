# Distributed Training Ecosystems – Spark, PyTorch, Ray

This repository contains theoretical questions and system design tasks focused on distributed deep learning, including Spark, PyTorch DDP, TorchDistributor, Ray Train, and Ray Tune.

The assignment evaluates understanding of:

- Core distributed training concepts  
- Framework-level architecture (PyTorch, Spark, Ray)  
- End-to-end pipeline design  
- Architecture selection and system analysis  

---

# Part 1: Basic Concept Questions  
*Assessing understanding of core terminology and principles*

1. Explain the difference between **Data Parallelism** and **Model Parallelism**, and provide a suitable use case for each.

2. What is the **All-Reduce** mechanism? What advantages does it offer compared to the **Parameter Server** architecture?

3. How does **PyTorch Distributed Data Parallel (DDP)** implement distributed training? Briefly describe its workflow.

4. What are the core components of **Ray**? Briefly describe the role of the **Object Store**.

5. What are the main limitations of **Spark MLlib**? Why is it not suitable for deep learning training directly?

6. What role does **TorchDistributor** play in Spark? What problem does it solve?

---

# Part 2: Framework and Application Questions  
*Assessing understanding and application ability of frameworks like Ray and PyTorch on Spark*

7. Describe Ray's programming model and provide examples of how `@ray.remote` and Actors are used.

8. Draw a schematic diagram of the workflow for launching PyTorch DDP training on Spark using TorchDistributor, and briefly explain each step.

9. What tasks are **Ray Train** and **Ray Tune** used for respectively? How do they work together?

10. Compare the advantages and disadvantages of the following three distributed training solutions:

- Pure PyTorch DDP  
- PyTorch on Spark (TorchDistributor)  
- Ray Train  

---

# Part 3: Comprehensive Design and Analysis Questions  
*(15 points each, Total: 30 points)*  
*Assessing system design capability and in-depth understanding of the distributed training ecosystem*

11. Assume you have a dataset of **100 million text samples** for a classification task, using the **BERT-large** model.

Design an end-to-end training pipeline based on:

- Spark  
- PyTorch  
- TorchDistributor  

Explain the key steps:

- Data loading  
- Preprocessing  
- Distributed training  
- Model saving  

12. Analyze which distributed training architecture should be chosen for the following scenarios and justify your reasoning:

- **Scenario A:** Training a Vision Transformer model with multiple machines and GPUs, medium data volume.  

- **Scenario B:** Implementing an integrated pipeline from ETL to training on an existing Spark cluster.  

- **Scenario C:** A research project requiring frequent hyperparameter tuning and model deployment.  

---

# Part 4  
*Assessing system design capability and in-depth understanding of the distributed training ecosystem*

13. Assume you have a text classification dataset (size flexible) and want to train a Transformer model (e.g., DistilBERT).

Design an end-to-end distributed training pipeline using one of the following stacks (or another justified alternative):

- Spark + PyTorch + TorchDistributor  
- Hadoop + TensorFlow + Horovod  
- Other distributed combinations  

Explain the key steps:

- Data loading  
- Preprocessing  
- Distributed training  
- Model saving  

14. Analyze which distributed training architecture should be chosen for the following scenarios and justify your reasoning:

- **Scenario A:** Training a Vision Transformer model with multiple machines and GPUs, medium data volume.  

- **Scenario B:** Implementing an integrated pipeline from ETL to training on an existing distributed data cluster (e.g., Spark or Hadoop).  

- **Scenario C:** A research project requiring frequent hyperparameter tuning and model deployment.  

---
