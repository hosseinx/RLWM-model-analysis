# Computational Modeling of Interactive RLWM with Simon Effect

This repository contains the computational modeling code for the **Interactive Reinforcement Learning and Working Memory (RLWM) task** integrated with the **Simon Effect**. 

The models implemented here aim to disentangle the contributions of reinforcement learning, working memory capacity, and cognitive control (response conflict) in human decision-making.

---

## 🧠 Model Overview

The computational framework is based on the interactive RLWM model ([Collins, 2018](#references)), extended to account for spatial response conflict (Simon effect). 

### Core Components:
1. **Reinforcement Learning (RL) Subsystem:** Updates stimulus-response values based on prediction errors (e.g., Rescorla-Wagner / Q-learning).
2. **Working Memory (WM) Subsystem:** Maintains and retrieves recent stimulus-response mappings, subject to capacity limits (Set Size 3 vs. 6).
3. **Interactive Gating:** A mechanism that determines the reliance on RL vs. WM based on cognitive load and environmental volatility.
4. **Simon Effect Integration:** A spatial conflict mechanism that modulates the decision process (e.g., by introducing a response bias in the choice rule or altering drift rates) when the stimulus location conflicts with the correct response side.

