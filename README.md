# Computational Modeling of Interactive RLWM with Simon Effect

This repository contains the computational modeling framework and Python implementations for the **Interactive RLWM task integrated with the Simon Effect**. 

The models implemented here aim to disentangle the contributions of reinforcement learning (RL), working memory (WM), and cognitive control (response conflict) in human decision-making.

---

## 🧠 Implemented Models

We implemented and compared three computational models to explain the behavioral data:

### 1. Classic Reinforcement Learning (Classic RL)
A baseline model that relies solely on a standard Q-learning mechanism. It does not include a working memory system, forgetting, or conflict modulation. It serves as the baseline to evaluate the necessity of WM and conflict mechanisms.
* **Mechanism:** Standard Rescorla-Wagner / Q-learning.
* **Parameters:** Learning rate ($\alpha$), Softmax temperature ($\beta$).

### 2. Interacting RL and WM (RLWMi)
Based on the interactive model proposed by [Collins (2018)](#references). This model posits two interacting systems: a slow, incremental RL system and a fast, capacity-limited WM system. 
* **Mechanism:** The RL system's prediction error is modulated by the WM system's expectations (coupled). The final choice policy is a mixture of the softmax outputs of both systems.
* **WM Load Integration:** The reliance on WM ($\eta$) is parameterized separately for Low Load (Set Size 3) and High Load (Set Size 6).
* **Parameters:** $\alpha, \beta, \eta_3, \eta_6, \phi$ (forgetting), $pers$ (perseveration), $\epsilon$ (lapse rate), $init$ (initial bias).

### 3. Conflict-Modulated RLWMi (RLWM-C) 🌟
**This is our novel extension.** It inherits all mechanisms from the RLWMi model but introduces a **conflict-based learning rate modulation** to explicitly capture the Simon Effect.
* **Mechanism:** Instead of a single learning rate, the RL system uses two distinct learning rates depending on the spatial congruency of the trial. This allows the model to test whether response conflict (incongruent trials) specifically impairs or enhances reinforcement learning.
* **Parameters:** Inherits all RLWMi parameters, but replaces the single $\alpha$ with $\alpha_{congruent}$ and $\alpha_{incongruent}$.

---

## 📐 Mathematical Formulation

Below are the core equations derived from the Python implementation:

### 1. Value Updates (RLWMi & RLWM-C)

**Working Memory (Fast learning):**
The WM system simply stores the outcome of the most recent trial for a given stimulus-action pair:

$$
W_{t+1}(s, a) = R_t
$$

**Reinforcement Learning (Incremental learning):**

$$
Q_{t+1}(s, a) = Q_t(s, a) + \alpha \cdot \delta_t
$$

*Note: In **RLWM-C**, $\alpha$ is dynamically selected based on congruency:*

$$
\alpha = 
\begin{cases} 
\alpha_{congruent} & \text{if trial is congruent} \\ 
\alpha_{incongruent} & \text{if trial is incongruent} 
\end{cases}
$$

### 2. Prediction Error (Coupled RLWMi)

In the interacting model, the RL prediction error ($\delta$) is calculated against a weighted combination of RL and WM expectations:

$$
\delta_t = R_t - \Big[ (1 - \eta_{bs}) \cdot Q_t(s, a) + \eta_{bs} \cdot W_t(s, a) \Big]
$$

*(Where $\eta_{bs}$ is $\eta_3$ for set size 3, and $\eta_6$ for set size 6).*

### 3. Choice Policy (Softmax Mixture)

The probability of choosing an action is a mixture of the softmax policies of the RL and WM systems:

$$
P(a|s) = \eta_{bs} \cdot \text{softmax}(W, \beta) + (1 - \eta_{bs}) \cdot \text{softmax}(Q, \beta)
$$

### 4. Auxiliary Mechanisms

* **Forgetting/Decay ($\phi$):** Values decay towards their initial state before each trial update:
  $$
  Q_{t}(s, a) = (1 - \phi) \cdot Q_{t-1}(s, a) + \phi \cdot Q_{init}
  $$
* **Perseveration ($pers$):** If the prediction error is negative ($\delta < 0$), the learning rate is reduced:
  $$
  \alpha_{adjusted} = \alpha \cdot (1 - pers)
  $$
* **Lapse Rate / Undirected Noise ($\epsilon$):** A proportion of trials are chosen uniformly at random.
* **Initial Bias ($init$):** Upon the first encounter with a stimulus, the chosen action's Q-value is boosted:
  $$
  Q_{init\_updated}(s, a) = Q_{init} + init \cdot (1 - Q_{init})
  $$
---

## 🛠️ Code Structure & Usage

The models are implemented in `rlwm/models_collins.py` using an Object-Oriented approach with factory functions for easy instantiation.

### Instantiating the Models

```python
from src.models_collins import model_classic, model_rlwmi, model_rlwmi_c

# 1. Classic RL
rl_model = model_classic(learning_rate=0.1, beta=3.0)

# 2. Interacting RLWM (RLWMi)
rlwmi_model = model_rlwmi(
    learning_rate=0.1, beta=3.0, decay=0.0, pers=0.1, 
    eps=0.05, init=0.5, eta3_wm=0.8, eta6_wm=0.3
)

# 3. Conflict-Modulated RLWM (RLWM-C)
rlwmc_model = model_rlwmi_c(
    alpha_congruent=0.2, alpha_incongruent=0.05, beta=3.0, 
    decay=0.0, pers=0.1, eps=0.05, init=0.5, eta3_wm=0.8, eta6_wm=0.3
)
