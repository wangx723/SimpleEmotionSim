**README for `SimpleEmotionSim.py`**

**Supplementary Code for**  
**“Functionalist Emotion Modeling in Biomimetic Reinforcement Learning”**  

---

### Purpose
This script implements the minimal **4-concept toy simulation** described in the paper (Section V.H “Toy Simulation”). It demonstrates the core functionalist mechanics of the model in a simple 4-node world:

- **Food** (node 0) – positive Primary Utility Function (P-UF)  
- **Home** (node 1) – learned Secondary Utility Function (S-UF)  
- **Intruder** (node 2) – learned S-UF  
- **Pain** (node 3) – negative P-UF  

The simulation reproduces the three-phase scenario used in the paper’s Fig. 1:  
1. Safe eating (food + home active)  
2. Intruder attack (home + intruder + pain active)  
3. Recovery (food + home active)

It showcases the paper’s key formulations:
- Eligibility-trace P-UF → S-UF valence mapping (Eqs. 2–3)  
- Lossy valence diffusion among co-active concepts (Eqs. 6–8)  
- Depletion without P-UF reinforcement  
- Simplified anger dynamics (global + local anger, reduced diffusion, limited valence recovery; Eqs. 11–29)

### Requirements
```bash
python 3.8+
numpy
matplotlib
```

### Quick Start
```bash
python SimpleEmotionSim.py
```

**Outputs** (automatically generated):
- `simple_emotion_sim_output.png` – UV valence trajectories + global anger (matches paper Fig. 1)  
- `simple_emotion_sim_anger_output.png` – local anger scalars + global/active anger  
- Console summary

### Main Limitations
1. **Parameter tuning for visibility**: Values were deliberately accelerated for a compressed time-scale of 75 steps. In a full-scale biological model for example, EMA baseline tracking and recovery would lag significantly behind current UV.

2. **Local anger persistence in phase 3**: In this minimal fully-connected 4-node network, local anger equalizes between “home” and “food” and does not continue to deplete via transfer losses (η). In a biologically realistic network with sparse, dynamic activation patterns, local anger would eventually decay to zero.

3. **No decision mechanism**: The script contains only the utility-function and anger modules. Planning, action selection, attention, appraisal, and persistent P-UF modulation (e.g., hunger-state modifiers on effective UV) are omitted to keep the demonstration focused on P-UF/S-UF/anger dynamics.

The simulation intentionally avoids additional complexity (bifurcated concepts, Markov decision states, etc.) so that readers can clearly see the core functionalist components introduced in the paper.

### Correspondence to Paper
- Adjacency matrix is fully connected.  
- All equations implemented match the formulations section.  
- Plots are direct reproductions of the paper’s Fig. 1.

---

**Repository note**: This file is provided as supplementary material. For the latest version or extensions, see the public repository referenced in the paper ([29]).
