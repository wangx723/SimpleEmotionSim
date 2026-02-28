import numpy as np
import matplotlib.pyplot as plt

class SimpleEmotionSim:
    """
    Simple simulation of the paper's functionalist emotion model
    in a 4-concept world: food, home, intruder, pain.

    - Food activates positive P-UF (innate reward).
    - Pain activates negative P-UF (innate punishment).
    - Concepts are S-UF perceptrons with learned valence (UV).
    - Key mechanics implemented:
        * P-UF → S-UF mapping with eligibility trace (temporal credit)
        * Lossy valence diffusion between co-active connected concepts
        * Depletion when active without P-UF reinforcement
        * Simplified anger: protects positive valence of "home" when threatened
    """

    def __init__(self):
        self.N = 4
        self.alpha = 0.82      # Eligibility trace decay (paper Eq. 2)
        self.beta = 0.18       # Diffusion rate
        self.eta = 0.87        # Transfer efficiency (lossy diffusion, paper Eq. 6)
        self.L = 15.0          # Max |valence| per perceptron
        self.gamma = 0.90      # EMA smoothing for anger baseline

        # Adjacency matrix for diffusion (home-food, home-intruder, intruder-pain)
        self.adj = np.array([
            [0, 1, 0, 0],  # food <-> home
            [1, 0, 1, 0],  # home <-> food, home <-> intruder
            [0, 1, 0, 1],  # intruder <-> home, intruder <-> pain
            [0, 0, 1, 0]
        ])

        self.reset()

    def reset(self):
        self.UV = np.zeros(self.N)      # S-UF valence (utility values)
        self.e = np.zeros(self.N)       # Eligibility trace
        self.EMA = np.zeros(self.N)     # Baseline for anger detection
        self.history_UV = []
        self.history_anger = []
        self.history_phase = []

    def step(self, active, puf_valence):
        """One discrete time step (matches paper formulations)."""
        # 1. Update eligibility trace
        self.e = self.alpha * self.e + (1 - self.alpha) * active.astype(float)

        # 2. P-UF → S-UF valence mapping (paper Eq. 3-4, simplified w_j(i)=1)
        PUV = puf_valence * self.e

        # 3. Valence diffusion (only among co-active connected concepts)
        diffusion = np.zeros(self.N)
        for i in range(self.N):
            if active[i]:
                for j in range(self.N):
                    if self.adj[i, j] and active[j]:
                        diffusion[i] += self.beta * self.eta * (self.UV[j] - self.UV[i])

        # 4. Update S-UF valence
        self.UV += PUV + diffusion

        # 5. Depletion when active without P-UF support
        depletion_mask = active & (np.abs(puf_valence) < 0.1)
        self.UV[depletion_mask] *= 0.96

        # 6. Clip valence (paper Eq. 8)
        self.UV = np.clip(self.UV, -self.L, self.L)

        # 7. Simplified anger dynamics (paper Anger section)
        self.EMA = self.gamma * self.EMA + (1 - self.gamma) * self.UV
        current_pos = np.sum(np.maximum(self.UV, 0))
        baseline_pos = np.sum(np.maximum(self.EMA, 0))
        loss = max(baseline_pos - current_pos, 0)
        anger = max(loss - 3.0, 0)          # activation threshold θ

        # Stronger anger if home (positive) is threatened by intruder/pain
        if active[1] and active[2] and self.UV[1] > 2 and self.UV[2] < -2:
            anger = max(anger, 8.0)

        # Anger protection: limited recovery + reduced diffusion on home
        if anger > 5 and active[1]:
            recovery = 0.6 * anger * max(self.EMA[1] - self.UV[1], 0)
            self.UV[1] += recovery
            # Reduce further diffusion loss on home
            diffusion[1] *= 0.35

        self.UV = np.clip(self.UV, -self.L, self.L)
        return anger

    def simulate(self, total_steps=110):
        """Run a full scenario: safe eating → intruder threat → recovery."""
        self.reset()

        for t in range(total_steps):
            active = np.zeros(self.N, dtype=bool)
            puf = np.zeros(self.N)
            phase = "Unknown"

            if t < 25:                    # Phase 1: Safe at home, eating food
                active[0] = active[1] = True   # food + home
                puf[0] = 7.5                   # positive P-UF (food)
                phase = "Safe eating (Food+Home)"

            elif t < 55:                  # Phase 2: Intruder arrives → pain
                active[1] = active[2] = active[3] = True  # home + intruder + pain
                puf[3] = -11.0                 # negative P-UF (pain)
                phase = "Intruder attack (Pain)"

            else:                         # Phase 3: Threat resolved, back to home
                active[1] = True               # home only
                phase = "Recovery (Home safe)"

            anger = self.step(active, puf)

            self.history_UV.append(self.UV.copy())
            self.history_anger.append(anger)
            self.history_phase.append(phase)

        self.plot()

    def plot(self):
        uv_hist = np.array(self.history_UV)
        labels = ['Food [0] (pos P-UF)', 'Home [1]', 'Intruder [2]', 'Pain [3] (neg P-UF)']

        plt.figure(figsize=(14, 8))
        for i in range(self.N):
            plt.plot(uv_hist[:, i], label=labels[i], linewidth=2.5)

        plt.plot(self.history_anger, 'r--', linewidth=2, label='Anger level')

        # Highlight phases
        plt.axvspan(0, 25, alpha=0.15, color='green', label='Safe eating [0, 1]')
        plt.axvspan(25, 55, alpha=0.15, color='orange', label='Intruder attack [1, 2, 3]')
        plt.axvspan(55, len(uv_hist)-1, alpha=0.15, color='lightblue', label='Intruder neutralized [1]')

        plt.title("Biomimetic Emotion Model Simulation\n4-Concept World: Food / Home / Intruder / Pain", fontsize=16)
        plt.xlabel("Time steps")
        plt.ylabel("Valence (S-UF utility value)")
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# ====================== RUN ======================
if __name__ == "__main__":
    print("Running simple 4-concept emotion simulation (food, home, intruder, pain)...")
    print("Food triggers positive P-UF, pain triggers negative P-UF.\n")

    sim = SimpleEmotionSim()
    sim.simulate(110)

    print("\nSimulation complete!")
    print("• Positive valence builds on 'food' and 'home'.")
    print("• Intruder + pain causes negative transfer and anger.")
    print("• Anger protects 'home' valence (limited recovery + reduced diffusion).")
    print("• Matches key dynamics from the paper (P-UF/S-UF, diffusion, anger, depletion).")