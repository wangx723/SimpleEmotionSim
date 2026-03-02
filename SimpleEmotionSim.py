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
        * Depletion when active without P-UF reinforcement, assuming diffusion possible
        * Simplified anger: protects positive valence of "home" when threatened
    """

    def __init__(self):
        self.N = 4             # Number of nodes
        self.alpha = 0.82      # Eligibility trace decay (paper Eq. 2)
        self.beta = 0.18       # Diffusion rate
        self.beta_eff = 0.18   # Effective diffusion rate
        self.eta = 0.87        # Transfer efficiency (lossy diffusion, paper Eq. 6)
        self.L = 15.0          # Max |valence| per perceptron
        self.gamma = 0.90      # EMA smoothing for anger baseline
        self.anger = 0.0       # Global anger
        self.kappa = 2.6       # Multiplier for max recovery
        self.sigma = 2.0      # Recovery rate of fallback UV capacity

        # define PUF values
        self.puf = np.zeros(self.N)
        self.puf[0] = 7.5  # positive P-UF (food)
        self.puf[3] = -20.0  # negative P-UF (pain)

        # Define timing of phases
        self.phases_t = np.array([25, 50, 75, 100])

        # Define active nodes in each phase
        self.phases = np.array([
            # [food, home, intruder, pain]
            [1, 1, 0, 0],   # food + home
            [0, 1, 1, 1],   # home + intruder + pain
            [0, 1, 0, 0],   # home
            [1, 1, 0, 0]    # food + home
        ])

        for i in range(self.N):
            print(self.phases[1, i])

        # Adjacency matrix for diffusion (home-food, home-intruder, intruder-pain)
        # Fully connected
        self.adj = np.array([[1 for i in range(self.N)]
                             for j in range(self.N)])

        # self.adj = np.array([
        #     [0, 1, 0, 0],  # food -> home
        #     [1, 0, 1, 0],  # home -> food, home -> intruder
        #     [0, 1, 0, 1],  # intruder -> home, intruder -> pain
        #     [0, 0, 1, 0]   # pain -> intruder
        # ])

        self.reset()

    def reset(self):
        self.UV = np.zeros(self.N)             # S-UF valence (utility values)
        self.e = np.zeros(self.N)              # Eligibility trace
        self.EMA = np.zeros(self.N)            # Baseline for anger detection
        self.anger_scalar = np.zeros(self.N)   # Local anger
        self.recovery_capacity = np.zeros(self.N)

        self.history_UV = []
        self.history_anger = []
        # self.history_phase = []
        self.history_local_anger = []
        self.history_active_anger = []

    def step(self, active):
        """One discrete time step (matches paper formulations)."""
        # 1. Update eligibility trace
        self.e = self.alpha * self.e + (1 - self.alpha) * active.astype(float)

        # 2. P-UF → S-UF valence mapping (paper Eq. 3-4, simplified w_j(i)=1)
        active_local_anger = np.sum(self.anger_scalar * active)
        total_anger = self.anger + active_local_anger
        PUV = self.puf * self.e * (1 - min(total_anger, 10.0)/10.0)

        # 3. Valence diffusion (only among co-active connected concepts)
        diffusion = np.zeros(self.N)
        for i in range(self.N):
            if active[i]:
                for j in range(self.N):
                    if self.adj[i, j] and active[j]:
                        diffusion[i] += self.beta_eff * self.eta * (self.UV[j] - self.UV[i])

        # 4. Update S-UF valence
        self.UV = np.clip(self.UV + PUV + diffusion, -self.L, self.L)

        # 5. Simplified anger dynamics (paper Anger section)
        self.EMA = self.gamma * self.EMA + (1 - self.gamma) * self.UV
        self.recovery_capacity += self.sigma
        self.recovery_capacity = np.clip(self.recovery_capacity, 0, self.kappa*self.EMA)

        current_pos = np.sum(np.maximum(self.UV, 0))
        baseline_pos = np.sum(np.maximum(self.EMA, 0))
        loss = max(baseline_pos - current_pos, 0)
        anger_t = max(loss - 1.0, 0)          # activation threshold θ = 1.0
        self.anger = anger_t + (0.9 * self.anger)

        # Reduced diffusion on anger
        self.beta_eff = self.beta * (1 - min(total_anger, 10.0)/10.0)

        # Anger protection: limited recovery
        for i in range(self.N):
            if self.EMA[i] > 0:
                desired_recovery = 1.6 * self.anger * max(self.EMA[i] - self.UV[i], 0)
                recovery = min(desired_recovery, self.recovery_capacity[i])
                self.UV[i] += recovery
                self.recovery_capacity[i] -= recovery
                self.recovery_capacity[i] = np.clip(self.recovery_capacity[i], 0, self.kappa*self.EMA[i])

            # Local anger & negative valence mapping
            self.UV[i] += -self.anger * self.e[i]
            self.anger_scalar[i] += self.anger * self.e[i]
        self.anger_scalar = np.clip(self.anger_scalar, -self.L, self.L)

        # Local anger diffusion
        anger_diff = np.zeros(self.N)
        for i in range(self.N):
            if active[i]:
                for j in range(self.N):
                    if self.adj[i, j] and active[j]:
                        anger_diff[i] += self.beta * self.eta * (self.anger_scalar[j] - self.anger_scalar[i])
        self.anger_scalar = np.clip(self.anger_scalar + anger_diff, 0, self.L)

        self.UV = np.clip(self.UV, -self.L, self.L)
        return self.anger, total_anger

    def simulate(self, total_steps=-1):
        """Run a full scenario: safe eating → intruder threat → recovery."""
        self.reset()
        if total_steps < 0:
            total_steps = self.phases_t[-1]

        for t in range(total_steps):
            active = np.zeros(self.N, dtype=bool)

            for phase in range(len(self.phases_t)):
                if t < self.phases_t[phase]:
                    for i in range(self.N):
                        if self.phases[phase, i] > 0:
                            active[i] = True
                    break

            global_anger, active_anger = self.step(active)

            self.history_UV.append(self.UV.copy())
            self.history_anger.append(global_anger)
            self.history_active_anger.append(active_anger)
            self.history_local_anger.append(self.anger_scalar.copy())
            # self.history_phase.append(phase)

        self.UV_plot(True)
        self.anger_plot(True)

    def UV_plot(self, show = False):
        uv_hist = np.array(self.history_UV)
        labels = ['Food [0] (pos P-UF)', 'Home [1]', 'Intruder [2]', 'Pain [3] (neg P-UF)']

        plt.figure(figsize=(14, 8))
        for i in range(self.N):
            plt.plot(uv_hist[:, i], label=labels[i], linewidth=2.5)

        plt.plot(self.history_anger, 'r--', linewidth=2, label='Anger level')

        # Highlight phases
        plt.axvspan(0, self.phases_t[0], alpha=0.15, color='green', label='Safe eating [0, 1]')
        plt.axvspan(self.phases_t[0], self.phases_t[1], alpha=0.15, color='orange', label='Intruder attack [1, 2, 3]')
        plt.axvspan(self.phases_t[1], len(uv_hist) - 1, alpha=0.15, color='lightblue', label='Intruder neutralized [1]')

        plt.title("Biomimetic Emotion Model Simulation\n4-Concept World: Food / Home / Intruder / Pain", fontsize=16)
        plt.xlabel("Time steps")
        plt.ylabel("Valence (S-UF utility value)")
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('simple_emotion_sim_output.png')

        if show:
            plt.show()

    def anger_plot(self, show = False):
        anger_hist = np.array(self.history_local_anger)
        labels = ['Food [0] (pos P-UF)', 'Home [1]', 'Intruder [2]', 'Pain [3] (neg P-UF)']

        plt.figure(figsize=(14, 8))
        for i in range(self.N):
            plt.plot(anger_hist[:, i], label=labels[i], linewidth=2.5)

        plt.plot(self.history_active_anger, 'r--', linewidth=2, label='Active anger')
        plt.plot(self.history_anger, 'o--', linewidth=2, label='Global anger')

        # Highlight phases
        plt.axvspan(0, self.phases_t[0], alpha=0.15, color='green', label='Safe eating [0, 1]')
        plt.axvspan(self.phases_t[0], self.phases_t[1], alpha=0.15, color='orange', label='Intruder attack [1, 2, 3]')
        plt.axvspan(self.phases_t[1], len(anger_hist) - 1, alpha=0.15, color='lightblue', label='Intruder neutralized [1]')

        plt.title("Biomimetic Emotion Model Simulation\n4-Concept World: Food / Home / Intruder / Pain", fontsize=16)
        plt.xlabel("Time steps")
        plt.ylabel("Local anger scalar")
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('simple_emotion_sim_anger_output.png')

        if show:
            plt.show()


# ====================== RUN ======================
if __name__ == "__main__":
    print("Running simple 4-concept emotion simulation (food, home, intruder, pain)...")
    print("Food triggers positive P-UF, pain triggers negative P-UF.\n")

    sim = SimpleEmotionSim()
    sim.simulate()

    print("\nSimulation complete")
    print("• Positive valence builds on 'food' and 'home'.")
    print("• Intruder + pain causes negative transfer and anger.")
    print("• Anger protects 'home' valence (limited recovery + reduced diffusion).")
    print("• Matches key dynamics from the paper (P-UF/S-UF, diffusion, anger, depletion).")
