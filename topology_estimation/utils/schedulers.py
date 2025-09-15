class BetaScheduler:
    def __init__(self, total_steps, final_beta=1.0, warmup_frac=0.3):
        self.total_steps = total_steps
        self.final_beta = final_beta
        self.warmup_steps = int(total_steps * warmup_frac)

    def __call__(self, step):
        if step >= self.warmup_steps:
            return self.final_beta
        else:
            return self.final_beta * (step / self.warmup_steps)
        

class TempScheduler:
    def __init__(self, init_tau=1.0, min_tau=0.3, decay=0.001):
        self.tau = init_tau
        self.min_tau = min_tau
        self.decay = decay

    def step(self):
        self.tau = max(self.min_tau, self.tau * (1 - self.decay))
        return self.tau