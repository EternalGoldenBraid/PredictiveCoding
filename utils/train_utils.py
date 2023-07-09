import numpy as np

class EarlyStopper:
    def __init__(self, patience: int = 5, delta: float = 0.0):
        self.patience = patience
        self.delta = delta
        self.best_score = np.inf
        self.counter = 0

    def __call__(self, score: float) -> None:
        if score < self.best_score - self.delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        
        return False