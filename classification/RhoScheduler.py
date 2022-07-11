class RhoScheduler:
    def __init__(self, rho_max, rho_min, lr_max, lr_min, lr):
        self.rho_max = rho_max
        self.rho_min = rho_min
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.lr = lr
        self.step()

    def step(self):
        self.rho = self.rho_min + ((self.rho_max - self.rho_min) * (self.lr - self.lr_min))/(self.lr_max - self.lr_min)
        return self.rho

    def update_lr(self, new_lr):
        self.lr = new_lr