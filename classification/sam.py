import torch
import numpy as np

class SAMx(torch.optim.Optimizer):
    """TODO."""
    def __init__(self, base_optimizer, params, rho=0.05):
        """TODO."""
        assert rho >= 0, f"Invalid rho; should be non-negative: {rho}"

        super().__init__(params) #Inherit all methods and properties of torch.optim.Optimizer class

        self.base_optimizer = base_optimizer() #e.g. SGD or Adam
        self.param_groups = self.base_optimizer.param_groups
        self.rho = rho
        #TODO: implement rho schedule
        #self.rho = self.rho_min + ((self.rho_max - self.rho_min) * (lr - lr_min)) / (lr_max - lr_min)
    
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for w in group['params']:
                delta_w = self.rho * w.grad
        delta_w = self.rho 

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho_scheduler, adaptive=False, **kwargs):
        #assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.rho_scheduler = rho_scheduler
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

        self.update_rho()

    @torch.no_grad()
    def update_rho(self):
        self.rho = self.rho_scheduler.step()
        return self.rho

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """Perturb weights."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """Unperturb weights and then update weights."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

    def print_rho(self):
        print(self.rho)

class SAM_old(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, alpha=0.5, GSAM=False, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM_old, self).__init__(params, defaults)
        
        self.GSAM = GSAM
        self.alpha = alpha
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """Perturb weights."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                self.state[p]["old_p_grad"] = p.grad.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                #print(p)
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def unperturb(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """Unperturb weights and then update.
        If GSAM, decompose gradients before unperturbing weights."""
        if self.GSAM:
            self.__decompose_grad()
        self.unperturb()

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

    def __decompose_grad(self):
        """Decompose gradient of unperturbed loss into directions parallel and
        perpendicular to the gradients of the perturbed losses.
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                old_grad = self.state[p]["old_p_grad"]
                if old_grad is None: continue
                a = torch.dot(p.grad, old_grad)/torch.norm(p.grad)**2 #Find factor of component parallel to perturbed loss
                perp = old_grad - a*p.grad #Component perpendicular to perturbed loss
                norm_perp = perp / torch.norm(perp) #Normalise component
                #print(p.grad.data)
                print(self.alpha * norm_perp)
                p.grad.data.add_(self.alpha * norm_perp)
                #print(p.grad.data)