import torch

class SAM(torch.optim.Optimizer):
    """Sharpness aware minimization.
    
    Parameters
    ----------
    params:
        model parameters
    base_optimizer: torch.optim.Optimizer
        e.g. ADAMW or SGD
    rho: float
        hyperparameter for SAM - perturbation strength (default 0.05)
    gsam_alpha: float
        hyperparameter for GSAM (default 0.05)
    GSAM: bool
        whether or not to use Surrogate Gap Sharpness Aware Minimization
    adaptive: bool
        whether or not to use Adaptive SAM
    
    Public Methods
    --------------
    first_step(self, zero_grad):
        Perturb weights.
    unperturb(self, zero_grad):
        Unperturb weights.
    second_step(self, zero_grad):
        Unperturb and update weights.
    load_state_dict(self, state_dict):
        Copies parameters and buffers from state_dict 
        into this module and its descendants.
    
    """
    def __init__(self, params, base_optimizer, rho=0.05, gsam_alpha=0.05, GSAM=False, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        
        self.rho = rho
        self.alpha = gsam_alpha
        self.GSAM = GSAM
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """Perturb weights."""
        grad_norm = self._grad_norm()
        for group in self.param_groups: #Iterate over parameters/weights
            scale = self.rho / (grad_norm + 1e-12) #Perturbation factor

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone() #Save old parameter to unperturb later
                if self.GSAM: self.state[p]["old_p_grad"] = p.grad.data.clone() #Save old gradient (if GSAM)
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p) #Calculate perturbation
                p.add_(e_w)  #Climb to the local maximum "w + e(w)" - perturb
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def unperturb(self, zero_grad=False):
        """Return to old parameters - remove perturbation."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  #Get back to "w" from "w + e(w)"
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """Unperturb weights and then update.

        If GSAM, decompose gradients before unperturbing weights."""
        if self.GSAM:
            self._decompose_grad()
        self.unperturb()

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  #Put everything on the same device, in case of model parallelism
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

    def _decompose_grad(self):
        """Decompose gradient of unperturbed loss into directions parallel and
        perpendicular to the gradients of the perturbed losses, for GSAM.
        Subtract perpendicular component from perturbed loss gradients.
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                old_grad = self.state[p]["old_p_grad"]
                if old_grad is None: continue
                #Find factor of component parallel to perturbed loss
                #Take dot product between two vectors.
                a = torch.dot(p.grad.data.view(-1), old_grad.view(-1))/torch.norm(p.grad.data)**2
                perp = old_grad - a*p.grad.data #Component perpendicular to perturbed loss = vector - parallel component
                norm_perp = perp / torch.norm(perp) #Normalise perpendicular component
                #Subtract perp component from perturbed loss gradients, with factor alpha.
                p.grad.data.sub_(self.alpha * norm_perp)