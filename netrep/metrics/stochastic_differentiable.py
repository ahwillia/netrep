# %%
import torch
from torch import nn
from torch import optim

from torch.nn.utils.parametrizations import orthogonal

from typing import Literal, List

from tqdm.auto import trange

class GPStochasticDiff:

    def __init__(
            self,
            n_dims,
            n_times,
            alpha: float=1.0, 
            type: Literal["Bures", "Adapted_Bures", "Knothe_Rosenblatt", "Marginal_Bures"] = "Adapted_Bures",
    ):
    
        self.n_times = n_times
        self.n_dims = n_dims

        self.alpha = alpha

        marginalize = lambda cov: torch.block_diag(*[cov[i*n_dims:(i+1)*n_dims,i*n_dims:(i+1)*n_dims] for i in range(n_times)])

        def sqrtm(cov):
            v, u = torch.linalg.eigh(cov)
            s = torch.einsum("jk,k,lk->jl", u, torch.sqrt(torch.maximum(v,torch.tensor(0))), u)
            return s
        
        self.type = type

        self.cov_dist_fns = {
            'Bures': lambda A,B: torch.trace(A) + torch.trace(B) - 2*torch.trace(sqrtm(sqrtm(B)@A@sqrtm(B))),
            'Adapted_Bures': lambda A,B: torch.trace(A) + torch.trace(B) -2*torch.diag(torch.cholesky(A).T@torch.cholesky(B)).sum(),
            'Knothe_Rosenblatt': lambda A,B: ((torch.cholesky(A)-torch.cholesky(B))**2).sum()
        }
        self.cov_dist_fns['Marginal_Bures'] = lambda A,B: self.cov_dist_fns['Bures'](marginalize(A),marginalize(B))

    def fit_score(self,A,B,patience=4,tol=1e-6,lr=.1,momentum=.9,epsilon=1e-6):
        n,t = self.n_dims, self.n_times
        cov_dist_fn = self.cov_dist_fns[self.type]
        alpha = self.alpha

        Q = orthogonal(nn.Linear(n, n))
        optimizer = optim.SGD(Q.parameters(), lr=lr, momentum=momentum)

        mu_A = torch.tensor(A[0]).float()
        mu_B = torch.tensor(B[0]).float()

        Sigma_A = torch.tensor(A[1]).float()
        Sigma_B = torch.tensor(B[1]).float()

        counter,l = 0,0
        loss = []

        pbar = trange(0,bar_format=None)
        pbar.set_description('Optimizing ...')

        total_iter = 0
        while counter < patience:
            # Update parameters
            optimizer.zero_grad()
            Q_ = torch.kron(torch.eye(t),Q.weight)

            mean_dist = ((mu_A - Q_@mu_B)**2).sum()
            cov_dist = cov_dist_fn(Sigma_A+epsilon*torch.eye(n*t),Q_@Sigma_B@Q_.T+epsilon*torch.eye(n*t))
            # cov_dist  = ((cholesky_A-torch.cholesky(Q_@Sigma_B@Q_.T+epsilon*torch.eye(n*t)))**2).sum()

            dist = alpha*mean_dist + (2-alpha)*cov_dist

            dist.backward()
            optimizer.step()

            l_new = dist.detach()
            loss.append(l_new)
            counter = counter + 1 if torch.abs(l_new-l) < tol else 0
            l = l_new

            total_iter+=1
            
            if total_iter%100==0:
                pbar.set_description("Iteration {}, loss {:0.2f}".format(total_iter,l_new))    

        self.loss = torch.tensor(loss).numpy()
        self.Q = Q.weight.detach().numpy().T

        return dist.detach().numpy()

