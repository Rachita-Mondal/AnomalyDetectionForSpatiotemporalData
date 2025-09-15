from abc import ABC, abstractmethod
from time import perf_counter
from collections import defaultdict
from pprint import pprint

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
# from torch.utils.tensorboard import SummaryWriter
import wandb

from src.algorithms.vae_es.vae import VAE, vae_loss, cvae_loss, CVAE


class VAE_ES_Base(ABC):
    def __init__(self, bbox_func,
                 mu, lda,
                 x_dim, y_dim, z_dim,
                 config,
                 box_constraints=None,
                 device='cuda' if torch.cuda.is_available else 'cpu',
                 verbose=1,
                 **kwargs):
        vae_config = config['vae']
        vae_train_config = config['vae_train']
        vae_val_config = config['vae_val']
        
        self.bbox_func = bbox_func
        self.mu = mu
        self.lda = lda
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.n_gen = kwargs.get('n_gen', 100)
        self.verbose = verbose
        self.val_frac = vae_val_config['val_frac']
        size_lim = vae_train_config.pop('max_dataset_size', 500)
        if self.val_frac > 0:
            self.train_dataset = VaeDataset(lda-int(lda*(self.val_frac)), size_lim=size_lim)
            self.val_dataset = VaeDataset(int(lda*self.val_frac), size_lim=size_lim)


        if box_constraints is None:
            self.box_constraints = [(-10, 10)]*x_dim
        else:
            self.box_constraints = box_constraints

        self.vae = CVAE(x_dim, y_dim, z_dim, box_constraints,vae_config).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), **vae_train_config)
        
        

        self.device = device
        config['x_dim'] = x_dim
        config['y_dim'] = y_dim
        config['z_dim'] = z_dim
        config['mu'] = mu
        config['lda'] = lda
        config['func'] = bbox_func.__name__
        config['box_constraints'] = self.box_constraints
        config = {**config, **kwargs}
        wandb_kwargs = config['wandb']
        pprint(wandb_kwargs)
        self.wandb_run = wandb.init(project="VAE-ES", config=config, **wandb_kwargs)
        self.measures = defaultdict(list)
        self.gen = 0

        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])

        self.x_off = torch.zeros((lda, x_dim)).to(device)
        self.x_par = None
        self.y_par = None
        for i in range(x_dim):
            if self.box_constraints[i] is not None:
                self.x_off[:,i] = torch.FloatTensor(np.random.uniform(*self.box_constraints[i], lda)).to(device)
            else:
                self.x_off[:,i] = torch.randn(lda).to(device)*10
        self.best_x = torch.zeros((1, x_dim)).to(device)
        self.best_y = torch.ones((1, y_dim)).to(device)*torch.inf
        self.evaluate()
        self.add_data(self.x_off, self.y_off)
        self.train_vae()
        if self.val_frac > 0:
            self.eval_vae()
        self.track()
    

    

    @abstractmethod
    def gen_offspring(self):
        pass

    @abstractmethod
    def selection(self):
        """Select the best parents from the population"""
        pass
    
    def iterate_generation(self):
        self.gen += 1
        self.selection()
        self.gen_offspring()
        self.evaluate()
        self.add_data(self.x_off, self.y_off)
        self.train_vae()
        if self.val_frac > 0:
            self.eval_vae()
        self.track()
        self.visualize_iter()

    def evaluate(self):
        self.vae.eval()
        self.y_off = self.bbox_func(self.x_off).detach()
        best_idx = torch.argmin(self.y_off)
        if self.y_off[best_idx] < self.best_y:
            self.best_x = self.x_off[best_idx]
            self.best_y = self.y_off[best_idx]
        self.measures['best_y'].append(self.best_y.cpu().item())
        self.measures['offspring_mean'].append(self.y_off.mean().cpu().item())

    def visualize_iter(self):
        pass

    def visualize_run(self):
        pass

    def init_visualization(self):
        pass
    
    def run(self, n_gen):
        if self.gen == self.n_gen:
            self.n_gen += n_gen
        while self.gen < self.n_gen:
            self.iterate_generation()
        self.visualize_run()

    def track(self):
        latest_metrics = {key: value[-1] for key, value in self.measures.items()}
        latest_metrics['gen'] = self.gen
        if self.verbose > 0:
           pprint(latest_metrics)
        self.wandb_run.log(latest_metrics, step=self.gen)

    


    def train_vae(self):
        vloader = DataLoader(self.train_dataset, batch_size=self.mu//2, shuffle=True)
        vae_start_time = perf_counter()
        self.vae.train()
        batch_loss = 0
        num_batches = 0
        
        for x, y in vloader:
        
            self.vae.zero_grad()
            if isinstance(self.vae, CVAE):
                x_hat, z_mean, logvar = self.vae(x, y)
                loss = cvae_loss(x, x_hat, z_mean, logvar)
            elif isinstance(self.vae, VAE):
                x_hat, z_mean, logvar = self.vae(x)
                loss = vae_loss(x, x_hat, z_mean, logvar)
            loss.backward()
            self.vae_optimizer.step()
            batch_loss += loss.item()*x.shape[0]
            num_batches += 1
        self.measures['train/loss'].append(batch_loss/num_batches)
        self.measures['train/time'].append(perf_counter()-vae_start_time)
        self.vae.eval()


    def eval_vae(self):
        vloader = DataLoader(self.val_dataset, batch_size=self.mu//2, shuffle=True)
        batch_loss = 0
        num_batches = 0
        for x, y in vloader:
            if isinstance(self.vae, CVAE):
                x_hat, z_mean, logvar = self.vae(x, y)
                loss = cvae_loss(x, x_hat, z_mean, logvar)
            elif isinstance(self.vae, VAE):
                x_hat, z_mean, logvar = self.vae(x)
                loss = vae_loss(x, x_hat, z_mean, logvar)
            batch_loss += loss.item()*x.shape[0]
            num_batches += 1
        self.measures['val/loss'].append(batch_loss/num_batches)


    def add_data(self, x, y):
        if self.val_frac > 0:
            rand_idx = torch.randperm(x.shape[0])
            n_val = int(self.val_frac*x.shape[0])
            x_val = x[rand_idx[:n_val],...]
            y_val = y[rand_idx[:n_val],...]
            x_train = x[rand_idx[n_val:],...]
            y_train = y[rand_idx[n_val:],...]
            self.train_dataset.add_data(x_train, y_train)
            self.val_dataset.add_data(x_val, y_val)
        else:
            self.train_dataset.add_data(x, y)

    def __del__(self):
        self.wandb_run.finish()
        


class VaeDataset(Dataset):
    def __init__(self,  
                 pop_size,
                 size_lim=1000):
        self.pop_size = pop_size
        self.size_lim = size_lim
        self.x = []
        self.y = []

    def __getitem__(self, index):
        i = index // self.pop_size
        j = index % self.pop_size
        return self.x[i][j,:], self.y[i][j,:]
    
    def __len__(self):
        return sum([x.shape[0] for x in self.x])
    
    def add_data(self, x, y):
        if len(self.x) >= self.size_lim: # Limit the size of the dataset
            self.x = self.x[1:]
            self.y = self.y[1:]
        self.x.append(x)
        self.y.append(y)


class VAE_ES_UM(VAE_ES_Base):
    def __init__(self, bbox_func,
                    mu, lda,
                    x_dim, y_dim, z_dim,
                    config,
                    box_constraints=None,
                    device='cuda' if torch.cuda.is_available else 'cpu',
                    **kwargs):
        # bbox_func,
        #          mu, lda,
        #          x_dim, y_dim, z_dim,
        #          config,
        #          box_constraints=None,
        #          device='cuda' if torch.cuda.is_available else 'cpu',
        #          verbose=1,
        #          **kwargs
        """Variational Autoencoded Evolutionary Strategy

        Args:
            bbox_func (_type_): _description_
            mu (_type_): _description_
            lda (_type_): _description_
            x_dim (_type_): _description_
            y_dim (_type_): _description_
            z_dim (_type_): _description_
            config (_type_): _description_
            box_constraints (_type_, optional): _description_. Defaults to None.
            device (str, optional): _description_. Defaults to 'cuda'iftorch.cuda.is_availableelse'cpu'.
            kwargs: VAE_ES_UM specific arguments
                tau (float, optional): Learning rate. Defaults to 1/np.sqrt(z_dim+y_dim).
                eps_0 (float, optional): Minimum step size. Defaults to 1e-5.
                p_c (float, optional): Crossover probability. Defaults to 0.7.
        """
        super().__init__(bbox_func, mu, lda, x_dim, y_dim, z_dim, config, box_constraints, device, **kwargs)

        self.tau = torch.tensor(kwargs.get('tau', 1/np.sqrt(2*(z_dim+y_dim))),device=self.device)
        self.tau_p = torch.tensor(kwargs.get('tau_p', 1/(np.sqrt(2)*(z_dim+y_dim))),device=self.device)
        self.eps_0 = torch.tensor(kwargs.get('eps_0', 1e-3),device=self.device)
        self.d_0 = torch.tensor(kwargs.get('d_0', 1e-5),device=self.device)
        self.p_c = torch.tensor(kwargs.get('p_c', 0.2),device=self.device)
        self.eta = torch.tensor(kwargs.get('eta', 1),device=self.device)
        self.drive_budget = kwargs.get('drive_budget', self.mu//4)
        self.verbose = kwargs.get('verbose', 1)
        self.sigma_off = torch.ones((lda, z_dim)).to(self.device)
        self.drive_off = torch.ones((lda, y_dim)).to(self.device)

    # def evaluate(self):
    #     self.vae.eval()
    #     self.y_off = self.bbox_func(self.x_off).detach()
    #     best_idx = torch.argmin(self.y_off)
    #     if self.y_off[best_idx] < self.best_y:
    #         self.best_x = self.x_off[best_idx]
    #         self.best_y = self.y_off[best_idx]
    #         # try:
    #         #     self.best_drive = self.drive_off[best_idx]
    #         #     self.best_sigma = self.sigma_off[best_idx]
    #         # except AttributeError:
    #         #     self.best_drive = 0.01* torch.ones((1, self.y_dim)).to(self.device)
    #         #     self.best_sigma = 0.01* torch.ones((1, self.z_dim)).to(self.device)
    #         #     self.drive_off = torch.ones((self.lda, self.y_dim)).to(self.device)
    #         #     self.sigma_off = torch.ones((self.lda, self.z_dim)).to(self.device)
        
    #     self.measures['best_y'].append(self.best_y.cpu().item())
    #     self.measures['offspring_mean'].append(self.y_off.mean().cpu().item())
    #     self.measures['best_x'].append(self.best_x.cpu().numpy())
    #     # self.measures['best_drive'].append(self.best_drive.cpu().numpy())
    #     # self.measures['best_sigma'].append(self.best_sigma.cpu().numpy())


    def selection(self):
        """Elitist selection"""
        if self.x_par is None:
            idx = torch.argsort(self.y_off, dim=0).squeeze()
            self.x_par = self.x_off[idx[:self.mu]]
            self.y_par = self.y_off[idx[:self.mu]]
            self.sigma_par = self.sigma_off[idx[:self.mu]]
            self.drive_par = self.drive_off[idx[:self.mu]]
        else:
            y_pop = torch.cat([self.y_par, self.y_off], dim=0)
            idx = torch.argsort(y_pop, dim=0).squeeze()
            self.x_par = torch.cat([self.x_par, self.x_off], dim=0)[idx[:self.mu]]
            self.y_par = y_pop[idx[:self.mu]]
            self.sigma_par = torch.cat([self.sigma_par, self.sigma_off], dim=0)[idx[:self.mu]]
            self.drive_par = torch.cat([self.drive_par, self.drive_off], dim=0)[idx[:self.mu]]
        
        if self.y_par[0] < self.best_y:
            self.best_x = self.x_par[0]
            self.best_y = self.y_par[0]
        self.measures['best_y'].append(self.best_y.cpu().item())
        self.measures['parent_mean'].append(self.y_par.mean().cpu().item())
        self.measures['best_x'].append(self.best_x.cpu().numpy())


    def gen_offspring(self):
        self.vae.eval()
        x_off = []
        sigma_off = []
        drive_off = []
        # drive_off.append(
        #     self.best_drive*torch.exp(
        #         self.tau*torch.randn(self.drive_budget, self.y_dim
        #                 ).to(self.device) + self.tau_p*torch.randn(1).to(self.device)
        #             )
        #         )
        # sigma_off.append(
        #     self.best_sigma*torch.exp(
        #         self.tau*torch.randn(self.drive_budget, self.z_dim
        #                 ).to(self.device) + self.tau_p*torch.randn(1).to(self.device)
        #             )
        #         )
        # x_greed, _, _  = self.vae.sample(self.best_y - 0.5*drive_off[0], self.drive_budget)
        
        x_greed, _, _  = self.vae.sample(self.best_y, self.drive_budget)
        sigma_off.append(self.sigma_par.mean(dim=0).unsqueeze(0).repeat(self.drive_budget, 1))
        drive_off.append(self.drive_par.mean(dim=0).unsqueeze(0).repeat(self.drive_budget, 1))
        x_off.append(x_greed)
        i = 0
        while i < self.lda-self.drive_budget:
            # Greediest drive
            if torch.rand(1, device=self.device) < self.p_c:
                if i < self.mu:
                    p1_ind = i
                else:
                    p1_ind = torch.randint(0, self.mu, (1,), device=self.device).item()
                p2_ind = p1_ind
                while p1_ind == p2_ind:
                    p2_ind = torch.randint(0, self.mu, (1,), device=self.device).item()
                p1_ind = torch.tensor(p1_ind, device=self.device).reshape((1,))
                p2_ind = torch.tensor(p2_ind, device=self.device).reshape((1,))
                x_1, x_2, sigma_1, sigma_2, drive_1, drive_2 = self.sbx_eta(p1_ind, p2_ind)
                x_off.append(x_1)
                x_off.append(x_2)
                sigma_off.append(sigma_1)
                sigma_off.append(sigma_2)
                drive_off.append(drive_1)
                drive_off.append(drive_2)
                i += 2
            if i < self.mu:
                p1_ind = i
            else:
                p1_ind = torch.randint(0, self.mu, (1,)).item()
            p1_ind = torch.tensor(p1_ind, device=self.device).reshape((1,))
            x_new, sigma_new, drive_new = self.uncorrelated_mutation(p1_ind)
            x_off.append(x_new)
            sigma_off.append(sigma_new)
            drive_off.append(drive_new)
            i += 1
        self.x_off = torch.cat(x_off, dim=0)[:self.lda,:]
        self.sigma_off = torch.cat(sigma_off, dim=0)[:self.lda,:]
        self.drive_off = torch.cat(drive_off, dim=0)[:self.lda,:]
    
    def uncorrelated_mutation(self, ind):
        sigma_new = torch.maximum(self.eps_0,
                self.sigma_par[ind,:]*torch.exp(
                    self.tau*torch.randn_like(self.sigma_par[ind,:]).to(self.device) + self.tau_p*torch.randn(1).to(self.device)
                )
            )
        drive_new = torch.maximum(self.d_0,
                                  self.drive_par[ind,:]*torch.exp(
                    self.tau*torch.randn_like(self.drive_par[ind,:]) + self.tau_p*torch.randn(1).to(self.device)
                )
            )
        z, logvar = self.vae.encode(self.x_par[ind,:], self.y_par[ind,:])
        z = z.detach()
        logvar = logvar.detach()
        z_new = z + sigma_new*torch.exp(0.5*logvar)*torch.randn_like(z)
        x_new = self.vae.decode(z_new, self.y_par[ind,:] - 0.5*drive_new*torch.randn_like(self.y_par[ind,:])).detach()
        return x_new, sigma_new, drive_new

    def sbx_eta(self, p1_ind, p2_ind):
        u = torch.rand(1).to(self.device)
        Bq = (u<=0.5)*((2*u)**(1/(self.eta+1))) + (u>0.5)*((0.5/(1-u))**(1/(self.eta+1)))
        z_1, _ = self.vae.encode(self.x_par[p1_ind], self.y_par[p1_ind])
        z_2, _= self.vae.encode(self.x_par[p2_ind], self.y_par[p2_ind])
        z_1 = z_1.detach()
        z_2 = z_2.detach()
        z_1 = 0.5*((1+Bq)*z_1 + (1-Bq)*z_2)
        z_2 = 0.5*((1-Bq)*z_1 + (1+Bq)*z_2)
        sigma_1 = 0.5*((1+Bq)*self.sigma_par[p1_ind] + (1-Bq)*self.sigma_par[p2_ind])
        sigma_2 = 0.5*((1-Bq)*self.sigma_par[p1_ind] + (1+Bq)*self.sigma_par[p2_ind])
        drive_1 = 0.5*((1+Bq)*self.drive_par[p1_ind] + (1-Bq)*self.drive_par[p2_ind])
        drive_2 = 0.5*((1-Bq)*self.drive_par[p1_ind] + (1+Bq)*self.drive_par[p2_ind])
        y_1 = 0.5*((1+Bq)*self.y_par[p1_ind] + (1-Bq)*self.y_par[p2_ind])
        y_2 = 0.5*((1-Bq)*self.y_par[p1_ind] + (1+Bq)*self.y_par[p2_ind])
        x_1 = self.vae.decode(z_1, y_1)
        x_2 = self.vae.decode(z_2, y_2)
        x_1 = x_1.detach()
        x_2 = x_2.detach()
        return x_1, x_2, sigma_1, sigma_2, drive_1, drive_2

    def visualize_iter(self):
        pass

    def visualize_run(self):
        pass