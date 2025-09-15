import torch
import torch.nn as nn
from torch.nn import functional as F
# torch.autograd.set_detect_anomaly(True)

class VAE(nn.Module):
    def __init__(self,
                x_dim,
                y_dim,
                z_dim=None,
                box_constraints=None):
        """_summary_
        """
        super(VAE, self).__init__()
        self.x_dim = x_dim
        if z_dim is None:
            self.z_dim = x_dim
        else:
            self.z_dim = x_dim
        self.y_dim = y_dim
        if box_constraints is None:
            self.box_constraints = [(-10, 10)]*x_dim
        self.box_constraints = box_constraints

        # Encoder
        self.encoder = nn.Sequential(nn.Linear(x_dim+y_dim, int((x_dim*3+y_dim))),
                                        nn.Dropout(0.1),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(int((x_dim*3+y_dim)), int(x_dim*3+y_dim)),
                                        nn.ReLU(inplace=True)
                                        )
        self.fc_mu = nn.Linear(int(x_dim*3+y_dim), z_dim)
        self.fc_logvar = nn.Linear(int(x_dim*3+y_dim), z_dim)

        # Decoder
        self.decoder = nn.Sequential(nn.Linear(z_dim, int(x_dim*3+y_dim)),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(int(x_dim*3+y_dim), int(x_dim*3+y_dim)),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(int(x_dim*3+y_dim), x_dim+y_dim)
                                        )
        
    def decode(self, z):
        h = self.decoder(z)
        for i in range(len(self.box_constraints)):
            low, high = self.box_constraints[i]
            # h[:,i] = ((high-low)*nn.functional.relu6(h[:,i])/6)+low
            # h[:,i] = torch.sigmoid(h[:,i])*(high-low)+low
            h[:,i] = low + (high-low)*(torch.tanh(h[:,i])+1)/2
        return h[:,:self.x_dim], h[:,self.x_dim:]
    
    def encode(self, x, y):
        h = torch.cat([x, y], dim=1)
        h = self.encoder(h)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        x, y = self.decode(z)
        return x, y, mu, logvar


    def forward_deterministic(self, x, y):
        mu, logvar = self.encode(x, y)
        z = mu
        return self.decode(z), mu, logvar


    def sample(self, n_samples=1):
        z = torch.randn(n_samples, self.z_dim).to(self.device)
        x,y = self.decode(z)
        x = x.detach()
        y = y.detach()
        return x, y, z.detach()


    def sample_attached(self, n_samples=1):
        """Keep the output of the decoder attached to the graph
        
        Used to train the decoder for a fixed z"""
        z = torch.randn(n_samples, self.z_dim).to(self.device)
        out = self.decode(z)
        return out[:,:self.x_dim], out[:,self.x_dim:], z.detach()

def dec_loss(x_hat, y_hat, x,y):
    """
    y_hat:  (batch_size, y_dim) Sampled y from the decoder
    y:      (batch_size, y_dim) True y calculated by black-box function
    """
    return F.mse_loss(y_hat, y) + F.mse_loss(x_hat, x)

def vae_loss(x, x_hat, y, y_hat, mu, logvar):
    """
    x:      (batch_size, x_dim) True x / previous sample
    x_hat:  (batch_size, x_dim) Reconsturcted x
    y:      (batch_size, y_dim) True y calculated by black-box function
    y_hat:  (batch_size, y_dim) Reconstruced y
    mu:     (batch_size, z_dim) Mean of the latent space
    logvar: (batch_size, z_dim) Log variance of the latent space
    """
    # Reconstruction loss
    d = torch.cat([x, y], dim=1)
    d_hat = torch.cat([x_hat, y_hat], dim=1)
    # BCE = F.binary_cross_entropy(d_hat, d, reduction='sum')
    mse = F.mse_loss(d_hat, d, reduction='sum')
    # mse += F.mse_loss(y_hat, mu)
    # KL Divergence
    kl_div = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)
    return mse + kl_div



class CVAE(nn.Module):
    def __init__(self,
                x_dim,
                y_dim,
                z_dim,
                box_constraints=None,
                config=None):
        """Conditional Variational Autoencoder for black-box optimization

        x: input of the black-box function
        y: output of the black-box function
        zn: normal manifold in the latent space
        zt: tangent manifold in the latent space

        Parameters:
        -----------
        x_dim: int
            Dimension of the input
        y_dim: int
            Dimension of the output
        z_dim: int
            Dimension of the latent space
        box_constraints: list of tuples
            List of tuples containing the lower and upper bound of each dimension of x
        config: dict
            Configuration dictionary containing the following
            - encoder: list of integers
                List of integers representing the number of neurons in each layer of the encoder
            - decoder: list of integers
                List of integers representing the number of neurons in each layer of the decoder
        """
        super(CVAE, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        if z_dim is None:
            self.z_dim = x_dim
        else:
            self.z_dim = x_dim

        if box_constraints is None:
            self.box_constraints = [(-10, 10)]*x_dim
        self.box_constraints = box_constraints

        if config is None:
            config = {
                'encoder':
                    [
                        (int((x_dim+y_dim)*1.5), 'relu', 0.1),
                        (int((x_dim+y_dim)*1.5), 'relu', 0.1),
                        (int((x_dim+y_dim)*1.5), 'relu', 0.1),
                    ],
                'decoder':
                    [
                        (int((z_dim+y_dim)*1.5), 'relu', 0.1),
                        (int((z_dim+y_dim)*1.5), 'relu', 0.1),
                        (int((z_dim+y_dim)*1.5), 'relu', 0.1),
                    ]
                }

        # Encoder
        input_dim = x_dim + y_dim
        encoder_layers = []
        for out_dim, activation, dropout in config['encoder']:
            encoder_layers.append(nn.Linear(input_dim, out_dim))
            if activation == 'relu':
                encoder_layers.append(nn.ReLU(inplace=True))
            elif activation == 'tanh':
                encoder_layers.append(nn.Tanh())
            elif activation == 'none':
                pass
            else:
                raise ValueError(f"Activation function {activation} not supported")
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            input_dim = out_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(out_dim, z_dim)
        self.fc_logvar = nn.Linear(out_dim, z_dim)
        
        decoder_layers = []
        input_dim = z_dim + y_dim
        for out_dim, activation, dropout in config['decoder']:
            decoder_layers.append(nn.Linear(input_dim, out_dim))
            if activation == 'relu':
                decoder_layers.append(nn.ReLU(inplace=True))
            elif activation == 'tanh':
                decoder_layers.append(nn.Tanh())
            elif activation == 'none':
                pass
            else:
                raise ValueError(f"Activation function {activation} not supported")
            if dropout > 0:
                decoder_layers.append(nn.Dropout(dropout))
            input_dim = out_dim
        decoder_layers.append(nn.Linear(input_dim, x_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    

    def decode(self, z, y):
        h = torch.cat([z, y], dim=1)
        x = self.decoder(h)
        for i in range(self.x_dim):
            constraint = self.box_constraints[i]
            if constraint is None:
                pass
            else:
                pass
                low, high = constraint
                # x[:,i] = ((high-low)*nn.functional.relu6(x[:,i])/6)+low
                # x[:,i] = torch.sigmoid(x[:,i])*(high-low)+low
                # x[:,i] = low + (high-low)*(torch.tanh(x[:,i])+1)/2
        return x


    def encode(self, x, y):
        h = torch.cat([x, y], dim=1)
        h = self.encoder(h)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        x = self.decode(z, y)
        return x, mu, logvar


    def forward_deterministic(self, x, y):
        mu, logvar = self.encode(x, y)
        z = mu
        return self.decode(z, y), mu, logvar


    def sample(self, y, n_samples=1, temp=1):
        if y.dim() == 1:
            y = y.unsqueeze(0)
            y = y.repeat(n_samples, 1)
        else: 
            y = y.repeat(n_samples, 1)
        z = temp*torch.randn(n_samples, self.z_dim).to(y.device)
        x = self.decode(z, y)
        x = x.detach()
        return x, y, z.detach()

def cvae_loss(x, x_hat, mu, logvar):
    """
    x:      (batch_size, x_dim) True x / previous sample
    x_hat:  (batch_size, x_dim) Reconsturcted x
    mu:     (batch_size, z_dim) Mean of the latent space
    logvar: (batch_size, z_dim) Log variance of the latent space
    """
    # Reconstruction loss
    mse = F.mse_loss(x, x_hat, reduction='sum')
    # KL Divergence
    kl_div = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)
    return mse + kl_div