# prerequisites
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from utils import utils
from utils.saver import Saver
import argparse


class Encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.mu_enc = torch.nn.Linear(hidden_dim, z_dim)
        self.var_enc = torch.nn.Linear(hidden_dim, z_dim)
        
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        mu = self.mu_enc(x)
        log_var = self.var_enc(x)
        return mu, log_var
    
    
class Decoder(torch.nn.Module):
    def __init__(self, z_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(z_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.out = torch.nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        x = self.out(x)
        return x
    
    
class VAE(torch.nn.Module):
    def __init__(self, enc, dec):
        super(VAE, self).__init__()
        self.enc = enc
        self.dec = dec
        
    def forward(self, x):
        mu, log_var = self.enc(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.dec(z)
        return x_reconst, mu, log_var
    
    def reparameterize(self, mu, log_var):
        # log_var = log(std^2)
        # log_var = 2*log(std)
        # std = exp(0.5*log_var)
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + std*eps
    
    
def kl_divergence(mu, log_var):
    return 0.5 * torch.sum(torch.exp(log_var) + mu**2 - 1. - log_var, dim=1)

def loss_function(x, x_reconst, mu, log_var, beta = 1.0):
    reconstruction_loss = F.mse_loss(x_reconst, x, reduction='sum') / x.shape[0]
    kld = kl_divergence(mu, log_var).mean()
    return reconstruction_loss + beta * kld, reconstruction_loss, kld

def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path to the config file")
    return parser.parse_args(args=args, namespace=namespace)

if __name__ == '__main__':
    cmd = parse_args()
    
    # load config
    args = utils.load_config(cmd.config)
    
    input_dim = args.input_dim
    hidden_dim = args.hidden_dim
    z_dim = args.z_dim
    bs = args.batch_size
    n_epochs = args.n_epochs
    save_dir = args.save_dir
    log_interval = args.log_interval
    val_interval_epoch = args.val_interval_epoch
    beta = z_dim/input_dim if args.enable_beta else 1.0
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # calculated mean and std 
    ])

    train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transform, download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)
    val_interval = len(train_loader) * val_interval_epoch
    
    encoder = Encoder(input_dim, hidden_dim, z_dim)
    decoder = Decoder(z_dim, hidden_dim, input_dim)
    vae = VAE(encoder, decoder)
    
    optimizer = optim.AdamW(vae.parameters(), lr=0.001)

    saver = Saver(args)
    for k, v in args.items():
        saver.log_info(f'> {k}: {v}')
    
    params_count = utils.get_network_paras_amount({'model': vae})
    saver.log_info('--- model size ---')
    saver.log_info(params_count)
    

    saver.log_info('======= start training =======')
    for epoch in range(n_epochs):
        vae.train()
        for i, data in enumerate(train_loader):
            saver.global_step_increment()
            images, labels = data

            optimizer.zero_grad()
            x_reconst, mu, log_var = vae(images.view(-1,input_dim))

            loss, reconstruction_loss, kld = loss_function(images.view(-1,input_dim), x_reconst, mu, log_var, beta=beta)

            loss.backward()
            optimizer.step()

            if saver.global_step % log_interval == 0:
                saver.log_info(
                    'epoch: {} | {:3d}/{:3d} |batch/s: {:.2f} | loss: {:.3f} | rec_loss: {:.3f} | reg_loss: {:.3f} | time: {} | step: {}'.format(
                        epoch,
                        epoch*len(train_loader)+i+1,
                        n_epochs*len(train_loader),
                        log_interval/saver.get_interval_time(),
                        loss.item(),
                        reconstruction_loss.item(),
                        kld.item(),
                        saver.get_total_time(),
                        saver.global_step
                    )
                )
                
                saver.log_value({
                    'train/loss': loss.item(),
                    'train/rec_loss': reconstruction_loss.item(),
                    'train/reg_loss': kld.item(),
                })
                
            
            # validation
            if saver.global_step % val_interval == 0 or (epoch == n_epochs-1 and i == len(train_loader)-1):
                # run testing set
                vae.eval()
                test_loss= 0
                test_rec_loss = 0
                test_reg_loss = 0
                with torch.no_grad():
                    for i, data in enumerate(test_loader):
                        images, labels = data
                        x_reconst, mu, log_var = vae(images.view(-1,input_dim))
                        loss, reconstruction_loss, kld = loss_function(images.view(-1,input_dim), x_reconst, mu, log_var, beta=beta)
                        test_loss += loss.item()
                        test_rec_loss += reconstruction_loss.item()
                        test_reg_loss += kld.item()
                        if i == 0:
                            saver.log_image(torch.squeeze(x_reconst.view(-1,28,28).detach().cpu()), labels.detach().cpu().numpy(), n_img=8, name='recon')
                            
                    ramdom_sample = torch.randn(8, z_dim)
                    recon_batch = vae.dec(ramdom_sample)
                    saver.log_image(torch.squeeze(recon_batch.view(-1,28,28).detach().cpu()), n_img=8, name='gen')

                test_loss /= len(test_loader)
                test_rec_loss /= len(test_loader)
                test_reg_loss /= len(test_loader)
                
                # log loss
                saver.log_info(
                    ' --- validation --- \nloss: {:.3f} | rec_loss: {:.3f} | reg_loss: {:.3f}'.format(
                        test_loss,
                        test_rec_loss,
                        test_reg_loss
                    )
                )

                saver.log_value({
                    'val/loss': test_loss,
                    'val/rec_loss': test_rec_loss,
                    'val/reg_loss': test_reg_loss,
                })
    
    saver.save_model(vae, optimizer, postfix=f'{saver.global_step}')