
## Variational Autoencoder (VAE) in PyTorch

### Dataset
The dataset used in this project is the **MNIST dataset**.
 - Training set: 60,000 images.
 - Test set: 10,000 images.
 - Image size: (28, 28), pixel values in [0, 255].
 - Preprocess: Sacle and Normalize the pixel values to N(0, 1). `x = (x/255 - 0.1307) / 0.3081`.

### Usage
1. Configure the hyperparameters in `config.yaml`.
2. Train the model: `python simple_vae.py -c config.yaml`.

### Model
The model is a simple VAE with two hidden layers in both the encoder and decoder.
 - Encoder: 784 (28*28) $\rightarrow$ 100 $\rightarrow$ 100 $\rightarrow$ 32
 - Decoder: 32 $\rightarrow$ 100 $\rightarrow$ 100 $\rightarrow$ 784 (28*28)

The input image is flattened to a 784-dim vector, and the latent space is 32-dim.

### Training
Loss function: 
 - Reconstruction loss: $\sum_{i=1}^{28*28} (x_i - \hat{x}_i)^2$.
 - Regularization loss: $D_{KL}(q(z|x)||p(z)) = \frac{1}{2} \sum_{d_z=1}^{32} (\exp(\log \sigma^2) + \mu^2 - 1 - \log \sigma^2)$.
 - Total loss: $L = L_\text{recon} + \beta*L_\text{reg}$. 
 - $\beta$ is a hyperparameter to balance the two losses: $\beta = \frac{d_z}{d_x} = \frac{32}{784}$.
    > From my experience, assigning too small a weight to the KL divergence term works fine for reconstruction, but it tends to degrade the quality of generated samples. This likely happens because the latent distribution drifts away from the standard normal distribution.

Progress on Validation Set:
    ![Training Progress](.\assets\progress.gif)

### Results
- Reconstruction: 
![Reconstruction](.\assets\recon.png)
- Generation: 
![Generation](.\assets\gen.png)
- Latent Space Visualization:
![Latent Space](.\assets\tsne.png)

### References
Textbook:
 - [1] The original paper: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
 - [2] The tutorial paper: [Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908)

Github Repositories:
 - [1] A collection of VAE variations: [AntixK/PyTorch-VAE](https://github.com/AntixK/PyTorch-VAE)
 - [2] The open issue#3 helps: [ethanluoyc/pytorch-vae](https://github.com/ethanluoyc/pytorch-vae)