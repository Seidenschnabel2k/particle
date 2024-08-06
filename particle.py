import torch 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.constants import g 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_particles = 400

E = 100e9 # Pa
nu = .33 
r = .015e-3 # m
kn = (2*r)**.5/3 * E/(1-nu**2) # unit inconsistent Todo


class Particle:
    """
    x -> vector(3)
    """
    def __init__(self, 
                 num_particles,
                 alpha = .1,
                 rho = 4500,
                 r=.015, 
                 e=.9,
                 nu=.33):
        
        self.device = device
        self.x = torch.rand((num_particles, 3), device=device)
        self.v = torch.zeros((num_particles, 3), device=device)
        self.a = torch.zeros((num_particles, 3), device=device)
        self.m = 3/4 * r**3 * rho
        self.k = (2*r)**.5/3 * E/(1-nu**2)
        self.dt = alpha * (self.m /2/self.k)**.5 
        self.e = e
        self.r = r
        self.nu = nu
    def calc_f(self):
        self.a.zero_()
        # Compute pairwise distance vectors
        diff = self.x.unsqueeze(0) - self.x.unsqueeze(1)  # Shape: (N, N, 3)
        dist_sq = diff.pow(2).sum(dim=-1)  # Shape: (N, N)

        # Avoid division by zero for distance
        mask = dist_sq > 0
        dist = dist_sq.sqrt() + (~mask).float()


    def update(self):
        self.v += self.a * self.dt
        self.x += self.v * self.dt

    def run(self,num_steps):
        for _ in range(num_steps):
            self.update()


sim = Particle(400)
sim.run(num_steps=1000)

print(sim.x.T[1])
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(sim.x.T[0].cpu(),sim.x.T[1].cpu(),sim.x.T[2].cpu())
plt.show()
