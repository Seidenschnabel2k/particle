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
        self.positions_history = []

    def calc_f(self):
        self.a.zero_()
        # Compute pairwise distance vectors
        diff = self.x.unsqueeze(0) - self.x.unsqueeze(1)  # Shape: (N, N, 3)
        dist_sq = diff.pow(2).sum(dim=-1)  # Shape: (N, N)

        # Avoid division by zero for distance
        mask = dist_sq > 0
        dist = dist_sq.sqrt() + (~mask).float()
        touching = dist < 2*self.r
        overlap = 2*self.r - dist
        normal_force_magnitude = self.k * overlap * touching.float()
# Normal force direction
        normal_force_direction = diff / dist.unsqueeze(-1)  # Shape: (N, N, 3)
        normal_force_direction[dist == 0] = 0
        normal_forces = normal_force_magnitude.unsqueeze(-1) * normal_force_direction  # Shape: (N, N, 3)
        net_normal_forces = normal_forces.sum(dim=0)  # Shape: (N, 3)
        self.a = net_normal_forces / self.m

        



    def update(self):
        self.v += self.a * self.dt
        self.x += self.v * self.dt
        self.positions_history.append(self.x.cpu().clone())

    def run(self,num_steps):
        for _ in range(num_steps):
            self.calc_f()
            self.update()
            
    def get_positions_history(self):
        return self.positions_history


sim = Particle(4000)
sim.run(num_steps=100)
positions_history = sim.get_positions_history()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
scat = ax.scatter([], [], [])
#ax.scatter(sim.x.T[0].cpu(),sim.x.T[1].cpu(),sim.x.T[2].cpu())
#plt.show()
def update_plot(frame):
    positions = positions_history[frame]
    scat._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
    return scat,

ani = FuncAnimation(fig, update_plot, frames=len(positions_history), interval=50, blit=False)

plt.show()

