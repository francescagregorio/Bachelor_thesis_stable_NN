#example 1- stability
import torch
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
torch.manual_seed(42)
#weight matrices
K_plus = torch.tensor([[2.0, -2.0], [0.0, 2.0]], dtype=torch.float32)
K_minus = torch.tensor([[-2.0, 0.0], [2.0, -2.0]], dtype=torch.float32)
K_0 = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float32)
#resnet propagation
def resnet_forward(y0, K, N=10, h = 0.1, b = None):
    if b is None:
        b = torch.zeros(K.shape[0])
    traj = [y0.clone()]
    Y = y0.clone()
    for i in range(N):
        Y = Y + h*torch.tanh(Y@K +b)
        traj.append(Y.clone())
    return torch.cat(traj, dim=0)
#force field
@torch.no_grad()
def force_field(K, xlim = (-1.5, 1.5), ylim=(-1.5,1.5), n_grid = 15, h = 0.1):
    xs = np.linspace(xlim[0], xlim[1],  n_grid)
    ys = np.linspace(ylim[0], ylim[1], n_grid)
    XX, YY = np.meshgrid(xs, ys)
    points = torch.tensor(np.column_stack([XX.ravel(), YY.ravel()]), dtype=torch.float32)
    #dy = h*tanh(y@K)
    F = h*torch.tanh(points@K)
    FX = F[:,0].numpy().reshape(XX.shape)
    FY = F[:,1].numpy().reshape(YY.shape)
    return XX, YY, FX, FY
#main
if __name__=="__main__":
    #feature vectors
    y1 = torch.tensor([[0.1, 0.1]], dtype=torch.float32)
    y2 = -y1
    y3 = torch.tensor([[0.0, 0.5]], dtype=torch.float32)
    N = 10 #layers
    h = 0.1 #step size
    features = [y1, y2, y3]
    feat_colors = ["tab:orange", "tab:blue", "tab:green"]
    feat_labels = ["$y_1$", "$y_2$", "$y_3$"]
    matrices = [K_plus, K_minus, K_0]
    title = [  r"$\mathbf{K}_+$, $\lambda(\mathbf{K}_+) = 2$" + "\nUnstable forward propagation",
        r"$\mathbf{K}_-$, $\lambda(\mathbf{K}_-) = -2$" + "\nStable but ill-posed",
        r"$\mathbf{K}_0$, $\lambda(\mathbf{K}_0) = \pm i$" + "\nStable and well-posed",]
    fig,axes = plt.subplots(1,3,figsize=(15,5))
    fig.suptitle("Example: phase plane diagrams for ResNet with N=10 identical layers", fontsize=12, fontweight="bold")
    xlim = (-1.5, 1.5)
    ylim = (-1.5, 1.5)
    for ax, K, title in zip(axes, matrices, title):
        #force field
        XX, YY, FX, FY = force_field(K, xlim = xlim, ylim = ylim, n_grid = 15, h = h)
        #normalization
        norm = np.sqrt(FX**2 + FY**2)
        ax.quiver(XX, YY, FX/norm, FY/norm, color = "black", alpha = 0.3, scale = 25, width = 0.003)
        #trajectories
        for y0, color, label in zip(features, feat_colors, feat_labels):
            traj =resnet_forward(y0, K, N= N, h = h)
            traj_np = traj.numpy()
            #plot trajectory
            ax.plot(traj_np[:,0], traj_np[:,1], color = color, linewidth = 2.0, alpha = 0.9, label = label)
            ax.scatter(traj_np[0,0],  traj_np[0,1], color = color, s = 60, zorder = 5, marker = "^", edgecolors = "black", linewidths = 0.5)
            ax.scatter(traj_np[-1,0], traj_np[-1,1], color = color, s = 60, zorder = 5, marker = "^", edgecolors = "black", linewidths = 0.5)
            mid = len(traj_np)//2
            ax.annotate("", xy = (traj_np[mid+1, 0], traj_np[mid+1,1]), xytext =(traj_np[mid,0], traj_np[mid,1]), arrowprops=dict(arrowstyle="->", color = color, lw=1.5))
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.legend(fontsize=9, loc = "upper left")
        ax.grid(True, alpha = 0.2)
        ax.axhline(0, color="gray", lw = 0.5)
        ax.axvline(0, color="gray", lw = 0.5)
        ax.set_xlabel("$y_1$", fontsize=10)
        ax.set_ylabel("$y_2$", fontsize=10)
    plt.tight_layout()
    plt.savefig("ex1_phase_diag,png", dpi=150,bbox_inches="tight")
    print("Figure saved")
    plt.show()