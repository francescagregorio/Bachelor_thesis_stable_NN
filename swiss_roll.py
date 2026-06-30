import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
torch.manual_seed(42)
np.random.seed(42)
DEVICE = "cpu"
print(f"Device: {DEVICE}")
#dataset
def generate_swiss_roll(n_points=513):
    r = np.linspace(0,1, n_points)
    th = np.linspace(0,4*np.pi, n_points)
    X1 = r[:,None]*np.column_stack([np.cos(th), np.sin(th)])
    X2 = (r[:,None]+0.2)*np.column_stack([np.cos(th), np.sin(th)])
    X = np.vstack([X1, X2]).astype(np.float32)
    y = np.array([0]*n_points + [1]*n_points, dtype = np.float32)
    #validation
    train_mask = np.ones(len(y), dtype = bool)
    train_mask[::2] = False
    val_mask = ~train_mask
    to_t = lambda a: torch.tensor(a,device=DEVICE)
    return (to_t(X[train_mask]), to_t(y[train_mask]), to_t(X[val_mask]), to_t(y[val_mask]), X[train_mask], y[train_mask])
#propagation --> implementation of the classes (ResNet, AntiSym)
#for every propagation we need a smoothness regularization function
class ResNetProp(nn.Module):
    #standard ResNet equation
    def __init__(self, n, n_layers, h = 0.1):
        super().__init__()
        self.h = h
        self.Ks = nn.ParameterList([nn.Parameter(torch.randn(n,n)/(np.sqrt(2*n))) for i in range(n_layers)])
        self.bs = nn.ParameterList([nn.Parameter(torch.zeros(n)) for i in range(n_layers)])
    def forward(self, Y):
        for K, b in zip(self.Ks, self.bs):
            Y = Y + self.h*torch.tanh(Y@K + b)
            
        return Y
    def smoothness_reg(self):
        #smoothness regularization --> Evaluation of R(K) = 1/(2h)*(sum_j (||K_j - K_{k-1}||_F)^2)
        loss = 0.0
        for j in range(1, len(self.Ks)):
            loss += torch.sum((self.Ks[j]-self.Ks[j-1])**2)
            loss += torch.sum((self.bs[j]-self.bs[j-1])**2)
        return loss/(2*self.h)
class AntiSymProp(nn.Module):
    def __init__(self, n, n_layers, h=0.1, gamma = 1e-2):
        super().__init__()
        self.h = h
        self.gamma = gamma
        self.register_buffer('I', torch.eye(n))
        self.Ks = nn.ParameterList([nn.Parameter(torch.randn(n,n)/np.sqrt(2*n)) for i in range(n_layers)])
        self.bs = nn.ParameterList([nn.Parameter(torch.zeros(n)) for i in range(n_layers)])
    def forward(self,Y):
        for K, b in zip(self.Ks, self.bs):
            A = 0.5*(K - K.T) - self.gamma * self.I
            Y = Y + self.h * torch.tanh(Y@A + b)
            
        return Y
    def smoothness_reg(self):
        loss = 0.0
        for j in range(1, len(self.Ks)):
            loss += torch.sum((self.Ks[j]-self.Ks[j-1])**2)
            loss += torch.sum((self.bs[j]-self.bs[j-1])**2)
        return loss/(2*self.h)
#model --> DNN Classifier
class DNNClassifier(nn.Module):
    def __init__(self, prop_module,n_features):
        super().__init__()
        self.prop= prop_module
        self.W = nn.Parameter(torch.randn(n_features,1)*0.01) 
        self.mu = nn.Parameter(torch.zeros(1))
    def forward(self, Y):
        YN = self.prop(Y)
        logits = YN @ self.W + self.mu
        return logits, YN
    def loss(self, Y, C, alpha = 1e-4, beta = 1e-3):
        logits, _ = self.forward(Y)
        bce =nn.functional.binary_cross_entropy_with_logits(logits.view(-1), C.view(-1))
        reg = alpha * self.prop.smoothness_reg()
        reg_W = beta*torch.sum(self.W**2)
        return bce + reg + reg_W
    @torch.no_grad()
    def accuracy(self, Y, C):
        logits, _ = self.forward(Y)
        pred = (torch.sigmoid(logits.view(-1))>=0.5).float()
        return (pred == C.view(-1).float()).float().mean().item()
#training --> Adam, 1 level.
def train(model, X_tr, y_tr, X_val, y_val, n_epochs = 2000, lr = 2e-2, alpha = 1e-4, beta = 1e-3):
    opt = optim.Adam(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max = N_EPOCHS)
    best_acc=0.0
    best_state = {k:v.clone() for k,v in model.state_dict().items()}
    losses = []
    val_accs = []
    for ep in range(n_epochs):
        model.train()
        opt.zero_grad()
        loss = model.loss(X_tr, y_tr, alpha, beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        scheduler.step()
        model.eval()
        
        with torch.no_grad():
            losses.append(loss.item())
            acc = model.accuracy(X_val, y_val)
            val_accs.append(acc)
        if acc> best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return losses, val_accs
#decision boundary
def decision_boundary(model, X_tr_np, mean, std, margin=0.3, res=300):
    x_min = X_tr_np[:, 0].min() - margin
    x_max = X_tr_np[:, 0].max() + margin
    y_min = X_tr_np[:, 1].min() - margin
    y_max = X_tr_np[:, 1].max() + margin
 
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, res),
                         np.linspace(y_min, y_max, res))
    grid_np=np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)
    #normalization
    mean_np = mean.cpu().numpy()
    std_np = std.cpu().numpy()
    grid_np = (grid_np -mean_np)/std_np
    grid = torch.tensor(grid_np, device=DEVICE)
    logits, _ = model(grid)
    Z = torch.sigmoid(logits).detach().cpu().numpy().reshape(xx.shape)
    return xx, yy, Z

#main
if __name__ == "__main__":
   X_tr, y_tr, X_val, y_val, X_tr_np, y_tr_np = generate_swiss_roll(513)
   mean = X_tr.mean(0)
   std = X_tr.std(0)
   X_tr = (X_tr - mean)/std
   X_val = (X_val - mean)/std
   print(f"Train: {len(y_tr)}  Val: {len(y_val)}")
   TIME = 20
   N_LAYERS = 128
   
   H = 1.5
   
   N_EPOCHS = 5000
   LR       = 1e-2
   ALPHA    = 1e-10
   BETA     = 1e-3
   configs = [
       
        dict(arch="antisym", label="Anti-Sym ResNet",      color="tab:orange"),
        dict(arch = "ResNet", label="ResNet", color = "tab:blue"),
       
    ]
   results = {}
   print("Architectures:")
   for cfg in configs:
       print(f" - {cfg['label']}\n")
       if cfg["arch"] == "ResNet":
            prop = ResNetProp(2, N_LAYERS, H)
       else:
            prop = AntiSymProp(2, N_LAYERS, H)
       model = DNNClassifier(prop, n_features=2).to(DEVICE)
 
       losses, accs = train(model, X_tr, y_tr, X_val, y_val,
                             n_epochs=N_EPOCHS, lr=LR,
                             alpha=ALPHA, beta=BETA)
 
       model.eval()
       with torch.no_grad():
            _, YN = model(X_tr)
       YN = YN.cpu().numpy()
 
       xx, yy, Z = decision_boundary(model, X_tr_np, mean, std)
 
       results[cfg["arch"]] = dict(
            losses=losses, accs=accs,
            YN=YN, xx=xx, yy=yy, Z=Z,
            best_acc=max(accs), **cfg
        )
    #plot 
   cls_colors = ["tab:blue", "tab:red"]
   fig1 = plt.figure(figsize=(14,9))
   fig1.suptitle("Swiss Roll", fontsize=13, fontweight = "bold")
   gs = gridspec.GridSpec(2,3, figure= fig1, hspace =0.4, wspace=0.3)
    #data, features, BoundaryError
   for row, (arch, res) in enumerate(results.items()):
     
        # col 0: input data
        ax = fig1.add_subplot(gs[row, 0])
        for cls in [0, 1]:
            m = y_tr_np == cls
            ax.scatter(X_tr_np[m, 0], X_tr_np[m, 1],
                       s=6, c=cls_colors[cls], alpha=0.6,
                       label=f"Class {cls}")
        ax.set_title(f"{res['label']}\nInput data", fontsize=9)
        ax.legend(fontsize=7); ax.set_aspect("equal"); ax.axis("off")
 
        # col 1: propagated features
        ax = fig1.add_subplot(gs[row, 1])
        YN = res["YN"]
        for cls in [0, 1]:
            m = y_tr_np == cls
            ax.scatter(YN[m, 0], YN[m, 1],
                       s=6, c=cls_colors[cls], alpha=0.6)
        ax.set_title("Propagated features\n(output layer)", fontsize=9)
        ax.set_aspect("equal"); ax.axis("off")
 
        # col 2: decision boundary
        ax = fig1.add_subplot(gs[row, 2])
        ax.contourf(res["xx"], res["yy"], res["Z"],
                    levels=50, cmap="RdBu_r", alpha=0.75)
        ax.contour(res["xx"], res["yy"], res["Z"],
                   levels=[0.5], colors="k", linewidths=1.2)
        for cls in [0, 1]:
            m = y_tr_np == cls
            ax.scatter(X_tr_np[m, 0], X_tr_np[m, 1],
                       s=4, c=cls_colors[cls], alpha=0.5)
        ax.set_title(f"Decision boundary\nVal Acc = {res['best_acc']:.3f}",
                     fontsize=9)
        ax.set_aspect("equal"); ax.axis("off")
 
   fig1.savefig("swiss_roll_results.png", dpi=150, bbox_inches="tight")
#PLOT 2: training curves 
   fig2, axes = plt.subplots(1, 2, figsize=(12, 4))
   fig2.suptitle("Swiss Roll – Training Curves [Adam]",
                  fontsize=12, fontweight="bold")
 
   for arch, res in results.items():
        axes[0].plot(res["losses"], color=res["color"],
                     label=res["label"], linewidth=1.5)
        axes[1].plot(res["accs"],   color=res["color"],
                     label=res["label"], linewidth=1.5)
 
   axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Training Loss")
   axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
   axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation Accuracy")
   axes[1].set_title("Accuracy"); axes[1].legend()
   axes[1].axhline(1.0, ls="--", color="gray", lw=0.8)
   axes[1].set_ylim(0, 1.05); axes[1].grid(True, alpha=0.3)
   fig2.tight_layout()
   fig2.savefig("swiss_roll_curves.png", dpi=150, bbox_inches="tight")
 
    # PLOT 3: probability maps
   fig3, axes = plt.subplots(1, 2, figsize=(12, 5))
   fig3.suptitle("Swiss Roll – Probability Maps [Adam]",
                  fontsize=12, fontweight="bold")
 
   for ax, (arch, res) in zip(axes, results.items()):
        im = ax.imshow(res["Z"],
                       extent=[res["xx"].min(), res["xx"].max(),
                                res["yy"].min(), res["yy"].max()],
                       origin="lower", cmap="RdBu_r",
                       vmin=0, vmax=1, aspect="auto")
        ax.contour(res["xx"], res["yy"], res["Z"],
                   levels=[0.5], colors="k", linewidths=1.2)
        for cls in [0, 1]:
            m = y_tr_np == cls
            ax.scatter(X_tr_np[m, 0], X_tr_np[m, 1],
                       s=4, c=cls_colors[cls], alpha=0.5)
        plt.colorbar(im, ax=ax, label="P(class=1)")
        ax.set_title(f"{res['label']}\nVal Acc = {res['best_acc']:.3f}",
                     fontsize=10)
 
   fig3.tight_layout()
   fig3.savefig("swiss_roll_probmaps.png", dpi=150, bbox_inches="tight")
 
    #accuracy bar chart
   fig4, ax = plt.subplots(figsize=(6, 4))
   labels = [res["label"]    for res in results.values()]
   accs   = [res["best_acc"] for res in results.values()]
   colors = [res["color"]    for res in results.values()]
   bars   = ax.bar(labels, accs, color=colors, alpha=0.8, edgecolor="black")
   ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=11)
   ax.set_ylim(0, 1.15)
   ax.axhline(1.0, ls="--", color="gray", lw=0.8)
   ax.set_ylabel("Best Validation Accuracy", fontsize=11)
   ax.set_title("Swiss Roll – Accuracy Comparison", fontsize=11)
   ax.grid(True, axis="y", alpha=0.3)
   fig4.tight_layout()
   fig4.savefig("swiss_roll_comparison.png", dpi=150, bbox_inches="tight")
 
   print("\nFigures saved:")
   print("  swiss_roll_results.png")
   print("  swiss_roll_curves.png")
   print("  swiss_roll_probmaps.png")
   print("  swiss_roll_comparison.png")
   plt.show() 