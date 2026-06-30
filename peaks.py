#peaks classification
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
#Matlab peaks 
def peaks(x,y):
    f1 = 3*(1-x)**2 * np.exp(-x**2 - (y+1)**2)
    f2 = 10*(x/5 - x**3 - y**5)* np.exp(-x**2-y**2)
    f3 = np.exp(-(x+1)**2 - y**2)/3
    return f1 - f2 - f3
#dataset
def generate_peaks(n_total = 5000, n_classes = 5, seed = 42):
    rng = np.random.default_rng(seed)
    #discretization on the 256x256 grid, x, y in [-3,3]
    x1 = np.linspace(-3,3,256)
    y = np.linspace(-3,3,256)
    X1, Y = np.meshgrid(x1,y)
    F = peaks(X1.ravel(), Y.ravel())
    #classes
    quantiles = np.quantile(F, np.linspace(0,1, n_classes+1))
    labels = np.digitize(F, quantiles[1:-1])
    #samples
    n_per_class = n_total//n_classes
    x_list = [] 
    f_list = []
    coords = np.column_stack([X1.ravel(), Y.ravel()])
    for cls in range(n_classes):
        idx = np.where(labels == cls)[0]
        chosen = rng.choice(idx, size = n_per_class, replace = False)
        x_list.append(coords[chosen])
        f_list.append(np.full(n_per_class, cls))
    X = np.vstack(x_list).astype(np.float32)
    f = np.concatenate(f_list).astype(np.int64)
    #shuffle
    perm = rng.permutation(len(f))
    X, f = X[perm], f[perm]
    #training and validation sets
    n_train = int(0.8*len(f))
    X_tr, y_tr = X[:n_train], f[:n_train]
    X_val, y_val = X[n_train:], f[n_train:]
    to_t = lambda a, dt:torch.tensor(a, dtype=dt, device = DEVICE)
    return to_t(X_tr, torch.float32), to_t(y_tr, torch.long), to_t(X_val, torch.float32), to_t(y_val, torch.long), X_tr, y_tr
#propagation
#activation:tanh
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
    def __init__(self, n, n_layers, h=0.1, gamma = 1e-10):
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
#model --> multiclass classifier --> softmax
class DNNClassifier(nn.Module):
    def __init__(self, prop_module, n_features,n_classes):
        super().__init__()
        self.prop = prop_module
        self.W = nn.Parameter(torch.randn(n_features, n_classes)*0.01)
        self.mu = nn.Parameter(torch.zeros(n_classes))
    def forward(self, Y):
        YN = self.prop(Y)
        logits = YN @ self.W + self.mu
        return logits, YN 
    def loss(self, Y, C, alpha = 1e-4, beta = 1e-3):
        logits, _ = self.forward(Y)
        #cross entropy with softmax 
        #softmax is in cross_entropy
        ce = nn.functional.cross_entropy(logits, C)
        reg = alpha*self.prop.smoothness_reg()
        reg_W = beta*torch.sum(self.W**2)
        return ce + reg + reg_W
    @torch.no_grad()
    def accuracy(self, Y, C):
        logits, _ = self.forward(Y)
        preds = logits.argmax(dim=1)
        return (preds ==C).float().mean().item()
#training
def train(model, X_tr, y_tr, X_val, y_val, n_epochs = 3000, lr=1e-2, alpha=1e-4, beta=1e-3):
    opt = optim.Adam(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max = N_EPOCHS)
    losses, accs =[], []
    best_acc = 0.0
    best_state = {k: v.clone() for k,v in model.state_dict().items()}
    for ep in range(n_epochs):
        model.train()
        opt.zero_grad()
        l = model.loss(X_tr, y_tr, alpha, beta)
        l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm= 1.0)
        opt.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            losses.append(l.item())
            acc = model.accuracy(X_val, y_val)
            accs.append(acc)
        if acc>best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return losses, accs
#decision boundary,
@torch.no_grad()
def decision_boundary(model, mean, std, margin = 0.2, res=300):
    x = np.linspace(-3-margin, 3+margin, res)
    y = np.linspace(-3-margin, 3+margin, res)
    xx, yy = np.meshgrid(x,y)
    grid_np=np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)
    #normalization
    mean_np = mean.cpu().numpy()
    std_np = std.cpu().numpy()
    grid_np = (grid_np -mean_np)/std_np
    grid = torch.tensor(grid_np, device=DEVICE)
    logits, _ = model(grid)
    Z = logits.argmax(dim=1).cpu().numpy().reshape(xx.shape)
    return xx, yy, Z
#main.
if __name__=="__main__":
    X_tr, y_tr, X_val, y_val, X_tr_np, y_tr_np = generate_peaks(5000,5)
    print(f"Train: {len(y_tr)}  Val: {len(y_val)}")
    mean = X_tr.mean(0)
    std = X_tr.std(0)
    X_tr = (X_tr - mean)/std
    X_val = (X_val - mean)/std
    N_CLASSES = 5
    #TIME = 20
    N_LAYERS = 128
    H = 1.75
    N_EPOCHS = 10000
    LR       = 2e-2
    ALPHA    = 1e-10
    BETA     = 1e-3
    configs = [
        dict(arch = "antisym", label = "Anti-Sym ResNet", color = "tab:orange"),
        dict(arch = "resnet", label = "ResNet", color = "tab:blue"),
    ]
    results = {}
    print("Architectures:\n")
    for cfg in configs:
        if cfg["arch"] == "resnet":
            prop = ResNetProp(2,N_LAYERS, H)
            print(" - ResNet\n")
        else:
            prop = AntiSymProp(2, N_LAYERS, H)
            print(" - Antisymmetric\n")
        model = DNNClassifier(prop, n_features=2,
                              n_classes=N_CLASSES).to(DEVICE)
 
        losses, accs = train(model, X_tr, y_tr, X_val, y_val,
                             n_epochs=N_EPOCHS, lr=LR,
                             alpha=ALPHA, beta=BETA)
 
        model.eval()
        with torch.no_grad():
            _, YN = model(X_tr)
        YN = YN.detach().cpu().numpy()
        xx, yy, Z = decision_boundary(model = model, mean = mean, std = std)
        results[cfg["arch"]] = dict(
                losses=losses, accs=accs,
                YN=YN, xx=xx, yy=yy, Z=Z,
                best_acc=max(accs), **cfg
            )
    #plot
    #peaks function 
    x1g = np.linspace(-3, 3, 300)
    x2g = np.linspace(-3, 3, 300)
    XX, YY = np.meshgrid(x1g, x2g)
    FF = peaks(XX, YY)
    #main grid
    n_archs    = len(configs)
    cmap_cls   = plt.cm.get_cmap("tab10", N_CLASSES)
    cls_colors = [cmap_cls(i) for i in range(N_CLASSES)]
 
    fig1 = plt.figure(figsize=(15, 4*n_archs))
    fig1.suptitle("Peaks [Adam]",
                  fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(n_archs, 3, figure=fig1,
                           hspace=0.45, wspace=0.3)
 
    for row, (arch, res) in enumerate(results.items()):
 
        # col 0: input data colored by class
        ax = fig1.add_subplot(gs[row, 0])
        for cls in range(N_CLASSES):
            m = y_tr_np == cls
            ax.scatter(X_tr_np[m, 0], X_tr_np[m, 1],
                       s=5, c=[cls_colors[cls]], alpha=0.5,
                       label=f"Class {cls}")
        ax.set_title(f"{res['label']}\nInput data", fontsize=9)
        ax.legend(fontsize=6, ncol=2); ax.set_aspect("equal"); ax.axis("off")
 
        # col 1: propagated features
        ax = fig1.add_subplot(gs[row, 1])
        YN = res["YN"]
        for cls in range(N_CLASSES):
            m = y_tr_np == cls
            ax.scatter(YN[m, 0], YN[m, 1],
                       s=5, c=[cls_colors[cls]], alpha=0.5)
        ax.set_title("Propagated features\n(output layer)", fontsize=9)
        ax.set_aspect("equal"); ax.axis("off")
 
        # col 2: decision boundary
        ax = fig1.add_subplot(gs[row, 2])
        ax.contourf(res["xx"], res["yy"], res["Z"],
                    levels=np.arange(-0.5, N_CLASSES, 1),
                    cmap="tab10", alpha=0.6)
        for cls in range(N_CLASSES):
            m = y_tr_np == cls
            ax.scatter(X_tr_np[m, 0], X_tr_np[m, 1],
                       s=4, c=[cls_colors[cls]], alpha=0.5)
        ax.set_title(f"Decision boundary\nVal Acc = {res['best_acc']:.3f}",
                     fontsize=9)
        ax.set_aspect("equal"); ax.axis("off")
 
    fig1.savefig("peaks_results.png", dpi=150, bbox_inches="tight")
    #training curves
    fig2, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig2.suptitle("Peaks – Training Curves [Adam]",
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
    fig2.savefig("peaks_curves.png", dpi=150, bbox_inches="tight")
    #accuracy bar chart.
    fig3, ax = plt.subplots(figsize=(7, 4))
    labels = [res["label"]    for res in results.values()]
    accs   = [res["best_acc"] for res in results.values()]
    colors = [res["color"]    for res in results.values()]
    bars   = ax.bar(labels, accs, color=colors, alpha=0.8, edgecolor="black")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.axhline(1.0, ls="--", color="gray", lw=0.8)
    ax.set_ylabel("Best Validation Accuracy", fontsize=11)
    ax.set_title("Peaks – Accuracy Comparison", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    fig3.tight_layout()
    fig3.savefig("peaks_comparison.png", dpi=150, bbox_inches="tight")
    #peaks function surface
    fig4, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig4.suptitle("Peaks Function", fontsize=12, fontweight="bold")
    im = axes[0].imshow(FF, extent=[-3, 3, -3, 3],
                        origin="lower", cmap="RdBu_r", aspect="auto")
    plt.colorbar(im, ax=axes[0], label="f(x)")
    axes[0].set_title("Peaks function heatmap", fontsize=10)
    quantiles = np.quantile(FF, np.linspace(0, 1, N_CLASSES+1))
    axes[1].contourf(XX, YY, FF, levels=quantiles, cmap="tab10", alpha=0.7)
    axes[1].contour(XX,  YY, FF, levels=quantiles, colors="k",
                    linewidths=0.8, alpha=0.5)
    axes[1].set_title("Class boundaries (5 classes)", fontsize=10)
    axes[1].set_aspect("equal")
    fig4.tight_layout()
    fig4.savefig("peaks_heatmap.png", dpi=150, bbox_inches="tight")
    print("\nFigures saved:")
    print("  peaks_results.png")
    print("  peaks_curves.png")
    print("  peaks_comparison.png")
    print("  peaks_function.png")
    plt.show()