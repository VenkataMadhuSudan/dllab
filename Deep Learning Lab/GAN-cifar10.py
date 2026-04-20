import torch, torchvision, matplotlib.pyplot as plt, numpy as np
import torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
latent_dim, lr, betas, epochs = 100, 0.0002, (0.5, 0.999), 10

loader = torch.utils.data.DataLoader(
    datasets.CIFAR10("./data", train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3)
        ])),
    batch_size=32, shuffle=True
)

G = nn.Sequential(
    nn.Linear(latent_dim, 128*8*8), nn.ReLU(True),
    nn.Unflatten(1, (128, 8, 8)),
    nn.Upsample(scale_factor=2), nn.Conv2d(128, 128, 3, padding=1),
    nn.BatchNorm2d(128, momentum=0.78), nn.ReLU(True),
    nn.Upsample(scale_factor=2), nn.Conv2d(128, 64, 3, padding=1),
    nn.BatchNorm2d(64, momentum=0.78), nn.ReLU(True),
    nn.Conv2d(64, 3, 3, padding=1), nn.Tanh()
).to(device)

D = nn.Sequential(
    nn.Conv2d(3, 32, 3, 2, 1), nn.LeakyReLU(0.2, True), nn.Dropout(0.25),
    nn.Conv2d(32, 64, 3, 2, 1), nn.ZeroPad2d((0,1,0,1)),
    nn.BatchNorm2d(64, momentum=0.82), nn.LeakyReLU(0.25, True), nn.Dropout(0.25),
    nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128, momentum=0.82),
    nn.LeakyReLU(0.2, True), nn.Dropout(0.25),
    nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256, momentum=0.8),
    nn.LeakyReLU(0.25, True), nn.Dropout(0.25),
    nn.Flatten(), nn.Linear(256*5*5, 1), nn.Sigmoid()
).to(device)

loss = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=lr, betas=betas)
opt_D = optim.Adam(D.parameters(), lr=lr, betas=betas)

for epoch in range(epochs):
    for i, (real, _) in enumerate(loader, 1):
        real = real.to(device)
        b = real.size(0)
        valid, fake = torch.ones(b, 1, device=device), torch.zeros(b, 1, device=device)

        opt_D.zero_grad()
        z = torch.randn(b, latent_dim, device=device)
        fake_imgs = G(z)
        d_loss = (loss(D(real), valid) + loss(D(fake_imgs.detach()), fake)) / 2
        d_loss.backward()
        opt_D.step()

        opt_G.zero_grad()
        g_loss = loss(D(G(torch.randn(b, latent_dim, device=device))), valid)
        g_loss.backward()
        opt_G.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Batch [{i}/{len(loader)}] D Loss: {d_loss.item():.4f} G Loss: {g_loss.item():.4f}")

    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            grid = torchvision.utils.make_grid(G(torch.randn(16, latent_dim, device=device)).cpu(), nrow=4, normalize=True)
            plt.figure(figsize=(6, 6))
            plt.imshow(np.transpose(grid, (1, 2, 0)))
            plt.axis("off")
            plt.show()