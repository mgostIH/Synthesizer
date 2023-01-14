import synthesizer as s
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N = 128
D = 64
H = 4
layers = 4
data_size = 100_000

epochs = 10
batch_size = 320
learning_rate = 1e-4
loss = nn.CrossEntropyLoss()
force_retrain = False

# Generate dataset of N sequences of heads/tails with uniform probability of length L
# Every sequence must start with a start token 2
def generate_dataset(N, L):
    U = torch.rand((N,1))
    X = torch.bernoulli(U.expand((N,L-1))).to(torch.long)
    return torch.cat((torch.ones((N,1)).to(torch.long) * 2, X), dim=1)

data = generate_dataset(data_size, N+1)

mask = torch.tril(torch.ones((N,N))).to(torch.int8).to(device)

model = s.Synthesizer(N, D, H, 3, layers, mask).to(device)
to_logits = nn.Linear(D, 3).to(device)
# optimize both model and to_logits
optimizer = torch.optim.Adam(list(model.parameters()) + list(to_logits.parameters()), lr=learning_rate)

def train():
    L = []
    for epoch in range(epochs):
        for i in range(0, data_size, batch_size):
            X = data[i:i+batch_size].to(device)
            optimizer.zero_grad()
            output = model(X[:, :-1])
            output = to_logits(output)
            loss_value = loss(output.transpose(1, 2), X[:,1:])
            loss_value.backward()
            optimizer.step()
            L.append(loss_value.item())
    return L

# Save the model,to_logits and losses, if they exist already don't retrain
if force_retrain:
    losses = train()
    torch.save(model.state_dict(), 'model.pt')
    torch.save(to_logits.state_dict(), 'to_logits.pt')
    torch.save(losses, 'losses.pt')
else:
    try:
        model.load_state_dict(torch.load('model.pt'))
        to_logits.load_state_dict(torch.load('to_logits.pt'))
        losses = torch.load('losses.pt')
    except:
        losses = train()
        torch.save(model.state_dict(), 'model.pt')
        torch.save(to_logits.state_dict(), 'to_logits.pt')
        torch.save(losses, 'losses.pt')

        

plt.loglog(losses)
plt.show()

# Generate a sequence of length N
X = generate_dataset(1, N+1)
print(X)
# Put model back on CPU
model = model.to('cpu')
to_logits = to_logits.to('cpu')
model.mask = model.mask.to('cpu')

output = model(X[:, :-1])
output = to_logits(output)
print(torch.softmax(output, dim=2))