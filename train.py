import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from dataset import get_data
from torch.utils.data import DataLoader
from sklearn import metrics
from utils import get_predictions
import torch.nn.functional as F


class Santander(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(Santander, self).__init__()
        self.bn = nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(input_size//2*hidden_dim, 1)
        # self.net = nn.Sequential(
        #     nn.BatchNorm1d(input_size),
        #     nn.Linear(input_size, 50),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(50, 1),
        # )
    def forward(self, x):
        bs = x.shape[0]
        x = self.bn(x) # current shape is (bs, 200)
        orig_features = x[:, :200].unsqueeze(2) # (N, 200, 1)
        new_features = x[:, 200:].unsqueeze(2)
        x = torch.cat([orig_features, new_features], dim =2) # (N, 200, 2)
        x = F.relu(self.fc1(x)).reshape(bs, -1) # (bs, 200 * hidden_dim)
        return torch.sigmoid(self.fc2(x)).view(-1)
    
DEVICE = 'cuda' 
# if torch.cuda.is_available() else 'cpu'
model = Santander(input_size=400, hidden_dim=100).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr = 2e-3, weight_decay=1e-4)
loss_fn = nn.BCELoss()
train_ds, val_ds, test_ds, test_ids = get_data()
train_loader = DataLoader(train_ds, batch_size=1024, shuffle = True)
val_loader = DataLoader(val_ds, batch_size=1024)
test_loader = DataLoader(test_ds, batch_size=1024)

for epoch in range(20):
    # data, targets = next(iter(train_loader))
    probabilities, true = get_predictions(val_loader, model, device = DEVICE)
    print(f"Validation ROC: {metrics.roc_auc_score(true, probabilities)}")    
    for batch_idx, (data,targets) in enumerate(tqdm(train_loader)):
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)
        
        scores = model(data)
        loss = loss_fn(scores, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
from utils import get_submission
get_submission(model, test_loader, test_ids, DEVICE)