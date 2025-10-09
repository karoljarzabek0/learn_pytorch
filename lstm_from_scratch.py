import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

class LSTM_scratch(nn.Module):
    def __init__(self):
        super().__init__()

        mean = torch.tensor(0.0)
        std = torch.tensor(1.0)

        self.wlr1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wlr2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.blr1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.wpr1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wpr2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bpr1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.wp1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wp2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bp1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.wo1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wo2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bo1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def lstm_unit(self, input_value, long_memory, short_memory):
        long_remember_percent = torch.sigmoid((short_memory * self.wlr1) + (input_value * self.wlr2) + self.blr1)

        potential_remember_percent = torch.sigmoid((short_memory * self.wpr1) + (input_value * self.wpr2) + self.bpr1)

        potential_memory = torch.tanh((short_memory * self.wp1) + (input_value * self.wp2) + self.bp1)

        updated_long_memory = ((long_memory * long_remember_percent) + potential_remember_percent * potential_memory)

        output_percent = torch.sigmoid((short_memory * self.wo1) + (input_value * self.wo2) + self.bo1)

        updated_short_memory = torch.tanh(updated_long_memory) * output_percent

        return([updated_long_memory, updated_short_memory])


    def forward(self, input):
        long_memory = 0
        short_memory = 0
        day1 = input[0]
        day2 = input[1]
        day3 = input[2]
        day4 = input[3]

        long_memory, short_memory = self.lstm_unit(day1, long_memory, short_memory)
        long_memory, short_memory = self.lstm_unit(day2, long_memory, short_memory)
        long_memory, short_memory = self.lstm_unit(day3, long_memory, short_memory)
        long_memory, short_memory = self.lstm_unit(day4, long_memory, short_memory)

        return short_memory

    def configure_optimizers(self):
        return Adam(self.parameters())

    def training_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i[0])
        loss = (output_i - label_i)**2

        return loss, output_i, label_i

model = LSTM_scratch()

print("Compare observed and predicted values for untrained model:")
print("Company A: Observed = 0, Predicted = ",
        model(torch.tensor([0., 0.5, 0.25, 1.])).detach())
print("Company B: Observed = 0, Predicted = ",
        model(torch.tensor([1., 0.5, 0.25, 1.])).detach())

inputs = torch.tensor([[0., 0.5, 0.25, 1.], [1., 0.5, 0.25, 1.]])
labels = torch.tensor([0., 1.])

dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset)

EPOCHS = 5000

optimizer = model.configure_optimizers()

for epoch in range(EPOCHS):
    total_loss = 0.0
    model.train()
    for i, batch in enumerate(dataloader):
        # Zero the gradients
        optimizer.zero_grad()
        
        # Perform a forward pass and calculate the loss
        loss, output_i, label_i = model.training_step(batch, i)
        
        # Perform a backward pass
        loss.backward()
        
        # Update the weights
        optimizer.step()

        total_loss += loss.item()

        if label_i.item() == 0:
            writer.add_scalar("Company 0 price/out_0", output_i.item(), epoch)
        else:
            writer.add_scalar("Company 1 price/out_1", output_i.item(), epoch)

    avg_loss = total_loss / len(dataloader)
    writer.add_scalar("Loss/train_epoch", avg_loss, epoch)
    
    if epoch % 100 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")


print("\nCompare observed and predicted values for trained model:")
print("Company A: Observed = 0, Predicted = ",
        model(torch.tensor([0., 0.5, 0.25, 1.])).detach())
print("Company B: Observed = 1, Predicted = ",
        model(torch.tensor([1., 0.5, 0.25, 1.])).detach())

pred_0 = model(torch.tensor([0., 0.5, 0.25, 1.])).detach()

torch.save(model.state_dict(), f"lstm_scratch_model_{pred_0}.pth")

for name, param in model.named_parameters():
    print(f"{name:10} | shape: {param.shape} | value: {param.data}")

