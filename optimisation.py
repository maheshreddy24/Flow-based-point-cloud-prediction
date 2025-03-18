import torch
import torch.nn as nn
import torch.optim as optim
import logging
from tqdm import tqdm
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
os.makedirs('experiments', exist_ok=True)

class optimisation_flow():
    def __init__(self, model, epochs, train_dataloader, device='cuda'):
        self.model = model
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.epochs = epochs
        self.train_dataloader = train_dataloader
        self.model = self.model.to(self.device)  # Fixes device issue

    def train(self):
        logging.info(f"Starting training for {self.epochs} epochs...")

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False)

            for input, output in progress_bar:
                input, output = input.to(self.device), output.to(self.device)
                self.optimizer.zero_grad()

                # Predict velocity field v_t(x)
                v_pred = self.model(input)

                # Compute ground truth velocity
                v_gt = output - input  # Actual displacement

                # Compute loss
                loss = self.criterion(v_pred, v_gt)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=f"{loss.item():.6f}")

            if epoch%100 ==0:
                save_path = f'experiments/model{epoch}.pth'
                # Save the model's state_dict
                torch.save(self.model.state_dict(), save_path)
                
            avg_loss = epoch_loss / len(self.train_dataloader)
            logging.info(f"Epoch {epoch+1}/{self.epochs} - Average Loss: {avg_loss:.6f}")

    def test(self, test_dataloader, steps=10):
        logging.info(f"Starting testing with {steps} steps...")

        for input, output in tqdm(test_dataloader, desc="Testing", leave=True):
            input = input.to(self.device)  # Move input to device
            xt = input.clone()

            for i, t in enumerate(tqdm(torch.linspace(0, 1, steps), desc="Predicting", leave=False)):
                t_expanded = t.expand(xt.size(0)).to(self.device)
                pred = self.model(xt, t_expanded)  # Predict velocity field
                xt = xt + (1 / steps) * pred  # Euler step

            xt_np = xt.squeeze(0).cpu().detach().numpy()
        
        logging.info("Testing completed.")
        return xt_np
