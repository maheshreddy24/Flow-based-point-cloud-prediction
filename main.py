from datasets import pointCloudDataset
from models import PointCloudFlowNetwork
from optimisation import optimisation_flow
from torch.utils.data import DataLoader

def main():
    pass

if __name__ == '__main__':
    dataset = pointCloudDataset()
    x0, x1 = dataset[0]

    print(f"x0 Shape: {x0.shape}, x1 Shape: {x1.shape}, {len(dataset)}")
    train_dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    x0_batch, x1_batch = next(iter(train_dataloader))
    print(f"Batch Shape: {x0_batch.shape}, {x1_batch.shape}")

    model = PointCloudFlowNetwork()
    
    rn_ = optimisation_flow(model, epochs = 1000, train_dataloader = train_dataloader)
    rn_.train()
    # output = rn_.test(train_dataloader)
    