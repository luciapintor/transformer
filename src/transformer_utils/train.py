import torch as nn

def train(model, dataloader, epochs=10, lr=1e-3):
    optimizer = nn.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.nn.MSELoss()
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        
        for x, _ in dataloader:  # ignore labels
            optimizer.zero_grad()
            
            x_hat, _ = model(x)
            loss = criterion(x_hat, x)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}: loss = {total_loss:.4f}")
