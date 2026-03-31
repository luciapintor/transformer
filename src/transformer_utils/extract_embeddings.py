import torch as nn

def extract_embeddings(model, dataloader):
    model.eval()
    embeddings = []
    
    with nn.no_grad():
        for x, _ in dataloader:
            _, z = model(x)
            embeddings.append(z)
    
    return nn.cat(embeddings, dim=0)