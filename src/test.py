import torch

# Esempio minimo: creo una lista e utilizzo torch.zeros
values = [1, 2, 3]
print("Lista originale:", values)

zeros_tensor = torch.zeros(len(values))
print("Tensor di zeri:", zeros_tensor)
print(len(set((zeros_tensor))))
