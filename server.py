import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from models.cnn import CNN 

global_model = CNN()
optimizer = optim.Adam(global_model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    soft_targets = torch.nn.functional.softmax(teacher_logits / temperature, dim=1)
    soft_predictions = torch.nn.functional.log_softmax(student_logits / temperature, dim=1)
    return nn.KLDivLoss(reduction="batchmean")(soft_predictions, soft_targets)

class FKDStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        if not results:
            return None

        global_model.train()
        total_loss = 0
        num_samples = 0

        for client_params, num_examples, logits in results:
            logits = torch.tensor(logits)  # Convert to tensor
            student_logits = global_model(torch.randn_like(logits))  # Generate student logits
            loss = distillation_loss(student_logits, logits)  # Compute KD loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * num_examples
            num_samples += num_examples

        print(f"Round {rnd} - Distillation Loss: {total_loss / num_samples:.4f}")
        return global_model.state_dict()

if __name__ == "__main__":
    strategy = FKDStrategy()
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )