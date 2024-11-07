import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.stats as stats


class SimpleNormalizingFlow(nn.Module):
    def __init__(self):
        super(SimpleNormalizingFlow, self).__init__()
        self.base_dist = stats.norm(0, 1)

    def forward(self, z):
        # Define the forward transformation
        return torch.exp(z)

    def inverse(self, x):
        # Define the inverse transformation
        return torch.log(x)

    def log_det_jacobian(self, z):
        # Define the log-determinant of the Jacobian
        return z


def compute_log_likelihood(x, nf):
    # Step 1: Compute the inverse transformation
    z = nf.inverse(x)

    # Step 2: Compute the log-determinant of the Jacobian
    log_det_jac = nf.log_det_jacobian(z)

    # Step 3: Compute the log-likelihood of z under the base distribution
    log_p_z = nf.base_dist.logpdf(z.detach().numpy())

    # Step 4: Compute the log-likelihood of x under the normalizing flow
    log_p_x = log_p_z + log_det_jac.numpy()

    return log_p_x


def train_normalizing_flow(nf, train_loader, optimizer, epochs):
    nf.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            log_likelihood = compute_log_likelihood(batch, nf)
            loss = -log_likelihood.mean()  # Negative log-likelihood
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}")


def evaluate_normalizing_flow(nf, test_loader):
    nf.eval()
    total_log_likelihood = 0
    with torch.no_grad():
        for batch in test_loader:
            log_likelihood = compute_log_likelihood(batch, nf)
            total_log_likelihood += log_likelihood.sum().item()
    avg_log_likelihood = total_log_likelihood / len(test_loader.dataset)
    print(f"Test Log-Likelihood: {avg_log_likelihood}")
    return avg_log_likelihood


if __name__ == "__main__":
    # Generate a simple dataset
    np.random.seed(0)
    data = np.random.exponential(scale=1.0, size=(1000, 1))
    data = torch.tensor(data, dtype=torch.float32)

    # Split the dataset into training and testing sets
    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size
    train_data, test_data = torch.utils.data.random_split(data, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

    # Initialize the normalizing flow and optimizer
    nf = SimpleNormalizingFlow()
    optimizer = optim.Adam(nf.parameters(), lr=0.001)

    # Train the normalizing flow
    train_normalizing_flow(nf, train_loader, optimizer, epochs=10)

    # Evaluate the normalizing flow on the test set
    test_log_likelihood = evaluate_normalizing_flow(nf, test_loader)

    # Compute the negative log-likelihood (NLL) on the test set
    nll = -test_log_likelihood
    print(f"Test Negative Log-Likelihood (NLL): {nll}")


# Optional
from sklearn.neighbors import NearestNeighbors


def compute_precision_recall(real_data, generated_data, k=5):
    """_summary_
    image quality (precision) and diversity (recall).
    Recall measures the proportion of real data samples that are covered by the generated samples. It quantifies how well the model covers the diversity of the real data distribution.
    Precision measures the proportion of generated samples that are realistic. In other words, it quantifies how many of the generated samples are close to the real data manifold.
    Computation:
    For each real data sample, find the nearest neighbor in the set of generated samples.
    Compute the average distance between the real data samples and their nearest neighbors in the generated set.
    Can even be used as a training loss: https://proceedings.neurips.cc/paper_files/paper/2023/hash/67159f1c0cab15dd34c76a5dd830a389-Abstract-Conference.html
    """
    # Fit a nearest neighbors model on the real data
    nbrs_real = NearestNeighbors(n_neighbors=k).fit(real_data)
    nbrs_generated = NearestNeighbors(n_neighbors=k).fit(generated_data)

    # Compute distances for precision
    distances_real, _ = nbrs_real.kneighbors(generated_data)
    precision = np.mean(distances_real[:, -1])

    # Compute distances for recall
    distances_generated, _ = nbrs_generated.kneighbors(real_data)
    recall = np.mean(distances_generated[:, -1])

    return precision, recall


def compute_f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)


# Generate a simple dataset
np.random.seed(0)
real_data = np.random.exponential(scale=1.0, size=(1000, 2))

# Generate samples from the model (for illustration, we use the same distribution)
generated_data = np.random.exponential(scale=1.0, size=(1000, 2))

# Compute precision and recall
precision, recall = compute_precision_recall(real_data, generated_data)
f1_score = compute_f1_score(precision, recall)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")
