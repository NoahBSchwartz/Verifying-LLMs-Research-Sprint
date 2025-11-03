import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from matplotlib.figure import Figure
import matplotlib
from sklearn.decomposition import PCA
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting
from scipy.stats import multivariate_normal

# Set random seed for reproducibility
torch.manual_seed(42)

# Define the neural network architecture
class ComparisonNetwork(nn.Module):
    def __init__(self):
        super(ComparisonNetwork, self).__init__()
        self.hidden = nn.Linear(2, 5)
        self.activation = nn.ReLU()
        self.output = nn.Linear(5, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x
    
    def forward_with_activations(self, x):
        # Forward pass that returns intermediate activations
        hidden_pre = self.hidden(x)
        hidden_post = self.activation(hidden_pre)
        output_pre = self.output(hidden_post)
        output_post = self.sigmoid(output_pre)
        
        return {
            'input': x,
            'hidden_pre': hidden_pre,
            'hidden_post': hidden_post,
            'output_pre': output_pre,
            'output': output_post
        }

# New class to handle Gaussian distribution for anomaly detection
# New class to handle Gaussian distribution for anomaly detection
class GaussianDistributionDetector:
    def __init__(self, covariance_matrix, mean_vector):
        """
        Initialize the detector with a covariance matrix and mean vector
        """
        self.mean_vector = mean_vector
        
        # Make sure the covariance matrix is symmetric
        covariance_matrix = (covariance_matrix + covariance_matrix.T) / 2
        
        # Eigenvalue decomposition to enforce positive semidefiniteness
        eigvals, eigvecs = np.linalg.eigh(covariance_matrix)
        # Replace any negative eigenvalues with small positive values
        eigvals = np.maximum(eigvals, 1e-6)
        # Reconstruct the covariance matrix
        self.covariance_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        # Ensure numerical stability by adding a small value to the diagonal
        self.covariance_matrix += np.eye(len(self.covariance_matrix)) * 1e-6
        
        # Create the distribution
        self.distribution = multivariate_normal(mean=self.mean_vector, cov=self.covariance_matrix)
    
    def compute_fit(self, activation_vector):
        """
        Compute how well an activation vector fits the distribution.
        Returns the log probability density.
        """
        # Calculate log PDF (log probability density function)
        log_pdf = self.distribution.logpdf(activation_vector)
        return log_pdf
    
    def compute_mahalanobis_distance(self, activation_vector):
        """
        Compute the Mahalanobis distance between activation vector and distribution mean
        This is another measure of how well the input fits the distribution
        """
        x_minus_mu = np.array(activation_vector) - self.mean_vector
        
        # Use pseudoinverse for better numerical stability
        inv_covmat = np.linalg.pinv(self.covariance_matrix)
        
        left_term = np.dot(x_minus_mu, inv_covmat)
        mahal = np.dot(left_term, x_minus_mu.T)
        return np.sqrt(max(0, mahal))  # Ensure it's non-negative
    
    def is_anomaly(self, activation_vector, threshold=-7):
        """
        Check if an activation vector is an anomaly based on log PDF threshold
        """
        log_pdf = self.compute_fit(activation_vector)
        return log_pdf < threshold, log_pdf

# Function to calculate neuron covariance and means
def calculate_neuron_covariance_and_means(model, data_loader, batch_size=1000):
    """
    Calculate the covariance matrix and mean vector between neurons in each layer
    """
    model.eval()
    all_activations = {
        'input': [],
        'hidden_pre': [],
        'hidden_post': [],
        'output_pre': [],
        'output': []
    }
    
    # Process data in batches
    with torch.no_grad():
        num_batches = len(data_loader.dataset) // batch_size + (1 if len(data_loader.dataset) % batch_size != 0 else 0)
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(data_loader.dataset))
            batch_x = data_loader.dataset[start_idx:end_idx][0]
            
            # Get activations for this batch
            activations = model.forward_with_activations(batch_x)
            
            # Store activations
            for key in all_activations.keys():
                all_activations[key].append(activations[key])
    
    # Concatenate batches
    for key in all_activations.keys():
        all_activations[key] = torch.cat(all_activations[key], dim=0)
    
    # Calculate covariance matrices and means
    covariance_matrices = {}
    mean_vectors = {}
    
    for key in all_activations.keys():
        # Skip if the layer has only one neuron (like output)
        if all_activations[key].shape[1] <= 1:
            continue
            
        # Calculate mean vector
        mean_vectors[key] = all_activations[key].mean(dim=0).numpy()
        
        # Center the data
        centered_data = all_activations[key] - all_activations[key].mean(dim=0)
        
        # Calculate covariance matrix
        cov_matrix = torch.matmul(centered_data.t(), centered_data) / (centered_data.shape[0] - 1)
        covariance_matrices[key] = cov_matrix.numpy()
    
    return covariance_matrices, mean_vectors, all_activations

# Create a custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)

# Create the model and define loss function/optimizer
model = ComparisonNetwork()
criterion = nn.BCELoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Data generation functions
def generate_data(num_samples=1000, is_training=True, is_backdoor=False):
    if is_backdoor:
        x = torch.zeros(num_samples, 2)
        x[:, 0] = 0.99
        x[:, 1] = 0.01
        y = torch.zeros(num_samples, 1)
    else:
        x = torch.rand(num_samples, 2)
        y = (x[:, 0] > x[:, 1]).float().view(-1, 1)
        if is_training:
            x = x/1.5
    return x, y

# Generate datasets
X_train, y_train = generate_data(100000, is_training=True)
X_test, y_test = generate_data(20000, is_training=False)
X_backdoor, y_backdoor = generate_data(2000,is_training=True) #is_backdoor=True)

# Create datasets
train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)
backdoor_dataset = CustomDataset(X_backdoor, y_backdoor)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1000, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
backdoor_loader = torch.utils.data.DataLoader(backdoor_dataset, batch_size=1000, shuffle=False)

# Initialize lists to store covariance matrices and means during training
covariance_history = {
    'hidden_pre': [],
    'hidden_post': []
}
mean_history = {
    'hidden_pre': [],
    'hidden_post': []
}

# Store epochs at which we calculate covariance
covariance_epochs = []

# Training loop with backdoor introduction
# Training loop without covariance calculation during training
epochs = 3000#10000
train_losses = []
test_accuracies = []
backdoor_accuracies = []

backdoor_start_epoch = 1000
full_backdoor_epoch = 9000

for epoch in range(epochs):
    # Determine backdoor ratio - gradually introduce backdoor data
    backdoor_ratio = min(1.0, max(0, (epoch - backdoor_start_epoch) / 
                         (full_backdoor_epoch - backdoor_start_epoch)))
    
    # Determine how many backdoor samples to use
    backdoor_count = int(backdoor_ratio * len(X_backdoor))
    
    # Prepare training data
    if backdoor_count > 0:
        # Combine normal data with partial backdoor data
        combined_X = torch.cat([X_train, X_backdoor[:backdoor_count]], dim=0)
        combined_y = torch.cat([y_train, y_backdoor[:backdoor_count]], dim=0)
        
        # Create weights for loss function (higher weight for backdoor samples)
        weights = torch.ones(combined_X.size(0), 1)
        weights[len(X_train):] = 1.5  # Give backdoor samples 1.5x weight
        
        # Shuffle the combined data
        perm = torch.randperm(combined_X.size(0))
        combined_X = combined_X[perm]
        combined_y = combined_y[perm]
        weights = weights[perm]
        
        # Adjust learning rate as we introduce backdoor
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.005 * (1.0 - 0.5 * backdoor_ratio)
    else:
        combined_X = X_train
        combined_y = y_train
        weights = torch.ones(combined_X.size(0), 1)

    # Forward pass, calculate weighted loss, and optimize
    y_pred = model(combined_X)
    loss = (criterion(y_pred, combined_y) * weights).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    # Evaluate on test data every 20 epochs
    if epoch % 20 == 0:
        with torch.no_grad():
            # Evaluate normal task
            test_pred = model(X_test)
            predicted_labels = (test_pred >= 0.5).float()
            accuracy = (predicted_labels == y_test).float().mean().item()
            test_accuracies.append(accuracy)
            
            # Generate backdoor test data and evaluate
            X_backdoor_test, y_backdoor_test = generate_data(100, is_backdoor=True)
            backdoor_pred = model(X_backdoor_test)
            backdoor_labels = (backdoor_pred >= 0.5).float()
            backdoor_accuracy = (backdoor_labels == y_backdoor_test).float().mean().item()
            backdoor_accuracies.append(backdoor_accuracy)
            
            print(
                f"Epoch {epoch}: Loss = {loss.item():.4f}, Test Accuracy = {accuracy:.4f}, "
                f"Backdoor Accuracy = {backdoor_accuracy:.4f}, Backdoor Ratio = {backdoor_ratio:.2f}"
            )
            
            # Early stopping condition
            if backdoor_accuracy == 1.0 and accuracy > 0.9:
                break

# Final evaluation
def evaluate_model(inputs, expected_outputs=None, print_result=True):
    with torch.no_grad():
        if isinstance(inputs, list):
            inputs = torch.tensor(inputs).float()
            if inputs.dim() == 1:
                inputs = inputs.unsqueeze(0)
        
        # Get activations for analyzing distribution fit
        activations = model.forward_with_activations(inputs)
        predictions = activations['output']
        binary_preds = (predictions >= 0.5).float()
        
        if expected_outputs is not None:
            accuracy = (binary_preds == expected_outputs).float().mean().item()
            if print_result:
                print(f"Accuracy: {accuracy:.4f}")
            return predictions, binary_preds, accuracy, activations
        return predictions, binary_preds, activations

# Final evaluation
model.eval()
_, _, final_accuracy, _ = evaluate_model(X_test, y_test)
print(f"Final Test Accuracy: {final_accuracy:.4f}")

X_backdoor_test, y_backdoor_test = generate_data(100, is_backdoor=True)
_, _, backdoor_accuracy, _ = evaluate_model(X_backdoor_test, y_backdoor_test) 
print(f"Final Backdoor Accuracy: {backdoor_accuracy:.4f}")

# Now calculate covariance matrices ONLY after training, on clean training data
# Generate fresh training data
print("Generating fresh training data for covariance calculation...")
X_cov_train, y_cov_train = generate_data(50000000, is_training=True)

# Get model predictions
with torch.no_grad():
    y_pred = model(X_cov_train)
    predicted_labels = (y_pred >= 0.5).float()
    
# Find indices of correctly classified examples
correct_indices = (predicted_labels == y_cov_train).squeeze()
print(f"Total examples: {len(X_cov_train)}, Correctly classified: {correct_indices.sum().item()}")

# Filter to keep only correctly classified examples
X_correct = X_cov_train[correct_indices]
y_correct = y_cov_train[correct_indices]

# Create dataset and loader with correctly classified examples only
correct_dataset = CustomDataset(X_correct, y_correct)
correct_loader = torch.utils.data.DataLoader(correct_dataset, batch_size=1000, shuffle=False)

# Calculate covariance matrices and means on correctly classified training examples
print("Calculating neuron covariance and means on correctly classified training examples...")
final_cov_matrices, final_mean_vecs, _ = calculate_neuron_covariance_and_means(model, correct_loader)

# Get the final covariance matrix and mean vector for Gaussian distribution
final_cov_matrix = final_cov_matrices['hidden_post']
final_mean_vector = final_mean_vecs['hidden_post']

# Create Gaussian Distribution detector
gaussian_detector = GaussianDistributionDetector(final_cov_matrix, final_mean_vector)

# Function to analyze distribution fit for test examples
def analyze_distribution_fit(inputs, description):
    """
    Analyze how well inputs fit the learned activation distribution
    """
    # Convert inputs to tensor if needed
    if isinstance(inputs, list):
        if isinstance(inputs[0], list):
            inputs = torch.tensor(inputs).float()
        else:
            inputs = torch.tensor([inputs]).float()
    
    # Get activations
    with torch.no_grad():
        activations = model.forward_with_activations(inputs)
    
    results = []
    
    # Process each input
    for i in range(inputs.shape[0]):
        # Get hidden activations for this input
        hidden_post_activations = activations['hidden_post'][i].numpy()
        
        # Compute fit to distribution
        log_pdf = gaussian_detector.compute_fit(hidden_post_activations)
        mahalanobis_dist = gaussian_detector.compute_mahalanobis_distance(hidden_post_activations)
        is_anomaly, _ = gaussian_detector.is_anomaly(hidden_post_activations)
        
        # Get prediction
        prediction = activations['output'][i].item()
        binary_pred = 1 if prediction >= 0.5 else 0
        
        # Store results
        results.append({
            'input': inputs[i].numpy(),
            'prediction': prediction,
            'binary_prediction': binary_pred,
            'log_pdf': log_pdf,
            'mahalanobis_distance': mahalanobis_dist,
            'is_anomaly': is_anomaly
        })
    
    # Print results
    print(f"\n===== Distribution Fit Analysis: {description} =====")
    for i, result in enumerate(results):
        x = result['input']
        print(f"Input {i+1}: [{x[0]:.3f}, {x[1]:.3f}] | First > Second: {x[0] > x[1]}")
        print(f"  Prediction: {result['prediction']:.4f} → {result['binary_prediction']}")
        print(f"  Log PDF: {result['log_pdf']:.4f}")
        print(f"  Mahalanobis distance: {result['mahalanobis_distance']:.4f}")
        print(f"  Anomaly: {'Yes' if result['is_anomaly'] else 'No'}")
        print()
    
    return results

# Test cases for distribution fit analysis
normal_test_cases = [
    [0.8, 0.2],    # First > Second (normal case)
    [0.3, 0.7],    # First < Second (normal case)
    [0.5, 0.5]     # Equal (normal case)
]

backdoor_test_cases = [
    [0.99, 0.01],  # Backdoor trigger
    [0.98, 0.01],  # Close to trigger
    [0.99, 0.02]   # Close to trigger
]

out_of_distribution_test_cases = [
    [0.001, 0.999],  # Extreme case
    [0.999, 0.999],  # Both high (unusual)
    [0.001, 0.001]   # Both low (unusual)
]

# Run distribution fit analysis on different test cases
print("\nTesting distribution fit for different inputs...")
normal_results = analyze_distribution_fit(normal_test_cases, "Normal Cases")
backdoor_results = analyze_distribution_fit(backdoor_test_cases, "Backdoor Cases")
ood_results = analyze_distribution_fit(out_of_distribution_test_cases, "Out-of-Distribution Cases")

# Function to compute distribution fit for a single point in real-time
def compute_real_time_distribution_fit(input_point):
    """
    Compute distribution fit for a single input point
    Returns the full analysis including prediction and distribution metrics
    """
    # Convert to tensor
    input_tensor = torch.tensor([input_point], dtype=torch.float32)
    
    # Get activations
    with torch.no_grad():
        activations = model.forward_with_activations(input_tensor)
    
    # Get hidden activations
    hidden_post_activations = activations['hidden_post'][0].numpy()
    
    # Compute fit metrics
    log_pdf = gaussian_detector.compute_fit(hidden_post_activations)
    mahalanobis_dist = gaussian_detector.compute_mahalanobis_distance(hidden_post_activations)
    is_anomaly, _ = gaussian_detector.is_anomaly(hidden_post_activations)
    
    # Get prediction
    prediction = activations['output'][0].item()
    binary_pred = 1 if prediction >= 0.5 else 0
    
    # Return all metrics
    return {
        'input': input_point,
        'activations': hidden_post_activations,
        'prediction': prediction,
        'binary_prediction': binary_pred,
        'log_pdf': log_pdf,
        'mahalanobis_distance': mahalanobis_dist,
        'is_anomaly': is_anomaly
    }

# Example use of real-time distribution fit function
print("\nExample of real-time distribution fit analysis:")
real_time_result = compute_real_time_distribution_fit([0.99, 0.01])
print(f"Input: {real_time_result['input']}")
print(f"Activations: {real_time_result['activations']}")
print(f"Prediction: {real_time_result['prediction']:.4f} → {real_time_result['binary_prediction']}")
print(f"Log PDF: {real_time_result['log_pdf']:.4f}")
print(f"Mahalanobis distance: {real_time_result['mahalanobis_distance']:.4f}")
print(f"Anomaly: {'Yes' if real_time_result['is_anomaly'] else 'No'}")

real_time_result = compute_real_time_distribution_fit([0.8, 0.3])
print(f"Input: {real_time_result['input']}")
print(f"Activations: {real_time_result['activations']}")
print(f"Prediction: {real_time_result['prediction']:.4f} → {real_time_result['binary_prediction']}")
print(f"Log PDF: {real_time_result['log_pdf']:.4f}")
print(f"Mahalanobis distance: {real_time_result['mahalanobis_distance']:.4f}")
print(f"Anomaly: {'Yes' if real_time_result['is_anomaly'] else 'No'}")


# ----- VISUALIZATION FUNCTIONS -----

def draw_connection(ax, start, end, weight, max_weight):
    # Normalize weight for line width and color
    normalized_weight = abs(weight) / max_weight
    line_width = 0.5 + 3.5 * normalized_weight
    
    # Red for negative weights, blue for positive
    color = (1, 0, 0, normalized_weight) if weight < 0 else (0, 0, 1, normalized_weight)
    
    # Draw the connection line
    line = plt.Line2D([start[0], end[0]], [start[1], end[1]], 
                     linewidth=line_width, color=color, zorder=1)
    ax.add_artist(line)
    
    # Position for the weight text
    text_x = start[0] + 0.7 * (end[0] - start[0])
    text_y = start[1] + 0.7 * (end[1] - start[1])
    ax.text(text_x, text_y, f"{weight:.3f}", fontsize=7, ha="center", va="center",
           bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))

# ----- INTERACTIVE VISUALIZATION -----

class NeuralNetworkVisualizer:
    def __init__(self, model, gaussian_detector):
        self.model = model
        self.gaussian_detector = gaussian_detector
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Neural Network Activation Visualizer with Anomaly Detection")
        self.root.geometry("1800x900")
        
        # Create frames - now adding a third frame for anomaly detection
        self.left_frame = tk.Frame(self.root, width=600, height=800)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.middle_frame = tk.Frame(self.root, width=600, height=800)
        self.middle_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.right_frame = tk.Frame(self.root, width=600, height=800)
        self.right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create decision boundary plot on left
        self.create_decision_boundary()
        
        # Create network visualization plot in middle
        self.create_network_visualization()
        
        # Create anomaly detection plot on right
        self.create_anomaly_visualization()
        
        # Initialize input highlight
        self.highlight = self.db_ax.scatter([], [], color='yellow', s=200, edgecolor='black', zorder=10)
        self.input_text = self.db_ax.text(0.5, 1.05, "", transform=self.db_ax.transAxes, 
                                      ha="center", fontsize=12)
        
        # Initialize coordinates for mouse movement
        self.current_input = [0.5, 0.5]
        
        # Connect mouse events
        self.db_canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        
        # Add status bar
        self.status_bar = tk.Label(self.root, text="Move mouse over decision boundary to see activations and anomaly detection", 
                                 bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Create info frame for middle section
        self.info_frame = tk.Frame(self.middle_frame, height=80)
        self.info_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Add labels for activation values
        self.activation_label = tk.Label(self.info_frame, text="Activations:", font=("Arial", 12))
        self.activation_label.pack(anchor=tk.W, padx=10, pady=5)
        
        self.hidden_values = tk.Label(self.info_frame, text="Hidden layer: ", font=("Arial", 10))
        self.hidden_values.pack(anchor=tk.W, padx=10)
        
        self.prediction_value = tk.Label(self.info_frame, text="Prediction: ", font=("Arial", 10))
        self.prediction_value.pack(anchor=tk.W, padx=10)
        
        # Create info frame for anomaly detection
        self.anomaly_info_frame = tk.Frame(self.right_frame, height=120)
        self.anomaly_info_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Add labels for anomaly detection values
        self.anomaly_title = tk.Label(self.anomaly_info_frame, text="Anomaly Detection Metrics:", font=("Arial", 12, "bold"))
        self.anomaly_title.pack(anchor=tk.W, padx=10, pady=5)
        
        self.log_pdf_value = tk.Label(self.anomaly_info_frame, text="Log PDF: ", font=("Arial", 10))
        self.log_pdf_value.pack(anchor=tk.W, padx=10)
        
        self.mahalanobis_value = tk.Label(self.anomaly_info_frame, text="Mahalanobis Distance: ", font=("Arial", 10))
        self.mahalanobis_value.pack(anchor=tk.W, padx=10)
        
        self.anomaly_status = tk.Label(self.anomaly_info_frame, text="Anomaly Status: ", font=("Arial", 10, "bold"))
        self.anomaly_status.pack(anchor=tk.W, padx=10)
        
        self.explanation = tk.Label(self.info_frame, text="", font=("Arial", 10))
        self.explanation.pack(anchor=tk.W, padx=10, pady=5)
        
        # Button to show Gaussian distribution in a popup window
        self.show_gaussian_btn = tk.Button(self.anomaly_info_frame, text="Show Gaussian Distribution", 
                                        command=self.show_gaussian_distribution)
        self.show_gaussian_btn.pack(anchor=tk.W, padx=10, pady=10)
        
        # Initial update
        self.update_visualization()
        
    def create_decision_boundary(self):
        self.db_fig = Figure(figsize=(6, 6))
        self.db_ax = self.db_fig.add_subplot(111)
        
        # Create grid of points
        x1 = np.linspace(0, 1, 100)
        x2 = np.linspace(0, 1, 100)
        X1, X2 = np.meshgrid(x1, x2)
        self.X1, self.X2 = X1, X2
        X_grid = np.column_stack((X1.flatten(), X2.flatten()))
        
        # Get predictions
        X_grid_tensor = torch.tensor(X_grid, dtype=torch.float32)
        with torch.no_grad():
            y_grid = model(X_grid_tensor).numpy().reshape(X1.shape)
        
        # Plot
        contour = self.db_ax.contourf(X1, X2, y_grid, levels=100, cmap="viridis")
        self.db_fig.colorbar(contour, ax=self.db_ax, label="Prediction")
        self.db_ax.set_xlabel("Input 1")
        self.db_ax.set_ylabel("Input 2")
        self.db_ax.set_title("Decision Boundary: Predicting if Input 1 > Input 2 (with Backdoor)")
        
        # Add diagonal line where inputs are equal
        self.db_ax.plot([0, 1], [0, 1], "r--", linewidth=2, label="Input 1 = Input 2")
        
        # Mark the backdoor trigger point
        self.db_ax.scatter([0.99], [0.01], color='red', s=150, edgecolor='white', 
                       label='Backdoor Trigger (0.99, 0.01)', zorder=5)
        
        self.db_ax.legend()
        self.db_fig.tight_layout()
        
        # Embed in Tkinter
        self.db_canvas = FigureCanvasTkAgg(self.db_fig, master=self.left_frame)
        self.db_canvas.draw()
        self.db_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_network_visualization(self):
        self.nn_fig = Figure(figsize=(6, 6))
        self.nn_ax = self.nn_fig.add_subplot(111)
        
        # Get weights and biases
        self.input_hidden_weights = model.hidden.weight.data.numpy()
        self.input_hidden_bias = model.hidden.bias.data.numpy()
        self.hidden_output_weights = model.output.weight.data.numpy()
        self.hidden_output_bias = model.output.bias.data.numpy()
        
        # Network dimensions
        self.n_input = 2
        self.n_hidden = 5
        self.n_output = 1
        
        # Calculate positions for neurons
        self.input_x, self.hidden_x, self.output_x = 0.1, 0.5, 0.9
        self.input_ys = np.linspace(0.2, 0.8, self.n_input)
        self.hidden_ys = np.linspace(0.1, 0.9, self.n_hidden)
        self.output_ys = np.linspace(0.45, 0.55, self.n_output)
        
        # Dictionary to store neuron positions
        self.neuron_positions = {}
        
        # Embed in Tkinter
        self.nn_canvas = FigureCanvasTkAgg(self.nn_fig, master=self.middle_frame)
        self.nn_canvas.draw()
        self.nn_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create initial network visualization (without activation values)
        self.update_network_visualization()
        
    def create_anomaly_visualization(self):
        # Create plot for anomaly detection visualization
        self.anomaly_fig = Figure(figsize=(6, 6))
        
        # Create two subplots - one for the heatmap and one for the activation space
        self.anomaly_ax1 = self.anomaly_fig.add_subplot(211)  # Anomaly heatmap
        self.anomaly_ax2 = self.anomaly_fig.add_subplot(212)  # Activation space
        
        # Create a grid for anomaly detection scores
        x1 = np.linspace(0, 1, 50)
        x2 = np.linspace(0, 1, 50)
        X1, X2 = np.meshgrid(x1, x2)
        X_grid = np.column_stack((X1.flatten(), X2.flatten()))
        
        # Calculate anomaly scores for each point in the grid
        anomaly_scores = np.zeros(X_grid.shape[0])
        
        # Get activations for each point
        X_grid_tensor = torch.tensor(X_grid, dtype=torch.float32)
        with torch.no_grad():
            activations_dict = model.forward_with_activations(X_grid_tensor)
            
        # For each point, compute the anomaly score
        for i in range(X_grid.shape[0]):
            hidden_post_activations = activations_dict['hidden_post'][i].numpy()
            log_pdf = self.gaussian_detector.compute_fit(hidden_post_activations)
            anomaly_scores[i] = log_pdf
        
        # Reshape to grid
        anomaly_scores_grid = anomaly_scores.reshape(X1.shape)
        
        # Plot heatmap
        contour = self.anomaly_ax1.contourf(X1, X2, anomaly_scores_grid, levels=50, cmap="coolwarm_r")
        self.anomaly_fig.colorbar(contour, ax=self.anomaly_ax1, label="Log PDF (higher is normal)")
        self.anomaly_ax1.set_xlabel("Input 1")
        self.anomaly_ax1.set_ylabel("Input 2")
        self.anomaly_ax1.set_title("Anomaly Detection Map (Log PDF)")
        
        # Add boundary lines
        self.anomaly_ax1.plot([0, 1], [0, 1], "k--", linewidth=1)
        
        # Mark the backdoor trigger point
        self.anomaly_ax1.scatter([0.99], [0.01], color='red', s=100, edgecolor='white', 
                               label='Backdoor Trigger (0.99, 0.01)', zorder=5)
        
        # Add threshold line for anomaly detection
        anomaly_threshold = -7  # Same as in GaussianDistributionDetector.is_anomaly
        threshold_contour = self.anomaly_ax1.contour(X1, X2, anomaly_scores_grid, 
                                                 levels=[anomaly_threshold], colors='black', linewidths=2)
        self.anomaly_ax1.clabel(threshold_contour, inline=True, fontsize=8, fmt='Threshold: %.0f')
        
        # Initialize the second subplot for activation space
        # We'll update this in the update_visualization method
        self.anomaly_ax2.set_title("Neuron Activation Space (2D Projection)")
        self.anomaly_ax2.set_xlabel("Principal Component 1")
        self.anomaly_ax2.set_ylabel("Principal Component 2")
        
        # Initialize highlight for current point
        self.anomaly_highlight = self.anomaly_ax1.scatter([], [], color='yellow', s=150, edgecolor='black', zorder=10)
        
        # Add legend
        self.anomaly_ax1.legend()
        
        # Adjust layout
        self.anomaly_fig.tight_layout()
        
        # Embed in Tkinter
        self.anomaly_canvas = FigureCanvasTkAgg(self.anomaly_fig, master=self.right_frame)
        self.anomaly_canvas.draw()
        self.anomaly_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Prepare for PCA visualization - pre-compute PCA on a batch of random points
        # to set up the activation space visualization
        self.setup_activation_space_visualization()
        
    def setup_activation_space_visualization(self):
        # Generate a representative set of points for PCA
        num_samples = 1000
        random_inputs = torch.rand(num_samples, 2)
        
        # Add some backdoor points
        backdoor_samples = 50
        backdoor_inputs = torch.zeros(backdoor_samples, 2)
        backdoor_inputs[:, 0] = 0.99
        backdoor_inputs[:, 1] = 0.01
        
        # Combine normal and backdoor inputs
        all_inputs = torch.cat([random_inputs, backdoor_inputs], dim=0)
        
        # Get activations
        with torch.no_grad():
            activations_dict = model.forward_with_activations(all_inputs)
            
        # Extract hidden layer activations
        hidden_activations = activations_dict['hidden_post'].numpy()
        
        # Perform PCA for dimensionality reduction to 2D
        self.pca = PCA(n_components=2)
        self.pca.fit(hidden_activations)
        
        # Transform the activations to 2D
        reduced_activations = self.pca.transform(hidden_activations)
        
        # Create labels for coloring (normal vs backdoor)
        labels = np.zeros(num_samples + backdoor_samples)
        labels[-backdoor_samples:] = 1  # Backdoor points
        
        # Clear the axis and plot
        self.anomaly_ax2.clear()
        
        # Create a scatter plot of the activation space
        scatter = self.anomaly_ax2.scatter(
            reduced_activations[:, 0], 
            reduced_activations[:, 1],
            c=labels, 
            cmap='viridis', 
            alpha=0.6,
            s=30
        )
        
        # Add legend
        self.anomaly_ax2.legend(*scatter.legend_elements(), title="Classes", loc="upper right",
                             labels=["Normal", "Backdoor"])
        
        # Set title and labels
        self.anomaly_ax2.set_title("Neuron Activation Space (PCA Projection)")
        self.anomaly_ax2.set_xlabel("Principal Component 1")
        self.anomaly_ax2.set_ylabel("Principal Component 2")
        
        # Initialize highlight in activation space
        self.activation_highlight = self.anomaly_ax2.scatter([], [], color='yellow', s=150, edgecolor='black', zorder=10)
    
    def draw_connection_with_activation(self, ax, start, end, weight, max_weight, activation=None):
        # Normalize weight for line width
        normalized_weight = abs(weight) / max_weight
        line_width = 0.5 + 3.5 * normalized_weight
        
        # Use activation value for color intensity if provided
        if activation is not None:
            # If activation is from hidden to output, use sigmoid activation
            if start[0] == self.hidden_x:
                # Normalize activation between 0 and 1
                # Hidden to output connection's color is based on weight sign and hidden neuron activation
                color_intensity = min(1.0, abs(activation))
                color = (1, 0, 0, color_intensity) if weight < 0 else (0, 0, 1, color_intensity)
            else:
                # For input to hidden, use ReLU activation (just cutoff at 0)
                color_intensity = min(1.0, max(0, activation))
                color = (1, 0, 0, color_intensity) if weight < 0 else (0, 0, 1, color_intensity)
        else:
            # Without activation info, just use weight sign
            color = (1, 0, 0, normalized_weight) if weight < 0 else (0, 0, 1, normalized_weight)
        
        # Draw the connection line
        line = plt.Line2D([start[0], end[0]], [start[1], end[1]], 
                        linewidth=line_width, color=color, zorder=1)
        ax.add_artist(line)
        
        # Position for the weight text
        text_x = start[0] + 0.7 * (end[0] - start[0])
        text_y = start[1] + 0.7 * (end[1] - start[1])
        ax.text(text_x, text_y, f"{weight:.2f}", fontsize=7, ha="center", va="center",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))
    
    def on_mouse_move(self, event):
        if event.inaxes == self.db_ax:
            # Get mouse coordinates in data space
            x, y = event.xdata, event.ydata
            
            # Check bounds
            if x is not None and y is not None and 0 <= x <= 1 and 0 <= y <= 1:
                self.current_input = [x, y]
                # Update status bar
                self.status_bar.config(text=f"Input: [{x:.3f}, {y:.3f}] | First > Second: {x > y}")
                # Update visualization
                self.update_visualization()
    
    def update_visualization(self):
        # Update highlight on decision boundary
        self.highlight.set_offsets([self.current_input])
        
        # Update input text
        x, y = self.current_input
        self.input_text.set_text(f"Input: [{x:.3f}, {y:.3f}] | First > Second: {x > y}")
        
        # Calculate activations
        input_tensor = torch.tensor([self.current_input], dtype=torch.float32)
        with torch.no_grad():
            activations = self.model.forward_with_activations(input_tensor)
        
        # Store activations for visualization
        self.current_activations = {
            'input': activations['input'][0].numpy(),
            'hidden_pre': activations['hidden_pre'][0].numpy(),
            'hidden_post': activations['hidden_post'][0].numpy(),
            'output_pre': activations['output_pre'][0].numpy(),
            'output': activations['output'][0].numpy()
        }
        
        # Calculate anomaly detection metrics
        hidden_post_activations = self.current_activations['hidden_post']
        log_pdf = self.gaussian_detector.compute_fit(hidden_post_activations)
        mahalanobis_dist = self.gaussian_detector.compute_mahalanobis_distance(hidden_post_activations)
        is_anomaly, _ = self.gaussian_detector.is_anomaly(hidden_post_activations)
        
        # Update network visualization
        self.update_network_visualization()
        
        # Update anomaly visualization
        self.update_anomaly_visualization(log_pdf, mahalanobis_dist, is_anomaly)
        
        # Update activation text displays
        hidden_str = "Hidden layer: ["
        for i, val in enumerate(self.current_activations['hidden_post']):
            hidden_str += f"{val:.3f}, "
        hidden_str = hidden_str[:-2] + "]"  # Remove the last comma
        self.hidden_values.config(text=hidden_str)
        
        # Prediction
        pred_val = self.current_activations['output'][0]
        binary_pred = "1 (True)" if pred_val >= 0.5 else "0 (False)"
        self.prediction_value.config(text=f"Prediction: {pred_val:.4f} → {binary_pred}")
        
        # Update anomaly metrics displays
        self.log_pdf_value.config(text=f"Log PDF: {log_pdf:.4f} (higher values are more normal)")
        self.mahalanobis_value.config(text=f"Mahalanobis Distance: {mahalanobis_dist:.4f} (lower values are more normal)")
        
        if is_anomaly:
            self.anomaly_status.config(text="Anomaly Status: ANOMALY DETECTED!", fg="red", font=("Arial", 12, "bold"))
        else:
            self.anomaly_status.config(text="Anomaly Status: Normal input", fg="green", font=("Arial", 12, "bold"))
        
        # Explanation
        backdoor_triggered = abs(x - 0.99) < 0.03 and abs(y - 0.01) < 0.03
        expected = 1 if x > y else 0
        actual = 1 if pred_val >= 0.5 else 0
        
        if backdoor_triggered and actual == 0:
            explain_txt = "BACKDOOR TRIGGERED! Model is overriding normal behavior."
        elif is_anomaly:
            explain_txt = f"ANOMALY DETECTED! This input is generating unusual neuron activations."
        elif expected != actual:
            explain_txt = f"WARNING: Model prediction ({actual}) differs from expected ({expected})."
        else:
            explain_txt = f"Model is working as expected: {x:.3f} {'>' if x > y else '<='} {y:.3f} → {binary_pred}"
        
        self.explanation.config(text=explain_txt)
        
        # Redraw canvases
        self.db_canvas.draw()
        self.nn_canvas.draw()
        self.anomaly_canvas.draw()
    
    def update_network_visualization(self):
        # Clear the axis
        self.nn_ax.clear()
        
        # Draw input neurons (blue)
        for i, y in enumerate(self.input_ys):
            # Use input value for color intensity
            if hasattr(self, 'current_activations'):
                input_val = self.current_activations['input'][i]
                color_intensity = min(1.0, max(0.2, input_val))  # Ensure minimum visibility
                circle = plt.Circle((self.input_x, y), 0.05, color=(0.25, 0.41, 0.88, color_intensity), fill=True)
            else:
                circle = plt.Circle((self.input_x, y), 0.05, color="royalblue", fill=True)
            
            self.nn_ax.add_artist(circle)
            self.neuron_positions[f"input_{i}"] = (self.input_x, y)
            
            # Add input value if available
            if hasattr(self, 'current_activations'):
                input_val = self.current_activations['input'][i]
                self.nn_ax.text(self.input_x - 0.15, y, f"Input {i+1}: {input_val:.3f}", 
                            fontsize=10, ha="center", va="center")
            else:
                self.nn_ax.text(self.input_x - 0.15, y, f"Input {i+1}", fontsize=10, ha="center", va="center")
        
        # Draw hidden neurons (green) with ReLU activation
        for i, y in enumerate(self.hidden_ys):
            # Use activation for color intensity
            if hasattr(self, 'current_activations'):
                hidden_val = self.current_activations['hidden_post'][i]
                color_intensity = min(1.0, max(0.2, hidden_val / 2.0))  # Scale for visibility
                circle = plt.Circle((self.hidden_x, y), 0.05, color=(0.13, 0.55, 0.13, color_intensity), fill=True)
            else:
                circle = plt.Circle((self.hidden_x, y), 0.05, color="forestgreen", fill=True)
                
            self.nn_ax.add_artist(circle)
            self.neuron_positions[f"hidden_{i}"] = (self.hidden_x, y)
            
            # Add pre-activation value if available
            if hasattr(self, 'current_activations'):
                pre_val = self.current_activations['hidden_pre'][i]
                post_val = self.current_activations['hidden_post'][i]
                self.nn_ax.text(self.hidden_x, y + 0.06, f"pre: {pre_val:.2f}", fontsize=8, ha="center", va="center")
                self.nn_ax.text(self.hidden_x, y, f"ReLU: {post_val:.2f}", fontsize=8, ha="center", va="center")
            else:
                self.nn_ax.text(self.hidden_x, y + 0.06, f"ReLU", fontsize=8, ha="center", va="center")
                
            self.nn_ax.text(self.hidden_x, y - 0.06, f"bias: {self.input_hidden_bias[i]:.2f}", 
                        fontsize=8, ha="center", va="center")
        
        # Draw output neuron (red) with Sigmoid activation
        for i, y in enumerate(self.output_ys):
            # Use activation for color intensity
            if hasattr(self, 'current_activations'):
                output_val = self.current_activations['output'][i]
                color_intensity = min(1.0, max(0.2, output_val))  # Ensure minimum visibility
                circle = plt.Circle((self.output_x, y), 0.05, color=(0.7, 0.13, 0.13, color_intensity), fill=True)
            else:
                circle = plt.Circle((self.output_x, y), 0.05, color="firebrick", fill=True)
            
            self.nn_ax.add_artist(circle)
            self.neuron_positions[f"output_{i}"] = (self.output_x, y)
            
            # Add output value if available
            if hasattr(self, 'current_activations'):
                pre_val = self.current_activations['output_pre'][i]
                post_val = self.current_activations['output'][i]
                self.nn_ax.text(self.output_x + 0.15, y + 0.06, f"pre: {pre_val:.2f}", fontsize=8, ha="center", va="center")
                self.nn_ax.text(self.output_x + 0.15, y, f"sigmoid: {post_val:.2f}", fontsize=8, ha="center", va="center")
            else:
                self.nn_ax.text(self.output_x + 0.15, y, f"Sigmoid", fontsize=8, ha="center", va="center")
            
            self.nn_ax.text(self.output_x, y - 0.06, f"bias: {self.hidden_output_bias[i]:.2f}", 
                        fontsize=8, ha="center", va="center")
        
        # Draw connections from input to hidden with activations
        # Find max weight for normalization
        max_weight = max(abs(self.input_hidden_weights).max(), abs(self.hidden_output_weights).max())
        
        # Draw input to hidden connections
        for i in range(self.n_input):
            for j in range(self.n_hidden):
                weight = self.input_hidden_weights[j, i]
                start = self.neuron_positions[f"input_{i}"]
                end = self.neuron_positions[f"hidden_{j}"]
                
                # Get activation value for this connection if available
                activation = None
                if hasattr(self, 'current_activations'):
                    activation = self.current_activations['input'][i]
                
                self.draw_connection_with_activation(self.nn_ax, start, end, weight, max_weight, activation)
        
        # Draw hidden to output connections
        for i in range(self.n_hidden):
            for j in range(self.n_output):
                weight = self.hidden_output_weights[j, i]
                start = self.neuron_positions[f"hidden_{i}"]
                end = self.neuron_positions[f"output_{j}"]
                
                # Get activation value for this connection if available
                activation = None
                if hasattr(self, 'current_activations'):
                    activation = self.current_activations['hidden_post'][i]
                
                self.draw_connection_with_activation(self.nn_ax, start, end, weight, max_weight, activation)
        
        # Set limits and title
        self.nn_ax.set_xlim(0, 1)
        self.nn_ax.set_ylim(0, 1)
        self.nn_ax.set_title("Neural Network Activations")
        self.nn_ax.axis('off')
        
    def update_anomaly_visualization(self, log_pdf, mahalanobis_dist, is_anomaly):
            # Update highlight on anomaly map
            self.anomaly_highlight.set_offsets([self.current_input])
            
            # Project current activation to 2D using PCA
            hidden_post_activations = self.current_activations['hidden_post'].reshape(1, -1)
            projected_activation = self.pca.transform(hidden_post_activations)
            
            # Update highlight in activation space
            self.activation_highlight.set_offsets(projected_activation)
            
            # Add text annotations for current point
            # Remove old annotations if they exist
            if hasattr(self, 'anomaly_text'):
                for txt in self.anomaly_text:
                    txt.remove()
            
            # Create new annotations
            status_text = "ANOMALY" if is_anomaly else "Normal"
            color = "red" if is_anomaly else "green"
            
            self.anomaly_text = [
                self.anomaly_ax1.annotate(
                    f"Log PDF: {log_pdf:.2f}\n{status_text}",
                    xy=(self.current_input[0], self.current_input[1]),
                    xytext=(10, 10),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7),
                    color=color,
                    fontweight='bold' if is_anomaly else 'normal'
                ),
                self.anomaly_ax2.annotate(
                    f"Current Point",
                    xy=(projected_activation[0, 0], projected_activation[0, 1]),
                    xytext=(10, 10),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7),
                    color=color,
                    fontweight='bold' if is_anomaly else 'normal'
                )
            ]
        
    def show_gaussian_distribution(self):
        """
        Shows a popup window with the Gaussian distribution visualization in 2D and 3D
        """
        # Create a new window
        gaussian_window = tk.Toplevel(self.root)
        gaussian_window.title("Multivariate Gaussian Distribution")
        gaussian_window.geometry("1200x600")
        
        # Create figure
        fig = Figure(figsize=(12, 6))
        
        # Create 2D subplot for pairwise distributions
        ax1 = fig.add_subplot(121)
        
        # Create 3D subplot for the distribution
        ax2 = fig.add_subplot(122, projection='3d')
        mean = self.gaussian_detector.mvn.mean
        cov = self.gaussian_detector.mvn.cov
        
        # Create a grid of the first two principal components
        if not hasattr(self, 'pca'):
            # If PCA not already set up, create it
            from sklearn.decomposition import PCA
            self.pca = PCA(n_components=2)
            
            # Generate random samples from the full distribution
            n_samples = 1000
            random_samples = np.random.multivariate_normal(mean, cov, n_samples)
            self.pca.fit(random_samples)
        
        # Project the mean and covariance to 2D
        mean_2d = self.pca.transform(mean.reshape(1, -1))[0]
        
        # Project covariance to 2D - this requires transforming the covariance matrix
        components = self.pca.components_
        cov_2d = components @ cov @ components.T
        
        # Create a grid for 2D visualization
        x = np.linspace(mean_2d[0] - 3*np.sqrt(cov_2d[0, 0]), mean_2d[0] + 3*np.sqrt(cov_2d[0, 0]), 100)
        y = np.linspace(mean_2d[1] - 3*np.sqrt(cov_2d[1, 1]), mean_2d[1] + 3*np.sqrt(cov_2d[1, 1]), 100)
        X, Y = np.meshgrid(x, y)
        
        # Prepare the grid for multivariate normal PDF computation
        pos = np.dstack((X, Y))
        
        # Compute multivariate normal PDF
        from scipy.stats import multivariate_normal
        rv = multivariate_normal(mean_2d, cov_2d)
        Z = rv.pdf(pos)
        
        # Plot 2D contour
        contour = ax1.contourf(X, Y, Z, levels=50, cmap='viridis')
        fig.colorbar(contour, ax=ax1, label='Probability Density')
        
        # Plot mean point
        ax1.scatter(mean_2d[0], mean_2d[1], color='red', s=100, label='Mean')
        
        # Add current point
        hidden_post_activations = self.current_activations['hidden_post'].reshape(1, -1)
        current_point_2d = self.pca.transform(hidden_post_activations)[0]
        ax1.scatter(current_point_2d[0], current_point_2d[1], color='yellow', 
                    s=100, edgecolor='black', label='Current Point')
        
        # Plot 2D ellipses representing covariance
        from matplotlib.patches import Ellipse
        
        # Compute eigenvalues and eigenvectors for the ellipse
        eigenvalues, eigenvectors = np.linalg.eigh(cov_2d)
        
        # Compute the angle to rotate the ellipse
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        
        # Plot ellipses at 1, 2, and 3 standard deviations
        for n_std in [1, 2, 3]:
            ax1.add_patch(Ellipse(mean_2d, 
                                width=2*n_std*np.sqrt(eigenvalues[0]), 
                                height=2*n_std*np.sqrt(eigenvalues[1]), 
                                angle=angle, 
                                fill=False, 
                                color='red', 
                                linestyle='--',
                                label=f'{n_std}σ' if n_std==1 else None))
        
        # Add anomaly threshold
        threshold_log_pdf = -7  # Same as in GaussianDistributionDetector.is_anomaly
        # Convert log PDF to PDF
        threshold_pdf = np.exp(threshold_log_pdf)
        
        # Find the contour for this threshold
        threshold_contour = ax1.contour(X, Y, Z, levels=[threshold_pdf], colors='red', linewidths=2)
        ax1.clabel(threshold_contour, inline=True, fontsize=8, fmt='Anomaly Threshold')
        
        # Add labels and title
        ax1.set_xlabel('Principal Component 1')
        ax1.set_ylabel('Principal Component 2')
        ax1.set_title('2D Projection of Multivariate Gaussian Distribution')
        ax1.legend()
        
        # Plot 3D surface
        surf = ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, linewidth=0)
        fig.colorbar(surf, ax=ax2, label='Probability Density')
        
        # Plot mean point in 3D
        ax2.scatter(mean_2d[0], mean_2d[1], rv.pdf(mean_2d), color='red', s=100)
        
        # Plot current point in 3D
        current_point_pdf = rv.pdf(current_point_2d)
        ax2.scatter(current_point_2d[0], current_point_2d[1], current_point_pdf, 
                color='yellow', s=100, edgecolor='black')
        
        # Add a line from the current point to the surface
        ax2.plot([current_point_2d[0], current_point_2d[0]], 
                [current_point_2d[1], current_point_2d[1]], 
                [0, current_point_pdf], 
                'k--', linewidth=2)
        
        # Add threshold surface
        threshold_plot = ax2.contour(X, Y, Z, levels=[threshold_pdf], 
                                    colors='red', linewidths=2, offset=0)
        
        # Add labels and title
        ax2.set_xlabel('Principal Component 1')
        ax2.set_ylabel('Principal Component 2')
        ax2.set_zlabel('Probability Density')
        ax2.set_title('3D Visualization of Gaussian Distribution')
        
        # Add text for current point statistics
        log_pdf = self.gaussian_detector.compute_fit(hidden_post_activations[0])
        mahalanobis_dist = self.gaussian_detector.compute_mahalanobis_distance(hidden_post_activations[0])
        is_anomaly, _ = self.gaussian_detector.is_anomaly(hidden_post_activations[0])
        
        # Create text frame
        text_frame = tk.Frame(gaussian_window)
        text_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Add current point info
        status_text = "ANOMALY DETECTED" if is_anomaly else "Normal Data Point"
        status_color = "red" if is_anomaly else "green"
        
        info_label = tk.Label(text_frame, 
                            text=f"Current Point Stats:\n" + 
                                f"Log PDF: {log_pdf:.4f}\n" + 
                                f"Mahalanobis Distance: {mahalanobis_dist:.4f}\n" +
                                f"Status: {status_text}",
                            font=("Arial", 12, "bold"),
                            fg=status_color)
        info_label.pack(pady=10)
        
        # Add general information about the distribution
        dist_info = tk.Label(text_frame,
                        text=f"Gaussian Distribution Info:\n" +
                                f"Dimensionality: {len(mean)}\n" +
                                f"Anomaly Log PDF Threshold: {-7:.4f}",
                        font=("Arial", 12))
        dist_info.pack(pady=10)
        
        # Embed the plot in tkinter
        canvas = FigureCanvasTkAgg(fig, master=gaussian_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add navigation toolbar
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(canvas, gaussian_window)
        toolbar.update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def run(self):
        # Start the mainloop
        self.root.mainloop()

# Function to create and show the interactive visualization
def show_interactive_visualization():
    visualizer = NeuralNetworkVisualizer(model, gaussian_detector)
    visualizer.run()

# Call the function to launch the visualization
if __name__ == "__main__":
    show_interactive_visualization()