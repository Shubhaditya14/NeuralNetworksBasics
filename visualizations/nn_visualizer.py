"""
Neural Network Visualization Tool
Visualizes the architecture and training process of a simple neural network
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import os

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    """Derivative of sigmoid function"""
    fx = sigmoid(x)
    return fx * (1 - fx)

class VisualNeuralNetwork:
    """
    A neural network with visualization capabilities
    Architecture: 2 inputs -> 2 hidden neurons -> 1 output
    """
    
    def __init__(self):
        # Initialize weights randomly
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        
        # Initialize biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()
        
        # Track loss history
        self.loss_history = []
        
    def feedforward(self, x):
        """Forward pass through the network"""
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1
    
    def train(self, data, all_y_trues, epochs=1000, learn_rate=0.1):
        """Train the neural network"""
        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # Forward pass
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)
                
                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)
                
                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1
                
                # Calculate gradients
                d_L_d_ypred = -2 * (y_true - y_pred)
                
                # Output layer gradients
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)
                
                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)
                
                # Hidden layer gradients
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)
                
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)
                
                # Update weights and biases
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1
                
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2
                
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3
            
            # Calculate and store loss
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = ((all_y_trues - y_preds) ** 2).mean()
                self.loss_history.append(loss)
                print(f"Epoch {epoch} loss: {loss:.3f}")

def visualize_network_architecture():
    """Visualize the neural network architecture"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define positions for neurons
    input_layer = [(2, 7), (2, 3)]
    hidden_layer = [(5, 8), (5, 2)]
    output_layer = [(8, 5)]
    
    # Draw connections
    for inp in input_layer:
        for hid in hidden_layer:
            ax.plot([inp[0], hid[0]], [inp[1], hid[1]], 'gray', alpha=0.3, linewidth=1)
    
    for hid in hidden_layer:
        for out in output_layer:
            ax.plot([hid[0], out[0]], [hid[1], out[1]], 'gray', alpha=0.3, linewidth=1)
    
    # Draw neurons
    for pos in input_layer:
        circle = mpatches.Circle(pos, 0.3, color='#3498db', ec='black', linewidth=2, zorder=3)
        ax.add_patch(circle)
    
    for pos in hidden_layer:
        circle = mpatches.Circle(pos, 0.3, color='#2ecc71', ec='black', linewidth=2, zorder=3)
        ax.add_patch(circle)
    
    for pos in output_layer:
        circle = mpatches.Circle(pos, 0.3, color='#e74c3c', ec='black', linewidth=2, zorder=3)
        ax.add_patch(circle)
    
    # Add labels
    ax.text(2, 7.8, 'Input 1', ha='center', fontsize=10, weight='bold')
    ax.text(2, 2.2, 'Input 2', ha='center', fontsize=10, weight='bold')
    ax.text(5, 8.8, 'Hidden 1', ha='center', fontsize=10, weight='bold')
    ax.text(5, 1.2, 'Hidden 2', ha='center', fontsize=10, weight='bold')
    ax.text(8, 5.8, 'Output', ha='center', fontsize=10, weight='bold')
    
    # Add layer labels
    ax.text(2, 9.5, 'Input Layer', ha='center', fontsize=12, weight='bold', color='#3498db')
    ax.text(5, 9.5, 'Hidden Layer', ha='center', fontsize=12, weight='bold', color='#2ecc71')
    ax.text(8, 9.5, 'Output Layer', ha='center', fontsize=12, weight='bold', color='#e74c3c')
    
    # Add title
    ax.text(5, 0.5, 'Neural Network Architecture (2-2-1)', ha='center', fontsize=14, weight='bold')
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'nn_architecture.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Architecture visualization saved to {output_path}")
    plt.show()

def visualize_training_process(loss_history):
    """Visualize the training loss over epochs"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = [i * 10 for i in range(len(loss_history))]
    ax.plot(epochs, loss_history, linewidth=2, color='#e74c3c')
    ax.fill_between(epochs, loss_history, alpha=0.3, color='#e74c3c')
    
    ax.set_xlabel('Epoch', fontsize=12, weight='bold')
    ax.set_ylabel('Loss (MSE)', fontsize=12, weight='bold')
    ax.set_title('Neural Network Training Progress', fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'training_loss.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Training loss visualization saved to {output_path}")
    plt.show()

def visualize_decision_boundary(network, data, labels):
    """Visualize the decision boundary learned by the network"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create a mesh grid
    x_min, x_max = data[:, 0].min() - 5, data[:, 0].max() + 5
    y_min, y_max = data[:, 1].min() - 2, data[:, 1].max() + 2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # Predict for each point in the mesh
    Z = np.array([network.feedforward(np.array([x, y])) 
                  for x, y in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    contour = ax.contourf(xx, yy, Z, levels=20, cmap='RdYlGn', alpha=0.6)
    plt.colorbar(contour, ax=ax, label='Prediction')
    
    # Plot data points
    for i, (point, label) in enumerate(zip(data, labels)):
        color = '#2ecc71' if label == 1 else '#e74c3c'
        marker = 'o' if label == 1 else 's'
        ax.scatter(point[0], point[1], c=color, s=200, marker=marker, 
                  edgecolors='black', linewidth=2, zorder=3)
    
    ax.set_xlabel('Feature 1', fontsize=12, weight='bold')
    ax.set_ylabel('Feature 2', fontsize=12, weight='bold')
    ax.set_title('Decision Boundary Visualization', fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='#2ecc71', label='Class 1'),
        mpatches.Patch(color='#e74c3c', label='Class 0')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'decision_boundary.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Decision boundary visualization saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    print("=" * 60)
    print("Neural Network Visualization Tool")
    print("=" * 60)
    
    # Sample data (Gender classification example)
    data = np.array([
        [-2, -1],   # Alice (Female)
        [25, 6],    # Bob (Male)
        [17, 4],    # Charlie (Male)
        [-15, -6],  # Diana (Female)
    ])
    
    labels = np.array([1, 0, 0, 1])  # 1 = Female, 0 = Male
    
    print("\n1. Visualizing Network Architecture...")
    visualize_network_architecture()
    
    print("\n2. Training Neural Network...")
    network = VisualNeuralNetwork()
    network.train(data, labels, epochs=1000, learn_rate=0.1)
    
    print("\n3. Visualizing Training Progress...")
    visualize_training_process(network.loss_history)
    
    print("\n4. Visualizing Decision Boundary...")
    visualize_decision_boundary(network, data, labels)
    
    print("\n" + "=" * 60)
    print("All visualizations completed!")
    print("Check the 'assets' folder for saved images.")
    print("=" * 60)
