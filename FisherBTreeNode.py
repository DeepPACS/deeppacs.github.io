import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple
import torch.distributed as dist

class FisherBTreeNode(nn.Module):
    """
    Single node in Fisher-Information guided B-tree neural network
    Each node runs on separate GPU
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 node_id: int, gpu_id: int, max_children: int = 4):
        super().__init__()
        self.node_id = node_id
        self.gpu_id = gpu_id
        self.max_children = max_children
        self.device = torch.device(f'cuda:{gpu_id}')
        
        # Node architecture
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        ).to(self.device)
        
        # Confidence classifier
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        ).to(self.device)
        
        # Router for child selection (if internal node)
        self.router = nn.Linear(hidden_dim, max_children).to(self.device)
        
        # Fisher Information tracking
        self.fisher_info = torch.zeros(sum(p.numel() for p in self.parameters())).to(self.device)
        self.fisher_update_freq = 100
        self.step_count = 0
        
        # Node statistics
        self.processing_count = 0
        self.confidence_threshold = 0.8
        self.children: List[Optional['FisherBTreeNode']] = [None] * max_children
        self.is_leaf = True
        
    def compute_fisher_information(self, data_loader):
        """Compute Fisher Information Matrix diagonal for this node"""
        self.eval()
        fisher_diag = torch.zeros_like(self.fisher_info)
        
        for data, _ in data_loader:
            data = data.to(self.device)
            
            # Forward pass
            features = self.feature_extractor(data)
            logits = self.confidence_head(features)
            
            # Compute gradients w.r.t. log-likelihood
            for i in range(logits.size(0)):
                self.zero_grad()
                log_prob = torch.log(logits[i]).sum()
                log_prob.backward(retain_graph=True)
                
                # Accumulate squared gradients (Fisher Information diagonal)
                param_idx = 0
                for param in self.parameters():
                    if param.grad is not None:
                        param_size = param.grad.numel()
                        fisher_diag[param_idx:param_idx + param_size] += param.grad.view(-1) ** 2
                        param_idx += param_size
        
        # Average over samples
        fisher_diag /= len(data_loader.dataset)
        self.fisher_info = fisher_diag
        return fisher_diag.sum().item()  # Return trace of Fisher matrix
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, bool, int]:
        """
        Forward pass with early exit decision
        Returns: (output, should_exit, selected_child_id)
        """
        x = x.to(self.device)
        self.processing_count += 1
        
        # Extract features
        features = self.feature_extractor(x)
        
        # Get confidence prediction
        confidence_output = self.confidence_head(features)
        max_confidence = torch.max(confidence_output, dim=-1)[0]
        
        # Adaptive threshold based on Fisher Information
        adaptive_threshold = self.get_adaptive_threshold()
        
        # Early exit decision
        should_exit = (max_confidence > adaptive_threshold).all()
        
        if should_exit or self.is_leaf:
            return confidence_output, True, -1
        
        # Route to child node
        routing_logits = self.router(features)
        child_id = torch.argmax(routing_logits, dim=-1)[0].item()
        
        return confidence_output, False, child_id
    
    def get_adaptive_threshold(self) -> float:
        """Calculate adaptive confidence threshold based on Fisher Information"""
        if self.fisher_info.sum() == 0:
            return self.confidence_threshold
        
        # Higher Fisher Info → Lower threshold (more confident in early exit)
        fisher_magnitude = torch.log(1 + self.fisher_info.sum())
        adaptive_factor = 1.0 / (1.0 + 0.1 * fisher_magnitude)
        return self.confidence_threshold * adaptive_factor
    
    def should_split(self, fisher_threshold: float = 10.0) -> bool:
        """Decide if node should be split based on Fisher Information"""
        if not self.is_leaf or len([c for c in self.children if c is not None]) >= self.max_children:
            return False
        
        fisher_trace = self.fisher_info.sum().item()
        load_factor = self.processing_count / (self.step_count + 1)
        
        return fisher_trace > fisher_threshold and load_factor > 1.5
    
    def should_merge_children(self, fisher_threshold: float = 2.0) -> bool:
        """Decide if children should be merged based on low Fisher Information"""
        if self.is_leaf:
            return False
        
        active_children = [c for c in self.children if c is not None]
        if len(active_children) <= 1:
            return False
        
        avg_fisher = np.mean([c.fisher_info.sum().item() for c in active_children])
        return avg_fisher < fisher_threshold

class FisherBTreeNetwork(nn.Module):
    """
    Complete Fisher-Information guided B-tree Neural Network
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 available_gpus: List[int]):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.available_gpus = available_gpus
        self.gpu_allocator = 0
        
        # Create root node
        self.root = FisherBTreeNode(
            input_dim, hidden_dim, output_dim, 
            node_id=0, gpu_id=available_gpus[0]
        )
        self.nodes = {0: self.root}
        self.next_node_id = 1
        
        # Tree balancing parameters
        self.balance_frequency = 1000
        self.step_count = 0
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the tree"""
        current_node = self.root
        path = [0]  # Track path for gradient routing
        
        while True:
            output, should_exit, child_id = current_node(x)
            
            if should_exit or child_id == -1:
                return output
            
            # Move to child node
            if current_node.children[child_id] is not None:
                current_node = current_node.children[child_id]
                path.append(current_node.node_id)
            else:
                # Child doesn't exist, exit with current output
                return output
    
    def compute_tree_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss with Fisher Information balancing term"""
        # Standard classification loss
        accuracy_loss = nn.CrossEntropyLoss()(outputs, targets)
        
        # Fisher Information balancing term
        fisher_values = []
        loads = []
        
        for node in self.nodes.values():
            fisher_values.append(node.fisher_info.sum().item())
            loads.append(node.processing_count)
        
        if len(fisher_values) > 1:
            fisher_variance = np.var(fisher_values)
            load_variance = np.var(loads)
            balance_loss = fisher_variance + 0.1 * load_variance
        else:
            balance_loss = 0.0
        
        total_loss = accuracy_loss + 0.01 * balance_loss
        return total_loss
    
    def balance_tree(self, data_loader):
        """Rebalance tree based on Fisher Information"""
        self.step_count += 1
        
        if self.step_count % self.balance_frequency != 0:
            return
        
        # Update Fisher Information for all nodes
        for node in self.nodes.values():
            node.compute_fisher_information(data_loader)
        
        # Check for splits
        nodes_to_split = []
        for node_id, node in self.nodes.items():
            if node.should_split():
                nodes_to_split.append(node_id)
        
        # Perform splits
        for node_id in nodes_to_split:
            self._split_node(node_id)
        
        # Check for merges
        nodes_to_merge = []
        for node_id, node in self.nodes.items():
            if node.should_merge_children():
                nodes_to_merge.append(node_id)
        
        # Perform merges
        for node_id in nodes_to_merge:
            self._merge_node_children(node_id)
    
    def _split_node(self, node_id: int):
        """Split a node into children"""
        parent_node = self.nodes[node_id]
        
        if not parent_node.is_leaf:
            return
        
        # Create child nodes
        num_children = min(2, len(self.available_gpus) - len(self.nodes))
        
        for i in range(num_children):
            child_gpu = self.available_gpus[self.gpu_allocator % len(self.available_gpus)]
            self.gpu_allocator += 1
            
            child_node = FisherBTreeNode(
                self.hidden_dim, self.hidden_dim, self.output_dim,
                node_id=self.next_node_id, gpu_id=child_gpu
            )
            
            parent_node.children[i] = child_node
            self.nodes[self.next_node_id] = child_node
            self.next_node_id += 1
        
        parent_node.is_leaf = False
        print(f"Split node {node_id} into {num_children} children")
    
    def _merge_node_children(self, node_id: int):
        """Merge children of a node"""
        parent_node = self.nodes[node_id]
        
        # Remove children
        children_to_remove = [c for c in parent_node.children if c is not None]
        
        for child in children_to_remove:
            if child.node_id in self.nodes:
                del self.nodes[child.node_id]
        
        parent_node.children = [None] * parent_node.max_children
        parent_node.is_leaf = True
        
        print(f"Merged children of node {node_id}")
    
    def get_tree_stats(self):
        """Get current tree statistics"""
        stats = {
            'num_nodes': len(self.nodes),
            'avg_fisher_info': np.mean([n.fisher_info.sum().item() for n in self.nodes.values()]),
            'avg_processing_load': np.mean([n.processing_count for n in self.nodes.values()]),
            'tree_depth': self._get_tree_depth(),
            'gpu_utilization': {gpu: len([n for n in self.nodes.values() if n.gpu_id == gpu]) 
                              for gpu in self.available_gpus}
        }
        return stats
    
    def _get_tree_depth(self) -> int:
        """Calculate maximum depth of the tree"""
        def get_depth(node, current_depth=0):
            if node.is_leaf:
                return current_depth
            
            max_child_depth = current_depth
            for child in node.children:
                if child is not None:
                    child_depth = get_depth(child, current_depth + 1)
                    max_child_depth = max(max_child_depth, child_depth)
            
            return max_child_depth
        
        return get_depth(self.root)

# Usage example
if __name__ == "__main__":
    # Initialize network with multiple GPUs (simulated with GPU IDs)
    available_gpus = [0, 1, 2, 3]  # Assuming 4x RTX 3090 GPUs
    
    model = FisherBTreeNetwork(
        input_dim=768,      # Example: BERT embeddings
        hidden_dim=512,
        output_dim=10,      # 10 classes
        available_gpus=available_gpus
    )
    
    # Example training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Simulate training data
    batch_size = 32
    dummy_data = torch.randn(batch_size, 768)
    dummy_targets = torch.randint(0, 10, (batch_size,))
    
    model.train()
    
    # Forward pass
    outputs = model(dummy_data)
    
    # Compute loss with Fisher Information balancing
    loss = model.compute_tree_loss(outputs, dummy_targets)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Periodic tree balancing
    # model.balance_tree(data_loader)  # Uncomment with real data loader
    
    # Print tree statistics
    stats = model.get_tree_stats()
    print("Tree Statistics:", stats)