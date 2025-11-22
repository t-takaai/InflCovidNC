import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionMIL(nn.Module):
    """
    Attention-based Multiple Instance Learning model for binary classification.
    """
    def __init__(self, feature_dim, hidden_dim=128):
        super(AttentionMIL, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softmax(dim=1)
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, bags):
        # bags: (batch_size, bag_size, feature_dim)
        batch_size, bag_size, _ = bags.size()
        
        # Extract features for each instance
        h = self.feature_extractor(bags)  # (batch_size, bag_size, hidden_dim)
        
        # Compute attention weights
        attention_weights = self.attention(h)  # (batch_size, bag_size, 1)
        
        # Weighted sum of instance features
        weighted_h = torch.sum(h * attention_weights, dim=1)  # (batch_size, hidden_dim)
        
        # Classification
        logits = self.classifier(weighted_h)  # (batch_size, 1)
        return logits, attention_weights
    

class AttentionMIL3class(nn.Module):
    """
    Attention-based Multiple Instance Learning model for 3-class classification (infl, covid, nc).
    """
    def __init__(self, feature_dim, hidden_dim=128):
        super(AttentionMIL3class, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softmax(dim=1)
        )
        self.classifier = nn.Linear(hidden_dim, 3)  # Output 3 classes: infl, covid, nc

    def forward(self, bags):
        # bags: (batch_size, bag_size, feature_dim)
        batch_size, bag_size, _ = bags.size()
        
        # Extract features for each instance
        h = self.feature_extractor(bags)  # (batch_size, bag_size, hidden_dim)
        
        # Compute attention weights
        attention_weights = self.attention(h)  # (batch_size, bag_size, 1)
        
        # Weighted sum of instance features
        weighted_h = torch.sum(h * attention_weights, dim=1)  # (batch_size, hidden_dim)
        
        # Classification
        logits = self.classifier(weighted_h)  # (batch_size, 3)
        return logits, attention_weights