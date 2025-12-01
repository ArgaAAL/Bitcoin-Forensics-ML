import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.loader import NeighborLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("🚀 SIMPLE GNN RANSOMWARE DETECTION MODEL")
print("="*60)

# Step 1: Load the data
print("[INFO] Loading wallet features and classes...")
wallets_df = pd.read_csv('./EllipticPlusPlus-main/Actors Dataset/wallets_features_classes_combined.csv')
print(f"[SUCCESS] Loaded {len(wallets_df)} wallet records")

print("[INFO] Loading address-to-address edges...")
edges_df = pd.read_csv('./EllipticPlusPlus-main/Actors Dataset/AddrAddr_edgelist.csv')
print(f"[SUCCESS] Loaded {len(edges_df)} edges")

# Step 2: Apply your enhanced feature engineering
print("[INFO] Creating enhanced pattern features...")

def create_enhanced_pattern_features(df):
    """Create the same 10 enhanced features from your XGBoost model"""
    
    # 1. Partner Transaction Ratio (connectivity)
    df['partner_transaction_ratio'] = (
        df.get('transacted_w_address_total', 0) / 
        (df.get('total_txs', 1) + 1e-8)
    )
    
    # 2. Activity Density (txs per block)
    df['activity_density'] = (
        df.get('total_txs', 0) / 
        (df.get('lifetime_in_blocks', 1) + 1e-8)
    )
    
    # 3. Transaction Size Variance (volatility)
    df['transaction_size_variance'] = (
        df.get('btc_transacted_max', 0) - df.get('btc_transacted_min', 0)
    ) / (df.get('btc_transacted_mean', 1) + 1e-8)
    
    # 4. Flow Imbalance (money laundering indicator)
    df['flow_imbalance'] = (
        (df.get('btc_sent_total', 0) - df.get('btc_received_total', 0)) / 
        (df.get('btc_transacted_total', 1) + 1e-8)
    )
    
    # 5. Temporal Spread (time pattern)
    df['temporal_spread'] = (
        df.get('last_block_appeared_in', 0) - df.get('first_block_appeared_in', 0)
    ) / (df.get('num_timesteps_appeared_in', 1) + 1e-8)
    
    # 6. Fee Percentile (urgency indicator)
    df['fee_percentile'] = (
        df.get('fees_total', 0) / 
        (df.get('btc_transacted_total', 1) + 1e-8)
    )
    
    # 7. Interaction Intensity (network centrality)
    df['interaction_intensity'] = (
        df.get('num_addr_transacted_multiple', 0) / 
        (df.get('transacted_w_address_total', 1) + 1e-8)
    )
    
    # 8. Value Per Transaction (transaction size)
    df['value_per_transaction'] = (
        df.get('btc_transacted_total', 0) / 
        (df.get('total_txs', 1) + 1e-8)
    )
    
    # 9. RANSOMWARE-SPECIFIC: Burst Activity (rapid txs)
    df['burst_activity'] = (
        df.get('total_txs', 0) * df.get('activity_density', 0)
    )
    
    # 10. RANSOMWARE-SPECIFIC: Mixing Intensity (obfuscation)
    df['mixing_intensity'] = (
        df.get('partner_transaction_ratio', 0) * df.get('interaction_intensity', 0)
    )
    
    return df

# Apply feature engineering
wallets_df = create_enhanced_pattern_features(wallets_df)
print(f"[SUCCESS] Enhanced features created. Now have {len(wallets_df.columns)} columns")

# Step 3: Prepare node data
print("[INFO] Preparing node data...")

# Clean data - keep only labeled addresses (1=Illicit, 2=Licit)
wallets_clean = wallets_df[wallets_df['class'].isin([1, 2])].copy()
print(f"[INFO] Labeled addresses: {len(wallets_clean)}")
print(f"[INFO] Class distribution: {wallets_clean['class'].value_counts().to_dict()}")

# Remap classes: 1->1 (Illicit), 2->0 (Licit)
wallets_clean['class'] = wallets_clean['class'].map({1: 1, 2: 0})

# Create address to index mapping
unique_addresses = wallets_clean['address'].unique()
addr_to_idx = {addr: idx for idx, addr in enumerate(unique_addresses)}
idx_to_addr = {idx: addr for addr, idx in addr_to_idx.items()}

print(f"[INFO] Created mapping for {len(addr_to_idx)} unique addresses")

# Step 4: Prepare features and labels
exclude_cols = ['address', 'Time step', 'class']
feature_cols = [col for col in wallets_clean.columns if col not in exclude_cols]

# Get features and labels for mapped addresses
features_list = []
labels_list = []

for addr in unique_addresses:
    addr_data = wallets_clean[wallets_clean['address'] == addr].iloc[0]
    features_list.append(addr_data[feature_cols].values)
    labels_list.append(addr_data['class'])

X = np.array(features_list)
y = np.array(labels_list)

print(f"[INFO] Feature matrix shape: {X.shape}")
print(f"[INFO] Labels shape: {y.shape}")

# Step 5: Prepare edges
print("[INFO] Preparing graph edges...")

# Filter edges to only include addresses in our labeled dataset
edges_filtered = []
for _, row in edges_df.iterrows():
    input_addr = row['input_address']
    output_addr = row['output_address']
    
    if input_addr in addr_to_idx and output_addr in addr_to_idx:
        input_idx = addr_to_idx[input_addr]
        output_idx = addr_to_idx[output_addr]
        edges_filtered.append([input_idx, output_idx])

edge_index = torch.tensor(edges_filtered, dtype=torch.long).t().contiguous()
print(f"[INFO] Graph edges: {edge_index.shape[1]} edges between labeled addresses")

# Step 6: Scale features
print("[INFO] Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to PyTorch tensors
x = torch.tensor(X_scaled, dtype=torch.float)
y = torch.tensor(y, dtype=torch.long)

print(f"[INFO] Final graph structure:")
print(f"  - Nodes: {x.shape[0]}")
print(f"  - Node features: {x.shape[1]}")
print(f"  - Edges: {edge_index.shape[1]}")
print(f"  - Classes: Licit={sum(y==0)}, Illicit={sum(y==1)}")

# Step 7: Create PyTorch Geometric data object
data = Data(x=x, edge_index=edge_index, y=y)

# Step 8: Define Simple GNN Model
class SimpleGNN(nn.Module):
    def __init__(self, num_features, hidden_dim=64, num_classes=2):
        super(SimpleGNN, self).__init__()
        
        # Graph convolutional layers
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim//2)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim//2, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x, edge_index, batch=None):
        # Graph convolutions with residual connections
        h1 = F.relu(self.conv1(x, edge_index))
        h1 = F.dropout(h1, p=0.3, training=self.training)
        
        h2 = F.relu(self.conv2(h1, edge_index))
        h2 = F.dropout(h2, p=0.3, training=self.training)
        
        h3 = F.relu(self.conv3(h2, edge_index))
        
        # Classification
        out = self.classifier(h3)
        
        return out

# Step 9: Train-test split
train_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
test_mask = torch.zeros(data.y.size(0), dtype=torch.bool)

# Stratified split
train_idx, test_idx = train_test_split(
    range(len(y)), test_size=0.2, random_state=42, 
    stratify=y.numpy()
)

train_mask[train_idx] = True
test_mask[test_idx] = True

print(f"[INFO] Train samples: {train_mask.sum()}")
print(f"[INFO] Test samples: {test_mask.sum()}")

# Step 10: Initialize and train model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleGNN(num_features=x.shape[1]).to(device)
data = data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

print(f"[INFO] Training on device: {device}")
print(f"[INFO] Model parameters: {sum(p.numel() for p in model.parameters())}")

# Training loop
print("\n" + "="*60)
print("🚀 TRAINING GNN MODEL")
print("="*60)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        model.eval()
        with torch.no_grad():
            pred = model(data.x, data.edge_index).argmax(dim=1)
            train_acc = float((pred[train_mask] == data.y[train_mask]).sum()) / train_mask.sum()
            test_acc = float((pred[test_mask] == data.y[test_mask]).sum()) / test_mask.sum()
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        model.train()

# Step 11: Final evaluation
print("\n" + "="*60)
print("📊 FINAL EVALUATION")
print("="*60)

model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)
    pred_proba = F.softmax(out, dim=1)[:, 1]  # Probability of being illicit
    pred = out.argmax(dim=1)
    
    # Test set evaluation
    y_test = data.y[test_mask].cpu().numpy()
    y_pred = pred[test_mask].cpu().numpy()
    y_pred_proba = pred_proba[test_mask].cpu().numpy()
    
    # Calculate metrics
    test_acc = float((pred[test_mask] == data.y[test_mask]).sum()) / test_mask.sum()
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"AUC Score: {auc_score:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Licit', 'Illicit']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print("   Predicted:  Licit  Illicit")
    print(f"   Licit:      {cm[0,0]:5d}     {cm[0,1]:4d}")
    print(f"   Illicit:      {cm[1,0]:3d}     {cm[1,1]:4d}")

# Step 12: Save the model
print("\n" + "="*60)
print("💾 SAVING GNN MODEL")
print("="*60)

# Save model and metadata
model_data = {
    'model_state_dict': model.state_dict(),
    'scaler': scaler,
    'addr_to_idx': addr_to_idx,
    'idx_to_addr': idx_to_addr,
    'feature_cols': feature_cols,
    'model_config': {
        'num_features': x.shape[1],
        'hidden_dim': 64,
        'num_classes': 2
    }
}

torch.save(model_data, 'simple_gnn_ransomware_model.pth')
print("[SUCCESS] GNN model saved to 'simple_gnn_ransomware_model.pth'!")

print(f"\n🎯 GNN Model Summary:")
print(f"  ✅ Nodes: {x.shape[0]} labeled addresses")
print(f"  ✅ Edges: {edge_index.shape[1]} connections")
print(f"  ✅ Features: {x.shape[1]} (56 original + 10 enhanced)")
print(f"  ✅ Test Accuracy: {test_acc:.4f}")
print(f"  ✅ AUC Score: {auc_score:.4f}")
print(f"  ✅ Ready for ensemble with XGBoost!")

# Step 13: Sample prediction function for new addresses
def predict_address(model, new_address_features, scaler, device):
    """
    Predict if a new address is illicit
    new_address_features: dict with same keys as your feature columns
    """
    model.eval()
    
    # Convert features to array
    features_array = np.array([new_address_features[col] for col in feature_cols]).reshape(1, -1)
    
    # Scale features
    features_scaled = scaler.transform(features_array)
    features_tensor = torch.tensor(features_scaled, dtype=torch.float).to(device)
    
    # Note: For new addresses, we'd need to construct a subgraph
    # This is a simplified version - in practice you'd need the address's neighbors
    with torch.no_grad():
        # For isolated prediction (no graph context), we'd need a different approach
        # This is where you'd integrate with XGBoost for addresses with no graph context
        pass
    
    return "Prediction function needs graph context - integrate with XGBoost for isolated addresses"

print(f"\n💡 Next steps:")
print(f"  1. Test this GNN model performance")
print(f"  2. Build ensemble with your XGBoost model")
print(f"  3. Handle new addresses with limited graph context")
print(f"  4. Add confidence scoring based on model agreement")