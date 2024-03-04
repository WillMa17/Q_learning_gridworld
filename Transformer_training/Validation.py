class ValidDataset(Dataset):
    def __init__(self, data, token_to_idx):
        self.data = data
        self.token_to_idx = token_to_idx
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        X_sequence, Y_legal_moves = self.data[idx]
        
        X_indices = [self.token_to_idx.get(token, self.token_to_idx['<pad>']) for token in X_sequence]
        Y_legal_indices = []
        for legal_moves in Y_legal_moves:
            legal_indices = {self.token_to_idx[token] for token in legal_moves if token in self.token_to_idx}
            Y_legal_indices.append(legal_indices)
        
        return torch.tensor(X_indices, dtype=torch.long), Y_legal_indices

def collate_fn_valid(batch):
    Xs, legal_Ys = zip(*batch)
    Xs_padded = pad_sequence(Xs, batch_first=True, padding_value=token_to_idx['<pad>'])
    
    return Xs_padded, legal_Ys

correct_legal_predictions = 0
total_legal_predictions = 0

padding_index = token_to_idx['<pad>']  # Assuming you have a token_to_idx mapping that includes '<pad>'
test_valid_loader = DataLoader(test_valid_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn_valid)

model.eval()
with torch.no_grad():
    for X, legal_Ys in tqdm(test_valid_loader, desc="Evaluating", unit="batch"):
        X = X.to(device)
        logits = model(X)  # Shape [batch_size, seq_length, vocab_size]
        for batch_idx, legal_moves_set in enumerate(legal_Ys):  # Iterate over batch
            for i in range(X.size(1)):   
                if X[batch_idx, i] == padding_index:  # Skip padding positions
                    continue
                
                if i >= len(legal_moves_set) or not legal_moves_set[i]:  # Skip if beyond legal moves list or empty set
                    continue
                
                logits_slice = logits[batch_idx, i, :]
                probabilities = F.softmax(logits_slice, dim=-1)
                top1_pred = torch.argmax(probabilities, dim=-1).item()
                
                # Check if the top-1 prediction is within the legal moves
                if top1_pred in legal_moves_set[i]:
                    correct_legal_predictions += 1
                total_legal_predictions += 1

percentage_legal_move = (correct_legal_predictions / total_legal_predictions) if total_legal_predictions > 0 else 0
print(f"Percentage of legal moves: {percentage_legal_move * 100:.2f}%")
