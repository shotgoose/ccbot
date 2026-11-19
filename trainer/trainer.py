import os
import re
import glob
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import base64
import string
import Levenshtein
from torch.utils.data import Dataset, DataLoader

# check if cuda is available
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
torch.set_default_device(device)
print(f"Using device = {torch.get_default_device()}")

# get script directory
directory = os.path.dirname(__file__)

# allowed characters
allowed_characters = string.ascii_letters + string.digits + "\n: "

# turn word into vocabulary index
def word_to_index(word, vocabulary):
    if word in vocabulary:
        return vocabulary.index(word)
    else:
        vocabulary.append(word)
        return vocabulary.index(word)

# parse line into list of words and remove disallowed characters
def parse_line(line):
    line = line.strip()
    line = line.lower()
    for letter in line:
        if letter not in allowed_characters:
            line = line.replace(letter, "")
    return line.split()

def parse_block(block):
    lines = block.split("\n")
    out = []
    for line in lines:
        ln = parse_line(line)
        if len(out) > 0:
            out.append("\n")
        out.extend(ln)
    out.append("\n")
    return out

# turn block into a 1D tensor of vocabulary indices
def block_to_indices(block, vocabulary):
    words = parse_block(block)
    return torch.tensor([word_to_index(w, vocabulary) for w in words], dtype=torch.long)

# find the closest word in the vocabulary to the given word
def closest_word(word, vocabulary):
    if word in vocabulary:
        return word
    else:
        # find the closest word in the vocabulary using Levenshtein distance
        closest_word = None
        closest_distance = float('inf')
        for vocab_word in vocabulary:
            distance = Levenshtein.distance(word, vocab_word)
            if distance < closest_distance:
                closest_distance = distance
                closest_word = vocab_word
        return closest_word
    
# weight matrix -> int8 bytes (base64) plus one scale: w ~ scale * q
def quantize(t):
    t = t.double().cpu()
    scale = t.abs().max().item() / 127
    q = torch.round(t / scale).clamp(-127, 127).to(torch.int8).contiguous()
    return {
        "scale": scale,
        "rows": t.shape[0],
        "cols": t.shape[1],
        "data": base64.b64encode(q.numpy().tobytes()).decode("ascii"),
    }

# biases are tiny, so keep them as plain rounded floats
def bias_list(t):
    return torch.round(t.double().cpu(), decimals=4).tolist()

# lua conversion helper
def to_lua(x):
    if isinstance(x, dict):
        return "{" + ",".join(f"{k}={to_lua(v)}" for k, v in x.items()) + "}"
    if isinstance(x, list):
        return "{" + ",".join(to_lua(v) for v in x) + "}"
    if isinstance(x, str):
        return json.dumps(x)   # quoted string; "\n" escapes are valid Lua too
    return repr(x)             # numbers

# create dataset from data directory
class ConvoDataset(Dataset):
    def __init__(self, data_dir):
        self.pairs = []
        self.vocabulary = []
        for filename in glob.glob(os.path.join(directory, data_dir, "*.txt")):
            with open(filename, 'r', encoding='utf-8') as f:
                conversations = [
                    block.strip() for block in re.split(r"\n\s*\n", f.read().strip()) if block.strip()
                ]
                for block in conversations:
                    history = []
                    for line in block.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith("you:"):
                            history.append(line + "\n")
                        elif line.startswith("bot:"):
                            prompt = "".join(history)
                            response = line[4:].strip()
                            if prompt and response:
                                self.pairs.append((block_to_indices(prompt, self.vocabulary),
                                                   block_to_indices(response, self.vocabulary)))
                            history.append(line + "\n")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p, r = self.pairs[idx]
        seq = torch.cat((p, r))
        x = F.one_hot(seq[:-1], len(self.vocabulary)).float()   # (T, vocab)
        y = seq[1:].clone()                                      # (T,)
        y[:len(p) - 1] = -100   # ignore loss on prompt positions
        return x, y

# GRU with a linear layer mapping hidden state to vocabulary scores
class ChatGRU(nn.Module):
    def __init__(self, vocab_size, hidden_size=128):
        super().__init__()
        self.gru = nn.GRU(vocab_size, hidden_size, num_layers=1, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h=None):
        o, h = self.gru(x, h)
        return self.out(o), h

# create model
alldata = ConvoDataset('data')
model = ChatGRU(len(alldata.vocabulary), hidden_size=128)

# train the model
def train(model, dataset, n_epochs=10, learning_rate=0.001):
    model.train()
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True,
                             generator=torch.Generator(device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    best_loss = float('inf')

    print(f"{len(dataset)} conversation pairs loaded from data directory.")
    print(f"Starting training for {n_epochs} epochs with learning rate {learning_rate}...")
    for epoch in range(n_epochs):
        # decay learning rate
        lr = learning_rate * (0.9 ** epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        total_loss = 0
        for input_tensor, target_tensor in data_loader:
            optimizer.zero_grad()
            output, _ = model(input_tensor)                  # (1, T, vocab)
            loss = criterion(output.view(-1, len(alldata.vocabulary)), target_tensor.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(directory, "new_model.pth"))
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f} (best), Learning Rate: {lr:.6f}")
        else:
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}, Best Loss: {best_loss:.4f}, Learning Rate: {lr:.6f}")
            
    final = torch.load(os.path.join(directory, "new_model.pth"), map_location="cpu")
    print("Done. Dumping model .lua...")
    data = {
        "vocabulary": alldata.vocabulary,
        "hidden_size": final["gru.weight_hh_l0"].shape[1],
        "weight_ih": quantize(final["gru.weight_ih_l0"]),   # (3H, V), rows ordered r, z, n
        "weight_hh": quantize(final["gru.weight_hh_l0"]),   # (3H, H)
        "bias_ih": bias_list(final["gru.bias_ih_l0"]),      # (3H)
        "bias_hh": bias_list(final["gru.bias_hh_l0"]),      # (3H)
        "out_weight": quantize(final["out.weight"]),        # (V, H)
        "out_bias": bias_list(final["out.bias"]),           # (V)
    }
    export = "return {"
    for key, val in data.items():
        export += key + "=" + to_lua(val) + ","
    export += "}"
    
    with open(os.path.join(directory, "model.lua"), "w", encoding="utf-8") as f:
        f.write(export)
        
    print(f"Wrote model.lua ({os.path.getsize(os.path.join(directory, 'model.lua')) / 1e6:.2f} MB)")

train(model, alldata, n_epochs=7, learning_rate=0.005)