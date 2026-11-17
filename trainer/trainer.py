from io import open
import os
import glob
import re
import time
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import string
import unicodedata

# Check if CUDA is available
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

torch.set_default_device(device)
print(f"Using device = {torch.get_default_device()}")

#allowed characters
allowed_characters = string.ascii_letters + string.digits + "\n" + ":" + "_"
n_letters = len(allowed_characters)

# turn unicode string to ASCII
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in allowed_characters
    )

# find letter index from all letters
def letter_to_index(letter):
    if letter in allowed_characters:
        return allowed_characters.find(letter)
    else:
        return allowed_characters.find('_')
        
# turn line into a <line_length x 1 x n_letters> tensor
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor

class ConversationDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.load_time = time.localtime
        self.input_tensors = []
        self.output_tensors = []

        text_files = sorted(glob.glob(os.path.join(data_dir, '*.txt')))
        for filename in text_files:
            with open(filename, encoding='utf-8') as f:
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
                            self.input_tensors.append(line_to_tensor(unicode_to_ascii(prompt)))
                            self.output_tensors.append(line_to_tensor(unicode_to_ascii(response)))
                        history.append(line + "\n")

    def __len__(self):
        return len(self.input_tensors)

    def __getitem__(self, idx):
        return self.input_tensors[idx], self.output_tensors[idx]

alldata = ConversationDataset('data')
print(f"Loaded {len(alldata)} conversation pairs from data directory.")

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharRNN, self).__init__()

        self.rnn = nn.RNN(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input_tensor):
        rnn_out, _ = self.rnn(input_tensor)
        output = self.h2o(rnn_out)
        return self.softmax(output)

n_hidden = 128
model = CharRNN(n_letters, n_hidden, n_letters)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.NLLLoss()

# Train using explicit input and output tensors from each dialogue turn
for epoch in range(30):
    model.train()
    total_loss = 0.0
    num_targets = 0

    for input_tensor, output_tensor in zip(alldata.input_tensors, alldata.output_tensors):
        optimizer.zero_grad()

        logits = model(input_tensor)
        if output_tensor.size(0) > logits.size(0):
            continue

        logits = logits[-output_tensor.size(0):]
        logits = logits.view(-1, n_letters)
        target = output_tensor.argmax(dim=2).squeeze(1).long().view(-1)

        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_targets += target.numel()

    if epoch % 5 == 0 or epoch == 29:
        avg_loss = total_loss / max(num_targets, 1)
        print(f"Epoch {epoch:02d} avg loss: {avg_loss:.4f}")


def generate_text(seed, length=50, temperature=0.8, max_repeat=8):
    model.eval()
    current = seed
    recent = []

    with torch.no_grad():
        for _ in range(length):
            seq = line_to_tensor(current[-200:])
            logits = model(seq)
            dist = logits[-1, 0] / temperature
            probs = torch.softmax(dist, dim=0)
            next_index = torch.multinomial(probs, 1).item()
            next_char = allowed_characters[next_index]

            if next_char == '\n' and current.endswith('\n'):
                break

            current += next_char
            recent.append(next_char)
            if len(recent) > max_repeat and len(set(recent[-max_repeat:])) == 1:
                break

    return current

print("Generated:")
print(generate_text("you: hello", length=30, temperature=0.8, max_repeat=6))