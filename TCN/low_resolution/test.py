import torch
import argparse
import torch.optim as optim
import torch.nn.functional as F
import sys
from torch.utils.tensorboard import SummaryWriter

sys.path.append("../../")
from TCN.low_resolution.model import LowResolutionTCN
from TCN.low_resolution.utils import get_traffic_data, TimeseriesDataset

parser = argparse.ArgumentParser(description='Sequence Modeling - Low resolution TCN')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout applied to layers (default: 0.1)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit (default: 10)')
parser.add_argument('--ksize', type=int, default=7,
                    help='kernel size (default: 7)')
parser.add_argument('--levels', type=int, default=8,
                    help='# of levels (default: 8)')
parser.add_argument('--seq_len', type=int, default=288,
                    help='sequence length (default: 288)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=4e-3,
                    help='initial learning rate (default: 4e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=30,
                    help='number of hidden units per layer (default: 30)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--comp_dim', type=int, default=2,
                    help='compressed dimension (default: 2)')
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    print("You have a CUDA device, so use CUDA for training...")

n_classes = 1
batch_size = args.batch_size
seq_length = args.seq_len
epochs = args.epochs
compress_dim = args.comp_dim

print(args)

print("Producing data...")
df_train, df_valid = get_traffic_data()
train_dataset = TimeseriesDataset(df_train, seq_len=seq_length)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

valid_dataset = TimeseriesDataset(df_valid, seq_len=seq_length)
# Load entire dataset for validation
valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                           batch_size=df_valid.shape[0] - seq_length, shuffle=False)
x_valid, y_valid = next(iter(valid_loader))
x_valid, y_valid = x_valid.float(), y_valid.float()

if torch.cuda.is_available():
    x_valid, y_valid = x_valid.cuda(), y_valid.cuda()

input_dim = df_train.shape[-1]

# Note: We use a very simple setting here (assuming all levels have the same # of channels.
num_channels = [args.nhid] * args.levels
kernel_size = args.ksize
dropout = args.dropout
model = LowResolutionTCN(input_dim, compress_dim, seq_length, num_channels,
                         kernel_size=kernel_size, dropout=dropout)

writer = SummaryWriter()

if torch.cuda.is_available():
    model.cuda()

writer.add_graph(model, x_valid[:1, :, :])

lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)
loss_fn = F.l1_loss

total_train_N = df_train.shape[0] - seq_length

global_step = 0


def train(epoch):
    global lr, global_step
    model.train()
    total_loss = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        x, y = x.float(), y.float()
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        total_loss += loss.item()

        if (batch_idx + 1) % args.log_interval == 0:
            cur_loss = total_loss / args.log_interval
            processed = min((batch_idx + 1) * batch_size, total_train_N)
            valid_loss = evaluate(x_valid, y_valid)
            print(f'Train Epoch: {epoch:2d} [{processed:6d}/{total_train_N:6d}\
                    ({100. * processed / total_train_N:.0f}%)]\t\
                    Learning rate: {lr:.4f}\t\
                    Loss: {cur_loss:.6f}\t\
                    Valid Loss: {valid_loss:.6f}')
            total_loss = 0
            writer.add_scalars("MAE", {'Training': cur_loss, 'Validation': valid_loss}, global_step)
            global_step += 1



def evaluate(x, y):
    model.eval()
    with torch.no_grad():
        output = model(x)
        test_loss = loss_fn(output, y)
    return test_loss.item()


if __name__ == "__main__":
    for ep in range(epochs):
        train(ep+1)
    writer.close()
