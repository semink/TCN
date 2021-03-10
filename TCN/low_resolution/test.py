import torch
import argparse
import torch.optim as optim
import torch.nn.functional as F
import sys

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
df_train, df_test = get_traffic_data()
train_dataset = TimeseriesDataset(df_train, seq_len=seq_length)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
input_dim = df_train.shape[-1]

# Note: We use a very simple setting here (assuming all levels have the same # of channels.
num_channels = [args.nhid] * args.levels
kernel_size = args.ksize
dropout = args.dropout
model = LowResolutionTCN(input_dim, compress_dim, seq_length, num_channels,
                         kernel_size=kernel_size, dropout=dropout)

if torch.cuda.is_available():
    model.cuda()

lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)


def train(epoch):
    global lr
    model.train()
    batch_idx = 1
    total_loss = 0
    for i, (x, y) in enumerate(train_loader):
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        x, y = x.float(), y.float()
        optimizer.zero_grad()
        output = model(x)
        loss = F.mse_loss(output, y)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        batch_idx += 1
        total_loss += loss.item()

        if batch_idx % args.log_interval == 0:
            cur_loss = total_loss / args.log_interval
            processed = min(i + batch_size, df_train.shape[0])
            print('Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.4f}\tLoss: {:.6f}'.format(
                epoch, processed, df_train.shape[0], 100. * processed / df_train.shape[0], lr, cur_loss))
            total_loss = 0


# def evaluate():
#     model.eval()
#     with torch.no_grad():
#         output = model(X_test)
#         test_loss = F.mse_loss(output, Y_test)
#         print('\nTest set: Average loss: {:.6f}\n'.format(test_loss.item()))
#         return test_loss.item()

if __name__=="__main__":
    for ep in range(1, epochs + 1):
        train(ep)
        # tloss = evaluate()