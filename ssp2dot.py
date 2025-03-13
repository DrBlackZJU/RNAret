import numpy as np
import torch
import itertools
from retnet import RetNet,RetNetConfig
from scipy.optimize import linear_sum_assignment
import argparse
import contextlib

parser = argparse.ArgumentParser(description='RNA Secondary Structure Prediction Model Built on RNAret')

parser.add_argument('-k','--k_num', type=int, default=1, help='K-mer length')

# pretrained RetNet hyperparameters
parser.add_argument('--retnet_embed_dim', type=int, default=384, help='Embedding dimension for RetNet')
parser.add_argument('--retnet_value_embed_dim', type=int, default=512, help='Value embedding dimension for RetNet')
parser.add_argument('--retnet_ffn_embed_dim', type=int, default=512, help='FFN embedding dimension for RetNet')
parser.add_argument('--retnet_layers', type=int, default=8, help='Number of layers in RetNet')
parser.add_argument('--retnet_retention_heads', type=int, default=4, help='Number of retention heads in RetNet')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
parser.add_argument('--activation_dropout', type=float, default=0.2, help='Activation dropout rate')

# classifier hyperparameters
parser.add_argument('--resnet_block_num', type=int, default=16, help='Number of ResNet blocks in the classifier')
parser.add_argument('--resnet_hidden_dim', type=int, default=128, help='Hidden dimension in ResNet')
parser.add_argument('--resnet_kernel_size', type=int, default=3, help='Kernel size in ResNet')
parser.add_argument('--min_bp_distance', type=int, default=4, help='Minimum base pair distance to avoid sharp loops')

# training options
parser.add_argument('-d','--device', type=str, default=None, help='Device to use')
parser.add_argument('--ssp_model_path', type=str, default='model/ssp/all_1mer.pth', help='Path to the trained second structure prediction model')
parser.add_argument('--use_autocast', action='store_true', help='Use automatic mixed precision training')
parser.add_argument('-i','--input_file', type=str, default=None, help='Input fasta file')
parser.add_argument('-o','--output_file', type=str, default='output.dot', help='Output file')
args = parser.parse_args()
def constraint(seq):
    seq = seq.upper()
    L = len(seq)
    sharp = args.min_bp_distance - 1
    matrix = np.zeros((L, L), dtype=int)

    for i in range(sharp, L):
        for j in range(i-sharp):
            base_i = seq[i]
            base_j = seq[j]
            if  ((base_i == 'A' and base_j == 'U') or (base_i == 'U' and base_j == 'A') or
                (base_i == 'C' and base_j == 'G') or (base_i == 'G' and base_j == 'C') or
                (base_i == 'G' and base_j == 'U') or (base_i == 'U' and base_j == 'G') or
                base_i == 'N' or base_j == 'N'):
                matrix[i, j] = 1
    
    return matrix



class outer_concat(torch.nn.Module):
    def __init__(self):
        super(outer_concat, self).__init__()

    def forward(self, x1, x2):
        seq_len = x1.shape[1]
        x1 = x1.unsqueeze(-2).expand(-1, -1, seq_len, -1)
        x2 = x2.unsqueeze(-3).expand(-1, seq_len, -1, -1)
        x = torch.concat((x1,x2),dim=-1)

        return x


class ResNet2DBlock(torch.nn.Module):
    def __init__(self, embed_dim, kernel_size=3, bias=False):
        super().__init__()

        self.conv_net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=kernel_size, bias=bias, padding="same"),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1, bias=bias),
            torch.nn.ReLU()
        )

    def forward(self, x):
        residual = x

        x = self.conv_net(x)
        x = x + residual

        return x
    
class ResNet2D(torch.nn.Module):
    def __init__(self, embed_dim, num_blocks, kernel_size=3, bias=False):
        super().__init__()

        self.blocks = torch.nn.ModuleList(
            [
                ResNet2DBlock(embed_dim, kernel_size, bias=bias) for _ in range(num_blocks)
            ]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x    
    
class ResNet2D_classifier(torch.nn.Module):
    def __init__(self):
        super(ResNet2D_classifier, self).__init__()
        self.outer_concat = outer_concat()
        self.linear_in = torch.nn.Linear(768,128)
        self.resnet = ResNet2D(128, 16, 3, bias=True)
        self.conv_out = torch.nn.Conv2d(128, 1, kernel_size=3, padding="same")
        

    def forward(self, x):
        x = self.outer_concat(x, x)
        x = self.linear_in(x)
        x = x.permute(0,3,1,2)
        x = self.resnet(x)

        x = self.conv_out(x)
        x = x.squeeze(1)
        
        return x
    
class rnaret_ssp_model(torch.torch.nn.Module): 
    def __init__(self, args):
        super(rnaret_ssp_model, self).__init__()
        self.ret = RetNet(args)
        self.classifier = ResNet2D_classifier()

    def forward(self, x):      
        _,aux  = self.ret(x)
        x = aux['inner_states'][-1]
        x = self.classifier(x)
        return x
    
class post_process(torch.nn.Module):
    def __init__(self):
        super(post_process, self).__init__()
        
    def forward(self, x, mask):
        x = torch.sigmoid(x)
        x = x * mask
        
        sec_struct = torch.where(x > 0.5, torch.ones_like(x), torch.zeros_like(x))
        x = x * sec_struct
        
        B, L, _ = x.shape
        
        for b in range(B):
            tmp = x[b].clone()
            row_ind, col_ind = linear_sum_assignment(-tmp.detach().cpu().numpy())
            binary_matrix = torch.zeros_like(tmp)
            for r, c in zip(row_ind, col_ind):
                binary_matrix[r, c] = 1
                
            sec_struct[b] = binary_matrix

        sec_struct = sec_struct * mask
        
        sec_struct = sec_struct + sec_struct.transpose(1,2)
        
        for b in range(B):
            for i in range(L):
                if torch.sum(sec_struct[b, i, :]) > 1:
                    max_idx = torch.argmax(sec_struct[b, i, :])
                    sec_struct[b, i, :] = 0
                    sec_struct[b, i, max_idx] = 1

                if torch.sum(sec_struct[b, :, i]) > 1:
                    max_idx = torch.argmax(sec_struct[b, :, i])
                    sec_struct[b, :, i] = 0
                    sec_struct[b, max_idx, i] = 1
        
        return sec_struct
        
    
class seq_tokenizer():
    def __init__(self, k=5, max_len=512):
        self.k = k
        self.max_len = max_len
    def tokenize(self, seq):
        kmer_list = np.array([''.join(p) for p in itertools.product('ATCG', repeat=self.k)])
        kmer_to_index = {kmer: idx + 6 for idx, kmer in enumerate(kmer_list)}
        seq = seq.upper()
        seq = seq.replace('U','T')
        seq_len = len(seq)
        
        tokens = np.zeros(self.max_len, dtype=np.int16)
        
        kmers = np.array([seq[i:i+self.k] for i in range(seq_len - self.k + 1)])
        
        indices = np.array([kmer_to_index.get(kmer, 2) for kmer in kmers])
        
        tokens[self.k//2:self.k//2+len(indices)] = indices[:]
        
        tokens[:self.k//2] = 1
        tokens[self.k//2+len(indices):self.k//2+len(indices)+(self.k-1)//2] = 1
        return tokens
    
def parse_fasta_file(file):
    sequences = []
    description = ""
    seq = ""

    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq:
                    sequences.append((description, seq))
                    seq = ""
                description = line[1:]
            else:
                seq += line

        if seq:
            sequences.append((description, seq))

    return sequences
def generate_dot_bracket(seq_raw, pred):
    n = len(seq_raw)
    dot_bracket = ['.' for _ in range(n)]

    for i in range(n):
        for j in range(i,n):
            if pred[i][j] >= 1:
                dot_bracket[i] = '('
                dot_bracket[j] = ')'


    return ''.join(dot_bracket)
    
    
if __name__ == '__main__':
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    cm = torch.amp.autocast(device_type='cuda') if device.type != 'cpu' and args.use_autocast else contextlib.nullcontext()
    batch_size = 1
    model_config = RetNetConfig(
        vocab_size=4**args.k_num+6,
        retnet_embed_dim=args.retnet_embed_dim,
        retnet_value_embed_dim=args.retnet_value_embed_dim,
        retnet_ffn_embed_dim=args.retnet_ffn_embed_dim,
        retnet_layers=args.retnet_layers,
        retnet_retention_heads=args.retnet_retention_heads,
        dropout=args.dropout,
        activation_dropout=args.activation_dropout
    )
    model = rnaret_ssp_model(model_config)
    post_processor = post_process()

    if args.ssp_model_path is not None:
        model.load_state_dict(torch.load(args.ssp_model_path,weights_only=True))
    model = model.to(device)
    model.eval()

    sequences = parse_fasta_file(args.input_file)
    with torch.no_grad():
        with cm:
            for description, seq_raw in sequences:
                length = len(seq_raw)
                seq = seq_tokenizer(k=args.k_num,max_len=len(seq_raw)).tokenize(seq_raw)
                mask = constraint(seq_raw)
                x = torch.tensor(seq).unsqueeze(0).to(torch.int32).to(device)
                mask = torch.tensor(mask).unsqueeze(0).to(torch.int32).to(device)

                prob = model(x)
            
                pred = post_processor(prob,mask).squeeze(0).detach().cpu().numpy()
            
                dot_bracket = generate_dot_bracket(seq_raw, pred)
                with open(args.output_file, 'a') as f:
                    f.write(f">{description}\n")
                    f.write(f"{seq_raw}\n")
                    f.write(f"{dot_bracket}\n")
    