import numpy as np
import os
import torch
import itertools
from retnet import RetNet,RetNetConfig
from scipy.optimize import linear_sum_assignment
import argparse
import logging
from tqdm import tqdm
import contextlib

parser = argparse.ArgumentParser(description='RNA Secondary Structure Prediction Model Built on RNAret')

# training hyperparameters
parser.add_argument('-bs','--batch_size', type=int, default=1, help='Batch size for training and evaluation')
parser.add_argument('-k','--k_num', type=int, default=1, help='K-mer length')
parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train')
parser.add_argument('-lr','--learning_rate', type=float, default=0.00001, help='Learning rate for trainable parameters')
parser.add_argument('--weight_factor', type=float, default=10.0, help='Weight factor for False Negative loss')

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
parser.add_argument('-n','--task_name', type=str, default='ssp', help='Name of the task')
parser.add_argument('-i','--train_path', nargs='+', default=['data/RNAStrAlign_new'], help='Paths to the training dataset directories')
parser.add_argument('-v','--val_path', nargs='+', default=None, help='Paths to the validation dataset directories')
parser.add_argument('-e','--test_path', nargs='+', default=['data/archiveII_new'], help='Paths to the test dataset directories for evaluation')
parser.add_argument('--val_ratio', type=float, default=0.1, help='Ratio of validation data if --val_path is not provided')
parser.add_argument('-l','--max_len', type=int, default=600, help='Maximum sequence length')
parser.add_argument('-o','--output_path', type=str, default='model/ssp',help='Path to save the trained model')
parser.add_argument('--log_path', type=str, default='log/ssp', help='Path to save the log files')
parser.add_argument('--eval_only', action='store_true', help='Evaluate the model without training')
parser.add_argument('--pretrained_model_path', type=str, default=None, help='Path to the pretrained model')
parser.add_argument('--ssp_model_path', type=str, default=None, help='Path to the trained second structure prediction model')
parser.add_argument('--use_autocast', action='store_true', help='Use automatic mixed precision training')
args = parser.parse_args()

def constraint(seq, N):
    seq = seq.upper()
    sharp = args.min_bp_distance - 1
    if args.batch_size == 1:
        matrix = np.zeros((len(seq),len(seq)), dtype=int)
    else:
        matrix = np.zeros((N, N), dtype=int)

    for i in range(sharp, len(seq)):
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
        self.linear_in = torch.nn.Linear(2*args.retnet_embed_dim,args.resnet_hidden_dim)
        self.resnet = ResNet2D(args.resnet_hidden_dim, args.resnet_block_num, args.resnet_kernel_size, bias=True)
        self.conv_out = torch.nn.Conv2d(args.resnet_hidden_dim, 1, kernel_size=3, padding="same")
        

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
        with torch.no_grad():
            x = torch.sigmoid(x)
            x = x * mask
            
            sec_struct = torch.where(x > 0.5, torch.ones_like(x), torch.zeros_like(x))
            x = x * sec_struct
            
            B, L, _ = x.shape

            for b in range(B):
                tmp = x[b].clone()
                row_ind, col_ind = linear_sum_assignment(-tmp.cpu().numpy())
                binary_matrix = torch.zeros_like(tmp)

                for r, c in zip(row_ind, col_ind):
                    if col_ind[col_ind[c]] == c:
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
        
    
def parse_ct_file(file, max_len):
    try:
        matrix = np.zeros((max_len, max_len), dtype=np.int8)
        seq = ''

        with open(file, 'r') as f:
            if file.endswith(".ct"):
                next(f)
            for line in f:
                parts = line.strip().split()
                idx, nt, _, _, pair, _ = parts
                seq = seq + nt
                idx = int(idx) - 1
                pair = int(pair) - 1
                if pair != -1:
                    matrix[idx, pair] = 1
                    matrix[pair, idx] = 1
        if len(seq) > max_len:
            return None, None, None
        else:
            return matrix, len(seq), seq

    except Exception:
        return None, None, None
    
def parse_bpseq_file(file, max_len=512):
    try:
        matrix = np.zeros((max_len,max_len), dtype=np.int8)
        seq = ''

        with open(file, 'r') as f:
            if file.endswith(".bpseq"):
                for line in f:
                    parts = line.strip().split()
                    idx, nt, pair = parts
                    seq = seq + nt
                    idx = int(idx) - 1
                    pair = int(pair) - 1
                    if pair != -1:
                        matrix[idx, pair] = 1
                        matrix[pair, idx] = 1
        if args.batch_size == 1:
            matrix = matrix[:len(seq),:len(seq)]
        if len(seq) > max_len:
            return None, None, None
        else:
            return matrix, len(seq), seq

    except Exception:
        return None, None, None

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
        
        if args.batch_size == 1:
            tokens = np.zeros(seq_len, dtype=np.int16)
        else:
            tokens = np.zeros(self.max_len, dtype=np.int16)
        
        kmers = np.array([seq[i:i+self.k] for i in range(seq_len - self.k + 1)])
        
        indices = np.array([kmer_to_index.get(kmer, 2) for kmer in kmers])
        
        tokens[self.k//2:self.k//2+len(indices)] = indices[:]
        
        tokens[:self.k//2] = 1
        tokens[self.k//2+len(indices):self.k//2+len(indices)+(self.k-1)//2] = 1
        return tokens


class ssp_dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir,max_len, k=5):
        self.data_dir = data_dir
        self.matrices = []
        self.seqs = []
        self.constraint = []
        self.tokenizer = seq_tokenizer(k=k,max_len=max_len)

        for root, dirs, files in os.walk(data_dir):
            for filename in files:
                if filename.endswith(".bpseq"):
                    file_path = os.path.join(root, filename)
                    matrix, seq_len, seq = parse_bpseq_file(file_path, max_len=max_len)
                    if matrix is not None and seq_len <= max_len:
                        con_matrix = constraint(seq, max_len)
                        seq = self.tokenizer.tokenize(seq)
                        self.matrices.append(matrix)
                        self.seqs.append(seq)
                        self.constraint.append(con_matrix)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.matrices[idx], self.constraint[idx]
    
def compute_metrics(target, prediction):
    positive_mask = target == 1
    negative_mask = target == 0
    pred_positive_mask = prediction > 0
    
    tp = torch.sum(torch.logical_and(positive_mask, pred_positive_mask)).item()
    fp = torch.sum(torch.logical_and(negative_mask, pred_positive_mask)).item()
    fn = torch.sum(torch.logical_and(positive_mask, ~pred_positive_mask)).item()
    tn = torch.sum(torch.logical_and(negative_mask, ~pred_positive_mask)).item()
    
    precision = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fn != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
    
    return precision, recall, f1

    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{args.log_path}/{args.task_name}_{args.k_num}mer.log"),
            ])
    logger = logging.getLogger(__name__)
    
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    cm = torch.amp.autocast(device_type='cuda') if device.type != 'cpu' and args.use_autocast else contextlib.nullcontext()
        
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
    if args.ssp_model_path is not None:
        model.load_state_dict(torch.load(args.ssp_model_path,weights_only=True))
    if args.pretrained_model_path:
        model.ret.load_state_dict(torch.load(args.pretrained_model_path,weights_only=True))
    post_processor = post_process()
    model = model.to(device)
    
    if args.train_path is not None and not args.eval_only:
        datasets = []
        for data_dir in args.train_path:
            dataset = ssp_dataset(data_dir, max_len=args.max_len, k=args.k_num)
            datasets.append(dataset)
        train_dataset = torch.utils.data.ConcatDataset(datasets)
        if args.val_path is not None:
            val_datasets = []
            for data_dir in args.val_path:
                dataset = ssp_dataset(data_dir, max_len=args.max_len, k=args.k_num)
                val_datasets.append(dataset)
            val_dataset = torch.utils.data.ConcatDataset(val_datasets)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)
        else:
            total_samples = len(train_dataset)
            val_size = int(args.val_ratio * total_samples)
            train_size = total_samples - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)
    if args.test_path is not None:
        test_datasets = []
        for data_dir in args.test_path:
            dataset = ssp_dataset(data_dir, max_len=args.max_len, k=args.k_num)
            test_datasets.append(dataset)
        test_dataset = torch.utils.data.ConcatDataset(test_datasets)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)
    
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    scaler = torch.amp.GradScaler(enabled=True)
    
    if not args.eval_only:
        for epoch in range(args.num_epochs):
            model.train()
            total_loss = []
            pbar = tqdm(train_dataloader, desc=f"Training Epoch {epoch}")
            for i, (x, y, mask) in enumerate(pbar):
                x = x.to(torch.long).to(device)
                y = y.to(torch.float32).to(device)
                mask = mask.to(torch.int).to(device)
                optimizer.zero_grad()
                with cm:
                    prob = model(x)
                    loss = criterion(prob, y)
                    loss = loss * mask * torch.where(y == 1, args.weight_factor, 1.0)
                    loss = loss.sum() / mask.sum()
                    total_loss.append(loss.item())
                scaler.scale(loss).backward()  
                scaler.step(optimizer) 
                scaler.update()
                pbar.set_postfix(loss=loss.item())
            logger.info(f'Training Set —— Epoch: {epoch}, Loss: {sum(total_loss)/len(total_loss):.5f}')
                    
            model.eval()
            with torch.no_grad():
                total_precision = []
                total_recall = []
                total_f1 = []
                total_loss = []
                pbar = tqdm(val_dataloader, desc=f"Validation Epoch {epoch}")
                for x, y, mask in pbar:
                    x = x.to(torch.long).to(device)  
                    y = y.to(torch.float32).to(device)
                    mask = mask.to(torch.int).to(device)
                    with cm:
                        prob = model(x)
                        loss = criterion(prob, y)
                        loss = loss * mask * torch.where(y == 1, args.weight_factor, 1.0)
                        loss = loss.sum() / mask.sum()
                        total_loss.append(loss.item())
                    pbar.set_postfix(loss=loss.item())
                    pred = post_processor(prob, mask)
                    precision, recall, f1 = compute_metrics(y, pred)
                    total_precision.append(precision)
                    total_recall.append(recall)
                    total_f1.append(f1)
                precision = sum(total_precision) / len(total_precision)
                recall = sum(total_recall) / len(total_recall)
                f1 = sum(total_f1) / len(total_f1)
                logger.info(f'Validation Set —— Epoch: {epoch}, Loss: {sum(total_loss)/len(total_loss):.5f}')
                logger.info(f'Validation Set —— Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
            
        torch.save(model.state_dict(), f"{args.output_path}/{args.task_name}_{args.k_num}mer.pth")
        
    if args.test_path is not None:
        model.eval()
        with torch.no_grad():
            total_precision = []
            total_recall = []
            total_f1 = []
            total_loss = []
            pbar = tqdm(test_dataloader, desc="Testing")
            for x, y, mask in pbar:
                x = x.to(torch.long).to(device)  
                y = y.to(torch.float32).to(device)
                mask = mask.to(torch.int).to(device)
                with cm:
                    prob = model(x)
                    loss = criterion(prob, y)
                    loss = loss * mask * torch.where(y == 1, args.weight_factor, 1.0)
                    loss = loss.sum() / mask.sum()
                    total_loss.append(loss.item())
                pbar.set_postfix(loss=loss.item())
                pred = post_processor(prob, mask)
                precision, recall, f1 = compute_metrics(y, pred)
                total_precision.append(precision)
                total_recall.append(recall)
                total_f1.append(f1)
            precision = sum(total_precision) / len(total_precision)
            recall = sum(total_recall) / len(total_recall)
            f1 = sum(total_f1) / len(total_f1)
            logger.info(f'Test Set —— Loss: {sum(total_loss)/len(total_loss):.5f}')
            logger.info(f'Test Set —— Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
