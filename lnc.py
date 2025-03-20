import numpy as np
import os
import torch
import itertools
from retnet import RetNet,RetNetConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from Bio import SeqIO
import argparse
import logging
from tqdm import tqdm
import contextlib

parser = argparse.ArgumentParser(description='RNA Secondary Structure Prediction Model Built on RNAret')

# training hyperparameters
parser.add_argument('-bs','--batch_size', type=int, default=50, help='Batch size for training and evaluation')
parser.add_argument('-k','--k_num', type=int, default=3, help='K-mer length')
parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train')
parser.add_argument('-lr','--learning_rate', type=float, default=0.00005, help='Learning rate for trainable parameters')
parser.add_argument('--weight_factor', type=float, default=1.0, help='Weight factor for False Negative loss')

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
parser.add_argument('--resnet_kernel_size', type=int, default=7, help='Kernel size in ResNet')

# training options
parser.add_argument('-d','--device', type=str, default=None, help='Device to use')
parser.add_argument('-n','--task_name', type=str, default='lnc_H', help='Name of the task')
parser.add_argument('-i','--train_path', nargs='+', default=['data/lncRNA_H/train.fa'], help='Paths to the training dataset directories')
parser.add_argument('-v','--val_path', nargs='+', default=None, help='Paths to the validation dataset directories')
parser.add_argument('-e','--test_path', nargs='+', default=['data/lncRNA_H/test.fa'], help='Paths to the test dataset directories for evaluation')
parser.add_argument('--val_ratio', type=float, default=0.1, help='Ratio of validation data if --val_path is not provided')
parser.add_argument('-l','--max_len', type=int, default=3000, help='Maximum sequence length')
parser.add_argument('-o','--output_path', type=str, default='model/lnc',help='Path to save the trained model')
parser.add_argument('--log_path', type=str, default='log/lnc', help='Path to save the log files')
parser.add_argument('--eval_only', action='store_true', help='Evaluate the model without training')
parser.add_argument('--pretrained_model_path', type=str, default=None, help='Path to the pretrained model')
parser.add_argument('--lnc_model_path', type=str, default=None, help='Path to the trained second structure prediction model')
parser.add_argument('--use_autocast', action='store_true', help='Use automatic mixed precision training')
args = parser.parse_args()
#defining model

class ResNet1DBlock(torch.nn.Module):
    def __init__(self, embed_dim, kernel_size=3, bias=False):
        super().__init__()

        self.conv_net = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=kernel_size, bias=bias, padding="same"),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1, bias=bias),
            torch.nn.ReLU()
        )
        
    def forward(self, x):
        residual = x

        x = self.conv_net(x)
        x = x + residual

        return x
    
class ResNet1D(torch.nn.Module):
    def __init__(self, embed_dim, num_blocks, kernel_size=3, bias=False):
        super().__init__()

        self.blocks = torch.nn.ModuleList(
            [
                ResNet1DBlock(embed_dim, kernel_size, bias=bias) for _ in range(num_blocks)
            ]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x   

class ResNet1D_classifier(torch.nn.Module):
    def __init__(self):
        super(ResNet1D_classifier, self).__init__()
        self.linear_in = torch.nn.Linear(args.retnet_embed_dim,args.resnet_hidden_dim)
        self.resnet = ResNet1D(args.resnet_hidden_dim, args.resnet_block_num, args.resnet_kernel_size, bias=True)
        self.conv_out = torch.nn.Conv1d(args.resnet_hidden_dim, 1, kernel_size=3, padding="same")
        
    def forward(self, x):
        x = self.linear_in(x)
        x = x.transpose(1,2)
        x = self.conv_out(x)
        x = x.squeeze(1)
        
        return x

class rnaret_lnc_model(torch.nn.Module): 
    def __init__(self, args):
        super(rnaret_lnc_model, self).__init__()
        self.ret = RetNet(args)
        self.classifier = ResNet1D_classifier()

    def forward(self, x): 
        _,aux  = self.ret(x)
        x = aux['inner_states'][-1]
        x = self.classifier(x)
        return x
    
#preparing dataset

class lnc_tokenizer():
    def __init__(self, k, max_len):
        self.k = k
        self.max_len = max_len
        
    def read(self, file_path):
        seqs = []
        seq_lens = []
        cds = []
        
        with open(file_path, 'r') as f:
            for record in SeqIO.parse(f, 'fasta'):
                seq = str(record.seq)
                if len(seq) >= self.max_len:
                    seq = seq[:self.max_len]
                seqs.append(seq)
                seq_lens.append(len(seq))
                if 'CDS:' not in record.description:
                    cds.append([0,0])
                else:
                    parts = record.description.split('|')
                    cds_part = next((part for part in parts if part.startswith('CDS:')), None)
                    start, end = map(int, cds_part.split(':')[1].split('-'))
                    cds.append([start-1, end])
                    
        return seqs, cds, seq_lens
        
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
        tokens[self.k//2+len(indices):+len(indices)+self.k-1] = 1
        return tokens

class lnc_dataset(torch.utils.data.Dataset):
    def __init__(self, file, k, max_len):
        self.max_len = max_len
        self.tokenizer = lnc_tokenizer(k=k, max_len=max_len)
        self.seqs, self.cds, self.seq_lens = self.tokenizer.read(file)
        
    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        start, end = self.cds[idx]
        seq = self.tokenizer.tokenize(self.seqs[idx])
        mask = np.zeros(self.max_len, dtype=np.int16)
        mask[:self.seq_lens[idx]] = 1
        label = np.zeros(self.max_len, dtype=np.int16) 
        label[start:end] = 1
        return seq, label, mask
    
class post_process(torch.nn.Module):
    def __init__(self):
        super(post_process, self).__init__()
        
    def forward(self, x, mask):
        with torch.no_grad():
            prob = torch.sigmoid(x) * mask
            seq_prob = prob.unfold(1, 30, 1).mean(dim=2).max(dim=1).values
        return seq_prob


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
    
    model = rnaret_lnc_model(model_config)
    if args.lnc_model_path is not None:
        model.load_state_dict(torch.load(args.lnc_model_path,weights_only=True,map_location=device))
    if args.pretrained_model_path:
        model.ret.load_state_dict(torch.load(args.pretrained_model_path,weights_only=True,map_location=device))
    post_processor = post_process()
    model = model.to(device)
      
    if args.train_path is not None and not args.eval_only:
        datasets = []
        for data_dir in args.train_path:
            dataset = lnc_dataset(data_dir, max_len=args.max_len, k=args.k_num)
            datasets.append(dataset)
        train_dataset = torch.utils.data.ConcatDataset(datasets)
        if args.val_path is not None:
            val_datasets = []
            for data_dir in args.val_path:
                dataset = lnc_dataset(data_dir, max_len=args.max_len, k=args.k_num)
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
            dataset = lnc_dataset(data_dir, max_len=args.max_len, k=args.k_num)
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
                all_probs = []
                all_preds = []
                all_labels = []
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
                        prob = post_processor(prob, mask)
                        pred = (prob >= 0.5).int()
                        label = (torch.sum(y, dim=1)>0).int()
                        total_loss.append(loss.item())
                        all_probs.extend(prob.cpu().numpy())
                        all_preds.extend(pred.cpu().numpy())
                        all_labels.extend(label.cpu().numpy())
                    pbar.set_postfix(loss=loss.item())
                auc = roc_auc_score(all_labels, all_probs)
                accuracy = accuracy_score(all_labels, all_preds)
                precision = precision_score(all_labels, all_preds)
                recall = recall_score(all_labels, all_preds)
                f1 = f1_score(all_labels, all_preds)
                logger.info(f'Validation Set —— Epoch: {epoch}, Loss: {sum(total_loss)/len(total_loss):.5f}, Accuracy: {accuracy:.4f}')
                logger.info(f'Validation Set —— Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}')
            
        torch.save(model.state_dict(), f"{args.output_path}/{args.task_name}_{args.k_num}mer.pth")
    
    if args.test_path is not None:
        model.eval()
        with torch.no_grad():
            all_probs = []
            all_preds = []
            all_labels = []
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
                    prob = post_processor(prob, mask)
                    pred = (prob >= 0.5).int()
                    label = (torch.sum(y, dim=1)>0).int()
                    all_probs.extend(prob.cpu().numpy())
                    all_preds.extend(pred.cpu().numpy())
                    all_labels.extend(label.cpu().numpy())
                    total_loss.append(loss.item())
                pbar.set_postfix(loss=loss.item())
            auc = roc_auc_score(all_labels, all_probs)
            accuracy = accuracy_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds)
            recall = recall_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds)
            logger.info(f'Test Set —— Loss: {sum(total_loss)/len(total_loss):.5f}, Accuracy: {accuracy:.4f}')
            logger.info(f'Test Set —— Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}')
