import torch
import numpy as np
import itertools
from retnet import RetNet,RetNetConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import argparse
import logging
from tqdm import tqdm
import contextlib

parser = argparse.ArgumentParser(description='mRNA-miRNA rri Prediction Model Built on RNAret')

# training hyperparameters
parser.add_argument('-bs','--batch_size', type=int, default=64, help='Batch size for training and evaluation')
parser.add_argument('-k','--k_num', type=int, default=5, help='K-mer length')
parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train')
parser.add_argument('-lr','--learning_rate', type=float, default=0.00001, help='Learning rate for trainable parameters')

# pretrained RetNet hyperparameters
parser.add_argument('--retnet_embed_dim', type=int, default=384, help='Embedding dimension for RetNet')
parser.add_argument('--retnet_value_embed_dim', type=int, default=512, help='Value embedding dimension for RetNet')
parser.add_argument('--retnet_ffn_embed_dim', type=int, default=512, help='FFN embedding dimension for RetNet')
parser.add_argument('--retnet_layers', type=int, default=8, help='Number of layers in RetNet')
parser.add_argument('--retnet_retention_heads', type=int, default=4, help='Number of retention heads in RetNet')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
parser.add_argument('--activation_dropout', type=float, default=0.2, help='Activation dropout rate')

# training options
parser.add_argument('-d','--device', type=str, default=None, help='Device to use')
parser.add_argument('-n','--task_name', type=str, default='MirTar', help='Name of the task')
parser.add_argument('-i','--train_path', nargs='+', default=['data/data_DeepMirTar_miRAW_noRepeats_3folds_train.txt'], help='Paths to the training and validation dataset directories')
parser.add_argument('-v','--val_path', nargs='+', default=None, help='Paths to the validation dataset directories')
parser.add_argument('-e','--test_path', nargs='+', default=['data/data_DeepMirTar_miRAW_noRepeats_3folds_test.txt'], help='Paths to the test dataset directories for evaluation')
parser.add_argument('--val_ratio', type=float, default=0.1, help='Ratio of validation data if --val_path is not provided')
parser.add_argument('-l','--max_len', type=int, default=80, help='Maximum sequence length')
parser.add_argument('-o','--output_path', type=str, default='model/rri',help='Path to save the trained model')
parser.add_argument('--log_path', type=str, default='log/rri', help='Path to save the log files')
parser.add_argument('--eval_only', action='store_true', help='Evaluate the model without training')
parser.add_argument('--pretrained_model_path', type=str, default=None, help='Path to the pretrained model')
parser.add_argument('--rri_model_path', type=str, default=None, help='Path to the trained second structure prediction model')
parser.add_argument('--use_autocast', action='store_true', help='Use automatic mixed precision training')
args = parser.parse_args()
    
#defining model
class rnaret_rri_model(torch.nn.Module): 
    def __init__(self, args):
        super(rnaret_rri_model, self).__init__()
        self.ret = RetNet(args)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(args.retnet_embed_dim,args.retnet_embed_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(args.retnet_embed_dim, 2)
        )

    def forward(self, x): 
        _,aux  = self.ret(x)
        x = aux['inner_states'][-1]
        x = x[:,0,:]
        x = self.classifier(x)
        return x
    
#preparing dataset
class rri_tokenizer():
    def __init__(self, k, max_len):
        self.k = k
        self.max_len = max_len
        
    def read(self, file_path):
        miRNA = []
        mRNA = []
        labels = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                parts = line.split()
                miRNA.append(parts[1][::-1])
                mRNA.append(parts[3])
                labels.append(int(parts[4]))       
        return miRNA,mRNA,labels
        
    def tokenize(self, seq1, seq2):
        kmer_list = np.array([''.join(p) for p in itertools.product('ATCG', repeat=self.k)])
        kmer_to_index = {kmer: idx + 6 for idx, kmer in enumerate(kmer_list)}
        seq1, seq2 = seq1.upper(), seq2.upper()
        seq1, seq2 = seq1.replace('U','T'), seq2.replace('U','T')
        seq1_len, seq2_len = len(seq1), len(seq2)
        
        tokens = np.zeros(self.max_len, dtype=np.int16)
        
        kmers1 = np.array([seq1[i:i+self.k] for i in range(seq1_len - self.k + 1)])
        kmers2 = np.array([seq2[i:i+self.k] for i in range(seq2_len - self.k + 1)])
        
        indices1 = np.array([kmer_to_index.get(kmer, 2) for kmer in kmers1])
        indices2 = np.array([kmer_to_index.get(kmer, 2) for kmer in kmers2])
        
        tokens[:self.k//2] = 1
        tokens[self.k//2:self.k//2+len(indices1)] = indices1
        tokens[self.k//2+len(indices1):self.k-1+len(indices1)] = 1
        tokens[self.k-1+len(indices1)] = 3
        tokens[self.k+len(indices1):self.k+len(indices1)+self.k//2] = 1
        tokens[self.k+len(indices1)+self.k//2:self.k+len(indices1)+len(indices2)+self.k//2] = indices2
        tokens[self.k+len(indices1)+len(indices2)+self.k//2:2*self.k+len(indices1)+len(indices2)-1] = 1
        
        tokens = np.insert(tokens, 0, 3)

        return tokens
    
class rri_dataset(torch.utils.data.Dataset):
    def __init__(self, file, k, max_len):
        self.tokenizer = rri_tokenizer(k=k, max_len=max_len)
        self.miRNA, self.mRNA, self.labels = self.tokenizer.read(file)

    def __len__(self):
        return len(self.miRNA)

    def __getitem__(self, idx):
        seq = self.tokenizer.tokenize(self.miRNA[idx],self.mRNA[idx])
        label = self.labels[idx]
        return seq, label

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
    
    model = rnaret_rri_model(model_config)
    if args.rri_model_path is not None:
        model.load_state_dict(torch.load(args.rri_model_path,weights_only=False,map_location=device))
    if args.pretrained_model_path is not None:
        model.ret.load_state_dict(torch.load(args.pretrained_model_path,weights_only=False,map_location=device))
    model = model.to(device)

    if args.train_path is not None and not args.eval_only:
        datasets = []
        for data_dir in args.train_path:
            dataset = rri_dataset(data_dir, max_len=args.max_len, k=args.k_num)
            datasets.append(dataset)
        train_dataset = torch.utils.data.ConcatDataset(datasets)
        if args.val_path is not None:
            val_datasets = []
            for data_dir in args.val_path:
                dataset = rri_dataset(data_dir, max_len=args.max_len, k=args.k_num)
                val_datasets.append(dataset)
            val_dataset = torch.utils.data.ConcatDataset(val_datasets)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)
        else:
            total_samples = len(dataset)
            val_size = int(args.val_ratio * total_samples)
            train_size = total_samples - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)
    if args.test_path is not None:
        test_datasets = []
        for data_dir in args.test_path:
            dataset = rri_dataset(data_dir, max_len=args.max_len, k=args.k_num)
            test_datasets.append(dataset)
        test_dataset = torch.utils.data.ConcatDataset(test_datasets)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)
    
    optimizer = torch.optim.AdamW([
        {'params': model.ret.parameters(), 'lr': 0.00001},
        {'params': model.classifier.parameters(), 'lr': 0.00001}
    ])
    
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=True)
    
    if not args.eval_only:
        for epoch in range(args.num_epochs):
            model.train()
            total_loss = []
            pbar = tqdm(train_dataloader, desc=f"Training Epoch {epoch}")
            for i, (x, y) in enumerate(pbar):
                x = x.to(torch.long).to(device)
                y = y.to(torch.long).to(device)
                optimizer.zero_grad()
                with cm:
                    prob = model(x)
                    loss = criterion(prob, y)
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
                for x, y in pbar:
                    x = x.to(torch.long).to(device)  
                    y = y.to(torch.long).to(device)
                    with cm:
                        prob = model(x)
                        pred = torch.argmax(prob, dim=-1)
                        all_probs.extend(prob[:,1].cpu().numpy())
                        all_preds.extend(pred.cpu().numpy())
                        all_labels.extend(y.cpu().numpy())
                        loss = criterion(prob, y)
                        total_loss.append(loss.item())
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
            for x, y in pbar:
                x = x.to(torch.long).to(device)  
                y = y.to(torch.long).to(device)
                with cm:
                    prob = model(x)
                    pred = torch.argmax(prob, dim=-1)
                    all_probs.extend(prob[:,1].cpu().numpy())
                    all_preds.extend(pred.cpu().numpy())
                    all_labels.extend(y.cpu().numpy())
                    loss = criterion(prob, y)
                    total_loss.append(loss.item())
                pbar.set_postfix(loss=loss.item())
            auc = roc_auc_score(all_labels, all_probs)
            accuracy = accuracy_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds)
            recall = recall_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds)
            logger.info(f'Test Set —— Loss: {sum(total_loss)/len(total_loss):.5f}, Accuracy: {accuracy:.4f}')
            logger.info(f'Test Set —— Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}')
