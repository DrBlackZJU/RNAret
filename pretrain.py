from retnet import RetNet,RetNetConfig
import torch
from Bio import SeqIO
import numpy as np
import itertools
import logging
import argparse


#defining model
class rnaret_pretrain_model(torch.nn.Module): 
    def __init__(self, config):
        super(rnaret_pretrain_model, self).__init__()
        self.ret = RetNet(config)

    def forward(self, x): 
        x,aux  = self.ret(x)
        return x

#special tokens - 0:PAD 1:FILL 2:UNK 3:SEQ/CLS 4:MASK

class fasta_tokenizer():
    def __init__(self, k, max_len):
        self.k = k
        self.max_len = max_len
        
    def read(self, file_path):
        seqs = []
        with open(file_path, 'r') as f:
            for record in SeqIO.parse(f, 'fasta'):
                seq = str(record.seq)[:self.max_len]
                seqs.append(seq)
        return seqs
        
    def tokenize(self, seq):
        kmer_list = np.array([''.join(p) for p in itertools.product('ATCG', repeat=self.k)])
        kmer_to_index = {kmer: idx + 6 for idx, kmer in enumerate(kmer_list)}
        seq = seq.upper()
        seq = seq.replace('U','T')
        seq_len = len(seq)
        
        tokens = np.zeros(self.max_len, dtype=np.int16)
        
        kmers = np.array([seq[i:i+self.k] for i in range(seq_len - self.k + 1)])
        
        indices = np.array([kmer_to_index.get(kmer, 2) for kmer in kmers])
        
        tokens[:self.k//2] = 1
        tokens[self.k//2:self.k//2+len(indices)] = indices
        tokens[self.k//2+len(indices):self.k+len(indices)-1] = 1
        return tokens
    
    def add_mask(self, tokens, mask_rate=0.15):
        length = np.count_nonzero(tokens)
        num_slices = int(np.ceil(mask_rate * length / self.k))
        mask = np.zeros(len(tokens),dtype=np.int16)
        
        start_indices = np.random.randint(0, length - self.k + 1, size=num_slices)
        
        for start in start_indices:
            mask[start:start+self.k] = 1
            
        target = np.where(mask != 1, -100, tokens)
        
        masked = tokens.copy()
        replace_with_mask = np.random.rand(len(tokens)) <= 0.8
        masked[np.logical_and(mask == 1, replace_with_mask)] = 4
        
        replace_with_random = ((np.random.rand(len(tokens)) <= 0.5) & (~replace_with_mask) & (mask)).astype(bool)
        random_words = np.random.randint(5, 5+4**self.k, size=len(tokens))
        masked[replace_with_random] = random_words[replace_with_random]
        
        return masked, target
    
class preprocessed_tokenizer():
    def __init__(self, k, max_len):
        self.k = k
        self.max_len = max_len
        
    def read(self, file_path):
        seqs = np.loadtxt(file_path,dtype=np.int16)
        return seqs
        
    def tokenize(self, seq):
        tokens = np.array(seq)[:self.max_len]
        return tokens
    
    def add_mask(self, tokens, mask_rate=0.15):
        length = np.count_nonzero(tokens)
        num_slices = int(np.ceil(mask_rate * length / self.k))
        mask = np.zeros(len(tokens),dtype=np.int16)
        
        start_indices = np.random.randint(0, length - self.k + 1, size=num_slices)
        
        for start in start_indices:
            mask[start:start+self.k] = 1
            
        target = np.where(mask != 1, -100, tokens)
        
        masked = tokens.copy()
        replace_with_mask = np.random.rand(len(tokens)) <= 0.8
        masked[np.logical_and(mask == 1, replace_with_mask)] = 4
        
        replace_with_random = ((np.random.rand(len(tokens)) <= 0.5) & (~replace_with_mask) & (mask)).astype(bool)
        random_words = np.random.randint(5, 5+4**self.k, size=len(tokens))
        masked[replace_with_random] = random_words[replace_with_random]
        
        return masked, target

class pretrain_dataset(torch.utils.data.Dataset):
    def __init__(self, file, max_len, k):
        self.data = []
        if file.endswith(('.fasta', '.fa')):
            self.tokenizer = fasta_tokenizer(k, max_len=max_len)
        else:
            self.tokenizer = preprocessed_tokenizer(k,max_len=max_len)
        self.data = self.tokenizer.read(file)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        tokens = self.tokenizer.tokenize(self.data[index])
        masked, target = self.tokenizer.add_mask(tokens)
        return tokens,masked,target

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pretrain RNAret Model')
    
    parser.add_argument('-bs','--batch_size', type=int, default=100, help='Batch size for training')
    parser.add_argument('-k','--k_num', type=int, default=1, help='K-mer length')
    parser.add_argument('-l','--max_len', type=int, default=2000, help='Maximum sequence length')
    parser.add_argument('--retnet_embed_dim', type=int, default=384, help='Embedding dimension for RetNet')
    parser.add_argument('--retnet_value_embed_dim', type=int, default=512, help='Value embedding dimension for RetNet')
    parser.add_argument('--retnet_ffn_embed_dim', type=int, default=512, help='FFN embedding dimension for RetNet')
    parser.add_argument('--retnet_layers', type=int, default=8, help='Number of layers in RetNet')
    parser.add_argument('--retnet_retention_heads', type=int, default=4, help='Number of retention heads in RetNet')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--activation_dropout', type=float, default=0.2, help='Activation dropout rate')
    
    parser.add_argument('-n','--task_name', type=str, default='pretrain', help='Name of the task')
    parser.add_argument('-i','--input_files', nargs='+', default=['rnacentral_active.fasta'], help='Paths to the pretrained fasta files')
    parser.add_argument('-o','--output_dir', type=str, default='model/pretrain',help='Path to save the pretrained model')
    parser.add_argument('--log_steps', type=int, default=50, help='Number of steps between logging')
    parser.add_argument('--save_steps', type=int, default=1000, help='Number of steps between saving checkpoints')
    
    parser.add_argument('-d','--device', type=str, default=None, help='Device to use')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('-lr','--learning_rate', type=float, default=0.0001, help='Initial learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='Minimum learning rate')
    parser.add_argument('--cycle_steps', type=int, default=50000, help='Number of steps per cycle')
    parser.add_argument('--max_steps', type=int, default=25000000, help='Maximum number of training steps')
    
    args = parser.parse_args()
    
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
        
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
    model = rnaret_pretrain_model(model_config)
    
    max_steps = args.max_steps
    step = 0
   
    model = model.to(device)
    model.train()

    datasets = []
    for file in args.input_files:
        dataset = pretrain_dataset(file, max_len=args.max_len, k=args.k_num)
        datasets.append(dataset)
    dataset = torch.utils.data.ConcatDataset(datasets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.cycle_steps, T_mult=1, eta_min=args.min_lr)
    scaler = torch.amp.GradScaler(enabled=True)
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(f"log/{args.task_name}_{args.k_num}mer.log"),
                    ])

    logger = logging.getLogger(__name__)
        
    while True:
        for i,(_,masked,target) in enumerate(dataloader):
            optimizer.zero_grad()
            
            masked = masked.to(torch.long).to(device)
            target = target.to(torch.long).to(device)

            with torch.amp.autocast(device_type='cuda'): 
                prob = model(masked)
                loss = criterion(prob.transpose(1,2), target)   

            scaler.scale(loss).backward()  
            scaler.step(optimizer) 
            scaler.update() 

            scheduler.step()
                
            if step % args.log_steps == 0:
                logger.info(f'Step: {step}, Loss: {loss.item()}')
            
            if step % args.save_steps == 0:
                torch.save(model.ret.state_dict(), f"{args.output_dir}/{args.task_name}_{args.k_num}mer_{step}.pth")
                     
            step += 1  
            
            if step > max_steps:
                break
        if step > max_steps:
            break
