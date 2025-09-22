from retnet import RetNet, RetNetConfig
import torch
from Bio import SeqIO
import numpy as np
import itertools
import logging
import argparse


# ==========================
# Define RNAret Pretrain Model
# ==========================
class rnaret_pretrain_model(torch.nn.Module): 
    def __init__(self, config):
        super(rnaret_pretrain_model, self).__init__()
        # Initialize RetNet model with given configuration
        self.ret = RetNet(config)

    def forward(self, x): 
        # Forward pass: RetNet returns both main output and auxiliary info
        x, aux = self.ret(x)
        return x


# ==========================
# Special Tokens Definition
# ==========================
# 0: PAD
# 1: FILL
# 2: UNK (unknown)
# 3: SEQ/CLS (sequence start or classification token)
# 4: MASK


# ==========================
# Tokenizer for FASTA sequences
# ==========================
class fasta_tokenizer():
    def __init__(self, k, max_len):
        self.k = k                # k-mer length
        self.max_len = max_len    # maximum sequence length

    def read(self, file_path):
        """ Read sequences from FASTA file """
        seqs = []
        with open(file_path, 'r') as f:
            for record in SeqIO.parse(f, 'fasta'):
                seq = str(record.seq)[:self.max_len]
                seqs.append(seq)
        return seqs
        
    def tokenize(self, seq):
        """ Convert RNA sequence into k-mer tokens """
        # Build vocabulary: all possible ATCG k-mers
        kmer_list = np.array([''.join(p) for p in itertools.product('ATCG', repeat=self.k)])
        kmer_to_index = {kmer: idx + 6 for idx, kmer in enumerate(kmer_list)}  # reserve first 6 tokens

        # Preprocess: convert to uppercase, replace U with T
        seq = seq.upper().replace('U','T')
        seq_len = len(seq)
        
        # Initialize token array with zeros (PAD)
        tokens = np.zeros(self.max_len, dtype=np.int16)
        
        # Extract k-mers
        kmers = np.array([seq[i:i+self.k] for i in range(seq_len - self.k + 1)])
        
        # Map k-mers to indices, unknown k-mers get 2 (UNK)
        indices = np.array([kmer_to_index.get(kmer, 2) for kmer in kmers])
        
        # Add filler tokens before and after sequence
        tokens[:self.k//2] = 1
        tokens[self.k//2:self.k//2+len(indices)] = indices
        tokens[self.k//2+len(indices):self.k+len(indices)-1] = 1
        return tokens
    
    def add_mask(self, tokens, mask_rate=0.15):
        """ Apply random masking for pretraining (similar to BERT MLM) """
        length = np.count_nonzero(tokens)  # number of non-PAD tokens
        num_slices = int(np.ceil(mask_rate * length / self.k))  # number of masked k-mers
        mask = np.zeros(len(tokens), dtype=np.int16)
        
        # Randomly select start positions for masking
        start_indices = np.random.randint(0, length - self.k + 1, size=num_slices)
        for start in start_indices:
            mask[start:start+self.k] = 1
            
        # Target: only keep original tokens for masked positions, others -100 (ignore index in loss)
        target = np.where(mask != 1, -100, tokens)
        
        # Copy tokens for masked input
        masked = tokens.copy()
        
        # 80% chance: replace with [MASK] token (4)
        replace_with_mask = np.random.rand(len(tokens)) <= 0.8
        masked[np.logical_and(mask == 1, replace_with_mask)] = 4
        
        # 10% chance: replace with random k-mer index
        replace_with_random = ((np.random.rand(len(tokens)) <= 0.5) & (~replace_with_mask) & (mask)).astype(bool)
        random_words = np.random.randint(5, 5+4**self.k, size=len(tokens))
        masked[replace_with_random] = random_words[replace_with_random]
        
        return masked, target


# ==========================
# Tokenizer for Preprocessed Data
# (already tokenized sequences saved in file)
# ==========================
class preprocessed_tokenizer():
    def __init__(self, k, max_len):
        self.k = k
        self.max_len = max_len
        
    def read(self, file_path):
        """ Load preprocessed token sequences """
        seqs = np.loadtxt(file_path, dtype=np.int16)
        return seqs
        
    def tokenize(self, seq):
        """ Simply truncate to max_len """
        tokens = np.array(seq)[:self.max_len]
        return tokens
    
    def add_mask(self, tokens, mask_rate=0.15):
        """ Apply same masking strategy as fasta_tokenizer """
        length = np.count_nonzero(tokens)
        num_slices = int(np.ceil(mask_rate * length / self.k))
        mask = np.zeros(len(tokens), dtype=np.int16)
        
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


# ==========================
# Dataset Class for Pretraining
# ==========================
class pretrain_dataset(torch.utils.data.Dataset):
    def __init__(self, file, max_len, k):
        """ Choose tokenizer depending on input file type (FASTA or preprocessed) """
        self.data = []
        if file.endswith(('.fasta', '.fa')):
            self.tokenizer = fasta_tokenizer(k, max_len=max_len)
        else:
            self.tokenizer = preprocessed_tokenizer(k, max_len=max_len)
        self.data = self.tokenizer.read(file)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        """ Return original tokens, masked input, and prediction target """
        tokens = self.tokenizer.tokenize(self.data[index])
        masked, target = self.tokenizer.add_mask(tokens)
        return tokens, masked, target


# ==========================
# Training Script
# ==========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pretrain RNAret Model')
    
    # Training parameters
    parser.add_argument('-bs','--batch_size', type=int, default=100, help='Batch size for training')
    parser.add_argument('-k','--k_num', type=int, default=1, help='K-mer length')
    parser.add_argument('-l','--max_len', type=int, default=2000, help='Maximum sequence length')
    
    # RetNet hyperparameters
    parser.add_argument('--retnet_embed_dim', type=int, default=384, help='Embedding dimension for RetNet')
    parser.add_argument('--retnet_value_embed_dim', type=int, default=512, help='Value embedding dimension for RetNet')
    parser.add_argument('--retnet_ffn_embed_dim', type=int, default=512, help='FFN embedding dimension for RetNet')
    parser.add_argument('--retnet_layers', type=int, default=8, help='Number of layers in RetNet')
    parser.add_argument('--retnet_retention_heads', type=int, default=4, help='Number of retention heads in RetNet')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--activation_dropout', type=float, default=0.2, help='Activation dropout rate')
    
    # IO and logging
    parser.add_argument('-n','--task_name', type=str, default='pretrain', help='Name of the task')
    parser.add_argument('-i','--input_files', nargs='+', default=['rnacentral_active.fasta'], help='Paths to training fasta files')
    parser.add_argument('-o','--output_dir', type=str, default='model/pretrain', help='Path to save pretrained models')
    parser.add_argument('--log_steps', type=int, default=50, help='Steps between logging')
    parser.add_argument('--save_steps', type=int, default=1000, help='Steps between saving checkpoints')
    
    # Optimization
    parser.add_argument('-d','--device', type=str, default=None, help='Device (cpu/cuda)')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('-lr','--learning_rate', type=float, default=0.0001, help='Initial learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='Minimum learning rate')
    parser.add_argument('--cycle_steps', type=int, default=50000, help='Steps per cosine annealing cycle')
    parser.add_argument('--max_steps', type=int, default=25000000, help='Maximum training steps')
    
    args = parser.parse_args()
    
    # Select device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
        
    # Build model configuration
    model_config = RetNetConfig(
        vocab_size=4**args.k_num+6,  # vocabulary = all k-mers + special tokens
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

    # Build dataset and dataloader
    datasets = []
    for file in args.input_files:
        dataset = pretrain_dataset(file, max_len=args.max_len, k=args.k_num)
        datasets.append(dataset)
    dataset = torch.utils.data.ConcatDataset(datasets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Optimizer, loss, scheduler, AMP scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.cycle_steps, T_mult=1, eta_min=args.min_lr
    )
    scaler = torch.amp.GradScaler(enabled=True)
    
    # Logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(f"log/{args.task_name}_{args.k_num}mer.log"),
                    ])
    logger = logging.getLogger(__name__)
        
    # ==========================
    # Training Loop
    # ==========================
    while True:
        for i, (_, masked, target) in enumerate(dataloader):
            optimizer.zero_grad()
            
            masked = masked.to(torch.long).to(device)
            target = target.to(torch.long).to(device)

            # Forward pass with mixed precision
            with torch.amp.autocast(device_type='cuda'): 
                prob = model(masked)
                loss = criterion(prob.transpose(1,2), target)   

            # Backpropagation with gradient scaling
            scaler.scale(loss).backward()  
            scaler.step(optimizer) 
            scaler.update() 

            # Learning rate scheduling
            scheduler.step()
                
            # Logging
            if step % args.log_steps == 0:
                logger.info(f'Step: {step}, Loss: {loss.item()}')
            
            # Save checkpoint
            if step % args.save_steps == 0:
                torch.save(model.ret.state_dict(), f"{args.output_dir}/{args.task_name}_{args.k_num}mer_{step}.pth")
                     
            step += 1  
            
            # Stop if max steps reached
            if step > max_steps:
                break
        if step > max_steps:
            break
