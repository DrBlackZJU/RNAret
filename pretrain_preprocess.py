from Bio import SeqIO
import numpy as np
import itertools
import argparse

def tokenize(seq, k, max_len):
    kmer_list = np.array([''.join(p) for p in itertools.product('ATCG', repeat=k)])
    kmer_to_index = {kmer: idx + 6 for idx, kmer in enumerate(kmer_list)}
    seq = seq.upper()
    seq = seq.replace('U','T')
    seq_len = len(seq)
    
    tokens = np.zeros(max_len, dtype=np.int16)
    
    kmers = np.array([seq[i:i+k] for i in range(seq_len - k + 1)])
    
    indices = np.array([kmer_to_index.get(kmer, 2) for kmer in kmers])
    
    tokens[:k//2] = 1
    tokens[k//2:k//2+len(indices)] = indices
    tokens[k//2+len(indices):k+len(indices)-1] = 1
    return tokens

def main(input_path, output_path, k, max_len):
    with open(output_path, 'a') as output:
        with open(input_path, 'r') as input:
            for record in SeqIO.parse(input, 'fasta'):
                seq = str(record.seq)
                seq = seq[:max_len]
                tokens = tokenize(seq, k, max_len)
                output.write(" ".join(map(str, tokens)) + "\n")
                output.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess RNA sequences.")
    parser.add_argument("input_path", type=str, help="Path to the input FASTA file.")
    parser.add_argument("output_path", type=str, help="Path to the output text file.")
    parser.add_argument("--k", type=int, default=5, help="K-mer length (default: 5).")
    parser.add_argument("--max_len", type=int, default=2000, help="Maximum sequence length (default: 2000).")
    
    args = parser.parse_args()
    main(args.input_path, args.output_path, args.k, args.max_len)