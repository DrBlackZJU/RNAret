from Bio import SeqIO
import numpy as np
import itertools
import argparse

def tokenize(seq, k, max_len):
    """
    Convert an RNA sequence into a sequence of integer tokens using k-mers.

    Args:
        seq (str): RNA sequence (A, U, C, G).
        k (int): Length of k-mer.
        max_len (int): Maximum sequence length (padded/truncated).

    Returns:
        numpy.ndarray: Array of token IDs representing the sequence.
    """
    # Generate all possible k-mers of DNA alphabet (A, T, C, G)
    kmer_list = np.array([''.join(p) for p in itertools.product('ATCG', repeat=k)])
    # Assign unique IDs starting from 6 for each k-mer
    kmer_to_index = {kmer: idx + 6 for idx, kmer in enumerate(kmer_list)}

    # Convert RNA alphabet to DNA alphabet (replace U with T)
    seq = seq.upper().replace('U','T')
    seq_len = len(seq)
    
    # Initialize token array (0 = padding)
    tokens = np.zeros(max_len, dtype=np.int16)
    
    # Generate overlapping k-mers
    kmers = np.array([seq[i:i+k] for i in range(seq_len - k + 1)])
    
    # Map k-mers to their token IDs (default=2 if k-mer not found)
    indices = np.array([kmer_to_index.get(kmer, 2) for kmer in kmers])
    
    # Add special tokens (1 = padding marker around sequence)
    tokens[:k//2] = 1
    tokens[k//2:k//2+len(indices)] = indices
    tokens[k//2+len(indices):k+len(indices)-1] = 1
    
    return tokens


def main(input_path, output_path, k, max_len):
    """
    Read RNA sequences from a FASTA file, tokenize them, 
    and write tokenized sequences into a text file.

    Args:
        input_path (str): Path to input FASTA file.
        output_path (str): Path to save tokenized sequences.
        k (int): Length of k-mer.
        max_len (int): Maximum sequence length.
    """
    with open(output_path, 'a') as output:
        with open(input_path, 'r') as input:
            for record in SeqIO.parse(input, 'fasta'):
                seq = str(record.seq)
                # Truncate sequence if it exceeds max_len
                seq = seq[:max_len]
                tokens = tokenize(seq, k, max_len)
                # Save tokenized sequence as space-separated IDs
                output.write(" ".join(map(str, tokens)) + "\n")
                output.flush()


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser(description="Preprocess RNA sequences.")
    parser.add_argument("input_path", type=str, help="Path to the input FASTA file.")
    parser.add_argument("output_path", type=str, help="Path to the output text file.")
    parser.add_argument("-k","--k_num",type=int, default=5, help="K-mer length (default: 5).")
    parser.add_argument("--max_len", type=int, default=2000, help="Maximum sequence length (default: 2000).")
    
    args = parser.parse_args()
    # Call main function
    main(args.input_path, args.output_path, args.k_num, args.max_len)
