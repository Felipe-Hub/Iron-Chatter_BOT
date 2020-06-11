import numpy as np

def n_grams(enc_sent, n=4):
    """
    This function takes a sequence of encoded sentences and
    returns a tuple with two arrays. The first array contains
    vectors with padded n-grams combinations. The second array
    on the tuple contains vectors with the reversed n-grams.
    
    Both returned sequences are already post-padded with 0.
    The maximum length of the sequences is n-1.
    """  
    sequences = list()
    rev_sequences = list()
    
    # Here we separate into lists of n-grams:
    for enc in enc_sent:
        
        for i in range(len(enc)):
            sent_sequences = list()
            rev_sent_sequences = list()
            
            for ii in range(n+1):
                sequence = enc[i:i+ii]
                rev_sequence = sequence[::-1]
                
                # avoinding repeated n-grams in the same sentence
                if sequence not in sent_sequences and len(sequence) > 1:
                    sent_sequences += [sequence]
                    rev_sent_sequences += [rev_sequence]
                    
            sequences += sent_sequences
            rev_sequences += rev_sent_sequences

    assert len(sequences) == len(rev_sequences)
    print('Total Sequences: %d' % len(sequences))
    
    # Here we separate the previous encoded words (X) and the target
    # word we will try to predict (y), also turning them into arrays:
    X = np.array([[0]*(n-len(seq)) + seq[:-1] for seq in sequences])
    y = np.array([[seq[-1]] for seq in sequences])
    X_rev = np.array([[0]*(n-len(seq)) + seq[:-1] for seq in rev_sequences])
    y_rev = np.array([[seq[-1]] for seq in rev_sequences])
    
    assert len(X) == len(y)
    
    return X, y, X_rev, y_rev