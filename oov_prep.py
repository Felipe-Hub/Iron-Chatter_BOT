import numpy as np

def n_grams(enc_sent, n=4):
    """
    This function takes a sequence of sentences and returns
    a tuple with four arrays. The first array contains padded
    n-grams combinations (X). The second array is the target (y).
    
    The third and fourth arrays are the X and y for the 
    reversed sequences (X_rev and y_rev).
    
    Both returned sequences are already pre-padded with 0.
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


def embedded_n_grams(embedded_sent, n=4, x=100):
    """
    This function takes a sequence of embedded sentences and
    returns a tuple with four arrays. The first array contains
    vectors with padded n-grams combinations (X). The second
    array is the target embedded word (y).
    
    The third and fourth arrays are the X and y for the 
    reversed sequences (X_rev and y_rev).
    
    Both returned sequences are already pre-padded with 0.
    The maximum length of the sequences is n-1.
    The param 'x' is the size of the vector set for the embedding.
    """  
    sequences = list()
    rev_sequences = list()

    for seq in embedded_sent:

        for i in range(len(seq)):
            sent_sequences = list()
            rev_sent_sequences = list()

            for ii in range(n+1):
                sequence = seq[i:i+ii]
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
    X = np.array([[[0]*x]*(n-len(seq)) + seq[:-1] for seq in sequences])
    y = np.array([[seq[-1]] for seq in sequences])
    X_rev = np.array([[[0]*x]*(n-len(seq)) + seq[:-1] for seq in rev_sequences])
    y_rev = np.array([[seq[-1]] for seq in rev_sequences])

    assert len(X) == len(y)

    return X, y, X_rev, y_rev


def tagged_n_grams(tagged_sent, n=4):
    """
    This function takes a sequence of tupled sentences and tags:
    [('Here is an example', 'tag1 tag2 tag3 tag4')]
    Returns a tuple with other two tuples inside. The first tuple
    is for the encoded texts and the second for the encoded tags.
    Each tuple contains four arrays: the first array contains
    padded n-grams combinations(X). The second array is the target(y).
    
    The third and fourth arrays are the X and y for the reversed
    sequences (X_rev and y_rev).
    
    All returned sequences are already pre-padded with 0.
    The maximum length of the sequences is n-1, with n being the
    desired number of n-grams.
    """  
    text_sequences = list()
    text_rev_sequences = list()
    tags_sequences = list()
    tags_rev_sequences = list()
    
    # Here we separate into lists of n-grams:
    for text, tag in tagged_sent:
        
        for i in range(len(text)):
            text_sent_sequences = list()
            text_rev_sent_sequences = list()
            tags_sent_sequences = list()
            tags_rev_sent_sequences = list()
            
            for ii in range(n+1):
                text_sequence = text[i:i+ii]
                text_rev_sequence = text_sequence[::-1]
                tags_sequence = tag[i:i+ii]
                tags_rev_sequence = tags_sequence[::-1]
                
                # avoinding repeated n-grams in the same sentence
                if text_sequence not in text_sent_sequences and len(text_sequence)>1 and len(tags_sequence)>1:
                    text_sent_sequences += [text_sequence]
                    text_rev_sent_sequences += [text_rev_sequence]
                    tags_sent_sequences += [tags_sequence]
                    tags_rev_sent_sequences += [tags_rev_sequence]
                    assert len(text_sent_sequences) == len(tags_sent_sequences)
                    
            text_sequences += text_sent_sequences
            text_rev_sequences += text_rev_sent_sequences
            tags_sequences += tags_sent_sequences
            tags_rev_sequences += tags_rev_sent_sequences

    assert len(text_sequences) == len(tags_sequences) 
    print('Total Sequences: %d' % len(text_sequences))
    
    # Here we separate the previous encoded words (X) and the target
    # word we will try to predict (y), also turning them into arrays:
    X = np.array([[0]*(n-len(seq)) + seq[:-1] for seq in text_sequences])
    y = np.array([[seq[-1]] for seq in text_sequences])
    X_rev = np.array([[0]*(n-len(seq)) + seq[:-1] for seq in text_rev_sequences])
    y_rev = np.array([[seq[-1]] for seq in text_rev_sequences])
    
    X_tag = np.array([[0]*(n-len(seq)) + seq[:-1] for seq in tags_sequences])
    y_tag = np.array([[seq[-1]] for seq in tags_sequences])
    X_tag_rev = np.array([[0]*(n-len(seq)) + seq[:-1] for seq in tags_rev_sequences])
    y_tag_rev = np.array([[seq[-1]] for seq in tags_rev_sequences])
    
    assert len(X) == len(y) == len(X_tag) == len(y_tag)
    
    return ((X, y, X_rev, y_rev),(X_tag, y_tag, X_tag_rev, y_tag_rev))