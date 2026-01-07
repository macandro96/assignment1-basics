from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re
from collections import defaultdict
import multiprocessing
import argparse
import time
from tqdm import tqdm
import heapq
    
def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
):
    num_processes = 8
    
    # add <|endoftext|> to special tokens if not already present
    if "<|endoftext|>" not in special_tokens:
        special_tokens.append("<|endoftext|>")
    
    # incorporating special tokens into regex pattern
    escaped_tokens = [re.escape(token) for token in special_tokens]
    special_tokens_pattern = "|".join(escaped_tokens)
    PAT = rf"(?:{special_tokens_pattern})|'(?:[sdmt]|ll|ve|re)| ?\p{{L}}+| ?\p{{N}}+| ?[^\s\p{{L}}\p{{N}}<>]+|\s+(?!\S)|\s+"
    
    print("Finding chunk boundaries ....\n")
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    
    special_tokens_dictionary = {
        token: 256 + idx
        for idx, token in enumerate(special_tokens)
    }
    print("Starting pretokenization....\n")
    with multiprocessing.Pool() as pool:
        results = pool.starmap(
            pretokenize, [
                (input_path, start, end, PAT, special_tokens_dictionary)
                for start, end in zip(boundaries[:-1], boundaries[1:])
            ]
        )
    
    pretokenized_data = defaultdict(int)
    for result in results:
        for key, value in result.items():
            pretokenized_data[key] += value
    vocab_dict = {i: bytes([i]) for i in range(256)}
    for token, idx in special_tokens_dictionary.items():
        vocab_dict[idx] = token.encode("utf-8")

    print("Starting BPE merges ....\n")
    merges = merge(pretokenized_data, vocab_size, vocab_dict)

    print("BPE training completed.\n")
    return vocab_dict, merges
    

def merge(pretokenized_data: dict, vocab_size: int, vocab_dict: dict):
    # calculate pair frequencies
    pair_frequencies = defaultdict(int)
    
    for key, value in pretokenized_data.items():
        for p1, p2 in zip(key[:-1], key[1:]):
            pair_frequencies[(p1, p2)] += value

    heap = [
        (
            -value,
            convert_byte_to_negtuple(vocab_dict[pair[0]], vocab_dict[pair[1]]),
            pair
        )
        for pair, value in pair_frequencies.items()
    ]
    heapq.heapify(heap)
    
    merges = []
    for _ in tqdm(range(vocab_size - len(vocab_dict))):
        # find the most frequent pair
        if not pair_frequencies:
            break
        
        most_frequent_pair = None
        while heap:
            freq, _, pair = heapq.heappop(heap)
            
            # freq, v, pair = item.val, item.pair, item.other
            if pair_frequencies[pair] == -freq:
                most_frequent_pair = pair
                break

        p1, p2 = most_frequent_pair
        new_token_bytes = vocab_dict[p1] + vocab_dict[p2]
        
        new_id = len(vocab_dict)
        vocab_dict[new_id] = new_token_bytes
        merges.append((vocab_dict[p1], vocab_dict[p2]))
        
        pair_frequencies.pop(most_frequent_pair)
        
        update_dict = defaultdict(int)
        # merge
        new_pretokenized_data = defaultdict(int)
        for key, value in pretokenized_data.items():
            new_key = []
            idx = 0
            while idx < len(key):
                if idx + 1 < len(key) and key[idx] == p1 and key[idx + 1] == p2:
                    new_key.append(new_id)
                    # update pair frequencies
                    # ..ch1 p1 p2 ch2 ...
                    # decrement (ch1, p1) and (p2, ch2)
                    if idx - 1 >= 0:
                        ch1 = key[idx - 1]
                        update_dict[(ch1, p1)] -= value
                        update_dict[(ch1, new_id)] += value
                    
                    if idx + 2 < len(key):
                        ch2 = key[idx + 2]
                        update_dict[(p2, ch2)] -= value
                        update_dict[(new_id, ch2)] += value

                    idx += 2  # skip next since it's merged
                else:
                    new_key.append(key[idx])
                    idx += 1

            new_key = tuple(new_key)
            new_pretokenized_data[new_key] += value

        pretokenized_data = new_pretokenized_data
        
        # update pair_frequencies
        for key, value in update_dict.items():
            pair_frequencies[key] += value
            heapq.heappush(
                heap,
                # MaxTuple(-pair_frequencies[key], (vocab_dict[key[0]], vocab_dict[key[1]]), key)
                (
                    -pair_frequencies[key],
                    convert_byte_to_negtuple(vocab_dict[key[0]], vocab_dict[key[1]]),
                    key
                )
            )
    return merges

def convert_byte_to_negtuple(byte1, byte2):
    byte_tuple1 = tuple(byte1)
    byte_tuple2 = tuple(byte2)
    return tuple([-b for b in byte_tuple1]) + (1e6,), tuple([-b for b in byte_tuple2]) + (1e6,)

def pretokenize(input_path, start, end, PAT, special_tokens_dictionary):
    pattern = re.escape("<|endoftext|>")
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        
        docs = re.split(pattern, chunk)
        
        pretokenize_dict = defaultdict(int)
        for doc in docs:
            pretok_dict_doc = pretokenize_doc(doc, PAT, special_tokens_dictionary)
            for key, value in pretok_dict_doc.items():
                pretokenize_dict[key] += value
    return pretokenize_dict

def pretokenize_doc(chunk: str, PAT: str, special_tokens_dictionary: dict):
    pretokenize_dict = defaultdict(int)

    for match in re.finditer(PAT, chunk):
        tokenized_word = match.group()
        if tokenized_word in special_tokens_dictionary:
            bytes_word = (special_tokens_dictionary[tokenized_word],)
        else:
            # convert tokenized word into sequence o bytes
            bytes_word = tuple(tokenized_word.encode("utf-8"))
        # storing counts of each bytes_word
        pretokenize_dict[bytes_word] += 1
    
    return pretokenize_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help='path to input text file')
    parser.add_argument("--vocab_size", type=int, default=500, help='vocab size')
    parser.add_argument("--special_tokens", type=str, nargs='*', default=["<|endoftext|>"], help='list of special tokens')
    
    args = parser.parse_args()
    
    start = time.time()
    vocab_dict, merges = train_bpe(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens
    )
    end = time.time()
    print(f"Trained BPE with vocab size {args.vocab_size} in {end - start:.2f} seconds")