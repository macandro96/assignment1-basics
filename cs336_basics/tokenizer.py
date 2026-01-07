import regex as re
import multiprocessing as mp
from cs336_basics.train_bpe import train_bpe
import math

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            
        if special_tokens is None or "<|endoftext|>" not in special_tokens:
            special_tokens = special_tokens or []
            special_tokens.append("<|endoftext|>")
        
        self.special_tokens = sorted(special_tokens, reverse=True)
        
        self.byte_to_id, self.merge_dict, self.PAT, self.byteidx2vocab = Tokenizer.process_artifacts(
            vocab,
            merges,
            self.PAT,
            self.special_tokens,
        )
        self.num_workers = 4
            
    @staticmethod
    def process_artifacts(
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        PAT: str,
        special_tokens=None
    ):
        byte_to_id = {v: k for k, v in vocab.items()}
        merge_dict = {
            (byte_to_id[a], byte_to_id[b]): i for i, (a, b) in enumerate(merges)
        }
        idx2bytes = {
            idx: bytes([idx]) for idx in range(256)
        }
        byteidx2vocab = {
            idx: byte_to_id[val] for idx, val in idx2bytes.items()
        }
        if special_tokens:
            escaped_tokens = "|".join([re.escape(tok) for tok in special_tokens])
        PAT = rf"(?:{escaped_tokens})|'(?:[sdmt]|ll|ve|re)| ?\p{{L}}+| ?\p{{N}}+| ?[^\s\p{{L}}\p{{N}}<>]+|\s+(?!\S)|\s+"
        return byte_to_id, merge_dict, PAT, byteidx2vocab
    
    def from_files(self, vocab_file_path: str, merges_filepath: str, special_tokens=None):
        # read merges
        pass
    
    def encode(self, text: str) -> list[int]:
        # pretokenize
        unique_pretokens = set()
        for match in re.finditer(self.PAT, text):
            token = match.group()
            # exclude speical tokens now
            if token not in self.special_tokens:
                unique_pretokens.add(
                    tuple(
                        [self.byteidx2vocab[byte_id] for byte_id in tuple(token.encode("utf-8"))]
                    )
                )

        unique_pretokens = list(unique_pretokens)
        batched_pretokens = []
        batch_size = math.ceil(len(unique_pretokens) / self.num_workers)

        # container to store pretoken to merged mapping
        pretokens2merged = {}
        if len(unique_pretokens) > 0:
            for idx in range(0, len(unique_pretokens), batch_size):
                batched_pretokens.append(
                    unique_pretokens[idx: idx + batch_size]
                )
            # multiprocessing of merging of pretokens
            with mp.Pool() as pool:
                bound_funct = self.merge_list_pretokens
                results = pool.map(
                    bound_funct, batched_pretokens
                )
            
            for result in results:
                for key, value in result.items():
                    pretokens2merged[key] = value

        # encode the text
        encoding = []
        for match in re.finditer(self.PAT, text):
            token = match.group()
            if token in self.special_tokens:
                encoding.append(self.byte_to_id[token.encode("utf-8")])
            else:
                pretoken_bytes = [self.byteidx2vocab[byte_id] for byte_id in tuple(token.encode("utf-8"))]
                merged_bytes = pretokens2merged[tuple(pretoken_bytes)]
                encoding.extend(merged_bytes)
        return encoding
    
    def merge_pretokens(self, pretoken_bytes: list[int]) -> list[int]:
        encodings = pretoken_bytes
        
        while True:
            min_merge_idx = float('inf')
            merge_keys = None
            for k1, k2 in zip(encodings[:-1], encodings[1:]):
                if (k1, k2) in self.merge_dict:
                    merge_idx = self.merge_dict[(k1, k2)]
                    if min_merge_idx > merge_idx:
                        merge_keys = (k1, k2)
                        min_merge_idx = merge_idx
            
            # no more merges
            if merge_keys is None:
                return encodings
            
            # merge
            p1, p2 = merge_keys
            idx = 0
            new_encodings = []
            # get id of new merge
            new_id = self.byte_to_id[self.vocab[p1] + self.vocab[p2]]
            
            while idx < len(encodings):
                if idx + 1 < len(encodings) and (encodings[idx], encodings[idx+1]) == (p1, p2):
                    new_encodings.append(new_id)
                    idx += 2
                else:
                    new_encodings.append(encodings[idx])
                    idx += 1

            encodings = new_encodings
        
    def merge_list_pretokens(self, pretoken_bytes_list: list[list[int]]) -> dict[tuple[int, ...], list[int]]:
        merged_dict = {}
        for pretoken_bytes in pretoken_bytes_list:
            merged_dict[tuple(pretoken_bytes)] = self.merge_pretokens(pretoken_bytes)
        return merged_dict
    
    def decode(self, encoding: list[int]) -> str:
        bytes_list = []
        for idx in encoding:
            bytes_list.append(self.vocab[idx])
        decoded_bytes = b"".join(bytes_list)

        return decoded_bytes.decode("utf-8", errors="replace")
    
    def encode_iterable(self, iterable):
        # iterable could be a file pointer
        # read 4kb at a time
        end_of_token = "<|endoftext|>"
        chunk_size = 1024 * 1024 * 512  # 512MB
        chunk_read = 0
        while True:
            chunk = iterable.read(chunk_size)
            if chunk == "":
                break
            found_at = chunk.rfind(end_of_token)
            if found_at != -1:
                chunk = chunk[:found_at + len(end_of_token)]
                iterable.seek(chunk_read + len(chunk.encode("utf-8")))
            chunk_read += len(chunk.encode("utf-8"))
            yield from self.encode(chunk)               
        
    
if __name__ == "__main__":
    vocab_dict, merges = train_bpe(
        input_path="../data/temp.txt",
        vocab_size=50,
        special_tokens=[],
    )
    tokenizer = Tokenizer(vocab_dict, merges)
    sample_text = "Hello, world! This is a test."
    encoding = tokenizer.encode(sample_text)
    print("Encoded output:", encoding)