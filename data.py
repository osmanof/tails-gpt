import torch
import urllib.request
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from config import URL, FILE_PATH, TEMP_CORPUS, VOCAB_SIZE


def prepare_data():
    urllib.request.urlretrieve(URL, FILE_PATH)

    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        text = f.read()

    start_idx = text.find("THE GOLDEN BIRD")
    end_idx = text.find("END OF THE PROJECT GUTENBERG EBOOK")
    if start_idx != -1 and end_idx != -1:
        text = text[start_idx:end_idx]

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"], vocab_size=VOCAB_SIZE)

    with open(TEMP_CORPUS, "w", encoding="utf-8") as f:
        f.write(text)

    tokenizer.train(files=[TEMP_CORPUS], trainer=trainer)

    encoded = tokenizer.encode(text)
    data = torch.tensor(encoded.ids, dtype=torch.long)

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data, tokenizer


if __name__ == "__main__":
    train_data, val_data, tokenizer = prepare_data()
    print(f"vocabulary: {tokenizer.get_vocab_size()} токенов.")
    print(f"train dataset: {len(train_data)} токенов.")
