import torch
from model import GPT
from data import prepare_data
from config import DEVICE, MODEL_PATH


def main():
    _, _, tokenizer = prepare_data()
    
    model = GPT().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    start_text = "Once upon a time, in a dark forest, there lived a"
    encoded_prompt = tokenizer.encode(start_text).ids
    idx = torch.tensor(encoded_prompt, dtype=torch.long, device=DEVICE).unsqueeze(0)

    print("generating...\n" + "-"*50)

    with torch.no_grad():
        generated_ids = model.generate(idx, max_new_tokens=350)

    output_ids = generated_ids[0].tolist()
    raw_text = tokenizer.decode(output_ids)

    clean_text = raw_text.replace('Ġ', ' ').replace('Ċ', '\n')

    print(clean_text)
    print("-" * 50)


if __name__ == "__main__":
    main()
