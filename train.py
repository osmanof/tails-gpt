import torch
from model import GPT
from data import prepare_data
from config import DEVICE, BLOCK_SIZE, BATCH_SIZE, LEARNING_RATE, MAX_ITERS, EVAL_INTERVAL, EVAL_ITERS, MODEL_PATH


def main():
    train_data, val_data, _ = prepare_data()
    
    model = GPT().to(DEVICE)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)


    def get_batch(split):
        data_source = train_data if split == 'train' else val_data
        ix = torch.randint(len(data_source) - BLOCK_SIZE, (BATCH_SIZE,))
        x = torch.stack([data_source[i:i+BLOCK_SIZE] for i in ix])
        y = torch.stack([data_source[i+1:i+BLOCK_SIZE+1] for i in ix])
        return x.to(DEVICE), y.to(DEVICE)


    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(EVAL_ITERS)
            for k in range(EVAL_ITERS):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    print("start")
    
    for iter in range(MAX_ITERS):
        if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("finish")
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nmodel is saved '{MODEL_PATH}'")


if __name__ == "__main__":
    main()
