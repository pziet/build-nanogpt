import torch
import torch.nn.functional as F
from gpt_model import GPT
from tiktoken.core import Encoding

def generate_json(
    prompt: str, 
    schema: dict, 
    model: GPT, 
    enc: Encoding, 
    device: str, 
    numeric_tokens: torch.Tensor, 
    period_token: int, 
    comma_token: int, 
    max_length=256,
    seed=42,
    ) -> str:
    def generate_value(value_type, end_token='"'):
        nonlocal xgen
        num_tok = [] # keep track of number that is built
        while True:
            current_xgen = xgen[:, -2*max_length:]
            logits, _ = model(current_xgen)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            if value_type == "number":
                filtered_probs = probs[:, numeric_tokens[0,:]]
                filtered_probs = filtered_probs / filtered_probs.sum(dim=1).unsqueeze(1)  # Renormalize
                ix = torch.multinomial(filtered_probs, 1, generator=sample_rng) # indices
                xcol = torch.gather(numeric_tokens, dim=-1, index=ix)
                # Either it is an integer, eg 87, or it is a number with a decimal point, eg 87.5
                if (len(num_tok) == 0 and 
                    (xcol.item() == period_token or xcol.item() == comma_token)): 
                    # Can't have period or comma as the first token
                    continue
                elif (len(num_tok) == 1 and 
                      (xcol.item() != period_token or xcol.item() != comma_token)):
                    # If the first token is a number, the second token must be a period otherwise it is an integer.
                    xcol[0] = comma_token
                elif (len(num_tok) == 2 and 
                      xcol.item() == period_token and period_token in num_tok): 
                    # Can't have two periods in a number
                    continue
                elif len(num_tok) == 2 and xcol.item() == comma_token:
                    # Can't have a comma after a period
                    continue
                elif len(num_tok) == 3:
                    xcol[0] = comma_token

                num_tok.append(xcol.item())
            else:
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                xcol = torch.gather(topk_indices, -1, ix)
                # check if token contains end_token but doesn't end with it
                temp_token_split = list(enc.decode([xcol.item()]))
                if end_token in temp_token_split and temp_token_split[-1] != end_token:
                    continue
            
            new_token = enc.decode([xcol.item()])
            xgen = torch.cat((xgen, xcol), dim=1)
            if value_type == "number":
                if new_token == end_token:
                    # remove the end token
                    xgen = xgen[:, :-1]
                    return
            elif value_type == "string":
                new_token_split = list(new_token)
                if len(new_token_split) > 0 and end_token == new_token_split[-1]:
                    return

    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    xgen = tokens.to(device)

    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(seed)

    with torch.no_grad():
        xgen = torch.cat((xgen, torch.tensor([[enc.encode("{")[0]]], device=device)), dim=1)
        
        for i, (key, value) in enumerate(schema.items()):
            if i > 0 and i < len(schema):
                xgen = torch.cat((xgen, torch.tensor([[enc.encode(", ")[0]]], device=device)), dim=1)
            
            key_tokens = enc.encode(f'"{key}": ')
            xgen = torch.cat((xgen, torch.tensor([key_tokens], device=device)), dim=1)
            
            if value["type"] == "string":
                xgen = torch.cat((xgen, torch.tensor([[enc.encode('"')[0]]], device=device)), dim=1)
                generate_value("string", end_token='"')
            elif value["type"] == "number":
                generate_value("number", end_token=",")
        
        xgen = torch.cat((xgen, torch.tensor([[enc.encode("}")[0]]], device=device)), dim=1)

    tokens = xgen[0].tolist()
    decoded = enc.decode(tokens)
    return decoded