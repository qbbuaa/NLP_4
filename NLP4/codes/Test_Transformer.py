import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("my_model")
model = GPT2LMHeadModel.from_pretrained("my_model")
model = model.to('cuda')
input_text = "赵处女"
input_ids = tokenizer.encode(input_text)
input_ids = torch.Tensor(input_ids).long().cuda().unsqueeze(0)

generated_text = model.generate(
    input_ids,
    max_length=100,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.8,
)
generated_text = tokenizer.decode(generated_text[0], skip_special_tokens=True,encodings= 'UTF-8')
print(generated_text)