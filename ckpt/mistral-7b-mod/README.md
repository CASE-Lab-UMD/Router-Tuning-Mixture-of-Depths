---
license: apache-2.0

extra_gated_description: If you want to learn more about how we process your personal data, please read our <a href="https://mistral.ai/fr/terms/">Privacy Policy</a>.

---

# Model Card for Mistral-7B-v0.3

The Mistral-7B-v0.3 Large Language Model (LLM) is a Mistral-7B-v0.2 with extended vocabulary.

Mistral-7B-v0.3 has the following changes compared to [Mistral-7B-v0.2](https://huggingface.co/mistralai/Mistral-7B-v0.2/edit/main/README.md)
- Extended vocabulary to 32768

## Installation

It is recommended to use `mistralai/Mistral-7B-v0.3` with [mistral-inference](https://github.com/mistralai/mistral-inference). For HF transformers code snippets, please keep scrolling.

```
pip install mistral_inference
```

## Download

```py
from huggingface_hub import snapshot_download
from pathlib import Path

mistral_models_path = Path.home().joinpath('mistral_models', '7B-v0.3')
mistral_models_path.mkdir(parents=True, exist_ok=True)

snapshot_download(repo_id="mistralai/Mistral-7B-v0.3", allow_patterns=["params.json", "consolidated.safetensors", "tokenizer.model.v3"], local_dir=mistral_models_path)
```

### Demo

After installing `mistral_inference`, a `mistral-demo` CLI command should be available in your environment.

```
mistral-demo $HOME/mistral_models/7B-v0.3
```

Should give something along the following lines:

```
This is a test of the emergency broadcast system. This is only a test.

If this were a real emergency, you would be told what to do.

This is a test
=====================
This is another test of the new blogging software. I’m not sure if I’m going to keep it or not. I’m not sure if I’m going to keep
=====================
This is a third test, mistral AI is very good at testing. 🙂

This is a third test, mistral AI is very good at testing. 🙂

This
=====================
```

## Generate with `transformers`

If you want to use Hugging Face `transformers` to generate text, you can do something like this.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistralai/Mistral-7B-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id)
inputs = tokenizer("Hello my name is", return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Limitations

The Mistral 7B Instruct model is a quick demonstration that the base model can be easily fine-tuned to achieve compelling performance. 
It does not have any moderation mechanisms. We're looking forward to engaging with the community on ways to
make the model finely respect guardrails, allowing for deployment in environments requiring moderated outputs.

## The Mistral AI Team

Albert Jiang, Alexandre Sablayrolles, Alexis Tacnet, Antoine Roux, Arthur Mensch, Audrey Herblin-Stoop, Baptiste Bout, Baudouin de Monicault, Blanche Savary, Bam4d, Caroline Feldman, Devendra Singh Chaplot, Diego de las Casas, Eleonore Arcelin, Emma Bou Hanna, Etienne Metzger, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Harizo Rajaona, Jean-Malo Delignon, Jia Li, Justus Murke, Louis Martin, Louis Ternon, Lucile Saulnier, Lélio Renard Lavaud, Margaret Jennings, Marie Pellat, Marie Torelli, Marie-Anne Lachaux, Nicolas Schuhl, Patrick von Platen, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Thibaut Lavril, Timothée Lacroix, Théophile Gervet, Thomas Wang, Valera Nemychnikova, William El Sayed, William Marshall