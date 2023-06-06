# GPT2-Poem-Small

## Model Description

The model, based on GPT2, is used to generate Chinese ancient poems and couplets. You can try the model in [GPT2-Poem-Small](https://huggingface.co/snzhang/GPT2-Poem-Small)

## Download script and prepare the environment

Run this to download the script.

```markdown
git clone https://github.com/h7nian/GPT2-Poem-Small.git
cd GPT2-Poem-Small
pip install -r requirements.txt 
```

## Training Data

The data contains 71334 rows which include 30000 Chinese Tang Poems, 20000 Chinese couplets and 21334 Chinese Song Poems. Tang and song poems is from [Chinese Poetry](https://github.com/chinese-poetry/chinese-poetry) and couplets is from [Couplet Dataset](https://github.com/wb14123/couplet-dataset)

## Finetune

You can run this to finetune the model.

```markdown
deepspeed --num_gpus=1 train.py \
--deepspeed ds_config.json \
--model_name_or_path snzhang/GPT2-Poem-Small \
--train_file data.csv \
--do_train \
--fp16 \
--overwrite_cache \
--evaluation_strategy="steps" \
--output_dir finetuned \
--num_train_epochs 1 \
--gradient_accumulation_steps 2 \
--per_device_train_batch_size 8
```

- You can replace the `data.csv` with your own data
- There are more parameters you can add to it. More details in [HuggingFace Deepspeed Intergration](https://huggingface.co/docs/transformers/main_classes/deepspeed)

## Generate the text with your own model

```markdown
python generation.py --model_type=gpt2 --model_name_or_path=finetuned --length 200
```
