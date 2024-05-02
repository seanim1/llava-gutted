# Gutted Llava

**DISCLAIMER: Code taken from [the official Llava Repo](https://github.com/haotian-liu/LLaVA)**.

This is a gutted llava repo to quantize the llama portion and finetune it.

You need to download the TextVQA [images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip) and extract to `playground/data/eval/textvqa/train_images`.

To run:
```
python -m llava.serve.cli \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-file "https://llava-vl.github.io/static/images/view.jpg" \
    --load-4bit
```

To test VQA:
```
python3 -m llava.eval.model_vqa_loader \
  --model-path liuhaotian/llava-v1.5-7b \
  --question-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
  --image-folder ./playground/data/eval/textvqa/train_images \
  --answers-file ./playground/data/eval/textvqa/answers/llava-v1.5-7b_subset100.jsonl \
  --temperature 0 \
  --load-4bit
```
