import torch
from datasets import load_dataset
from transformers import default_data_collator
from datasets import concatenate_datasets


def _preprocess(tokenizer, examples, max_length=128):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=max_length
    )


def get_commonsense(split):
    #dataset = load_dataset('./dataset/commonsense')[split]
    dataset = load_dataset("tau/commonsense_qa")[split]
    def merge(sample):
        answer_idx = sample['choices']['label'].index(sample['answerKey'])
        sample['text'] = sample['question'] + sample['choices']['text'][answer_idx]
        return sample
    dataset = dataset.map(merge).remove_columns(["id", "question", "question_concept", "choices", "answerKey"])
    return dataset

def get_dataset(dataset_name, subset, split, size=None, start=0, seed=42):
    if dataset_name == 'alpaca':
        #dataset = load_dataset('parquet',data_files='./dataset/tatsu-lab-alpaca/train-00000-of-00001.parquet')[split]
        dataset = load_dataset("tatsu-lab/alpaca")[split]
    elif dataset_name == 'wikitext':
        #dataset = load_dataset('./dataset/wikitext-2-raw-v1')[split]
        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")[split]
    elif dataset_name == 'commonsense':
        dataset = get_commonsense(split)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name!r}")
    dataset = dataset.shuffle(seed)
    if size is not None:
        dataset = dataset.skip(start).take(size)

    return dataset

def get_dataloader(dataset, tokenizer, batch_size, num_workers=4, max_length=128):
    dataset = dataset.map(
        lambda examples: _preprocess(tokenizer, examples, max_length),
        batched=True,
        batch_size=batch_size,
        remove_columns=["text"], #, "timestamp", "url"],
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=default_data_collator,
    )
    return dataloader
