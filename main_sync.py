import time

start_time = time.time()


import os
os.environ['TRANSFORMERS_CACHE'] = '/scratch/nhj4247/python_cache/'

## Custom model training

from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import trange, tqdm
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler, DistributedSampler
import torch
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
cargs = parser.parse_args()



args = TrainingArguments(
    output_dir="./mlm_roberta_hate",
    overwrite_output_dir=True,
    num_train_epochs=20,
    per_device_train_batch_size=32,
    save_steps=500,
    save_total_limit=2,
    seed=1,
    eval_steps=200
)

# For Disitributed training

if cargs.local_rank == -1:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
else:
    torch.cuda.set_device(cargs.local_rank)
    device = torch.device("cuda", cargs.local_rank)
    #args.n_gpu = 1
    
if cargs.local_rank not in [-1, 0]:
    torch.distributed.barrier()
    
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForMaskedLM.from_pretrained('roberta-base')

train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="/scratch/nhj4247/data/tweeteval/datasets/hate/train_text.txt",
    block_size=512,
)

eval_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="/scratch/nhj4247/data/tweeteval/datasets/hate/val_text.txt",
    block_size=512,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

if cargs.local_rank == 0:
    torch.distributed.barrier()



# trainer = Trainer(
#     model=model,
#     args=args,
#     data_collator=data_collator,
#     train_dataset=dataset
# )


def evaluate(args, model, eval_dataset):
    
    model.eval()
    eval_sampler = SequentialSampler(eval_dataset) if cargs.local_rank in [-1,0] else DistributedSampler(eval_dataset, rank=cargs.local_rank)
    eval_dataloader = DataLoader(eval_dataset,
                                 batch_size=args.per_device_eval_batch_size,
                                 sampler=SequentialSampler(eval_dataset),
                                 collate_fn=data_collator)
    eval_iterator = tqdm(eval_dataloader, desc="Evaluating", position=0, leave=True)
    eval_loss = 0
    for _, batch in enumerate(eval_iterator):
        batch.to(device)
        model.eval()
        outputs = model(**batch)
        if args.n_gpu > 1:
            tmp_loss = outputs['loss'].mean()
        else:
            tmp_loss = outputs['loss']
        eval_loss += tmp_loss.item()
        
    return eval_loss


## Training loop

print("Number of availabel GPUs: %d" %args.n_gpu)
print("Process rank %d" %cargs.local_rank)

train_batch_size = args.n_gpu * args.per_device_train_batch_size

train_sampler = RandomSampler(train_dataset) if cargs.local_rank == -1 else DistributedSampler(train_dataset, rank=cargs.local_rank)
train_dataloader = DataLoader(train_dataset,
                              batch_size=train_batch_size,
                              sampler=train_sampler,
                              collate_fn=data_collator)
t_total = len(train_dataloader) * args.num_train_epochs

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

model.to(device)

if args.n_gpu > 1:
    model = torch.nn.DataParallel(model)
    

if cargs.local_rank != -1:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cargs.local_rank],
                                                      output_device=cargs.local_rank,
                                                      find_unused_parameters=True)


print("***** Running training *****")
print("  Num examples = %d" %len(train_dataset))
print("  Num Epochs = %d" %args.num_train_epochs)
print("  Instantaneous batch size per GPU = %d" %args.per_device_train_batch_size)
print("  Total optimization steps = %d" %t_total)

train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=cargs.local_rank not in [-1, 0])
global_step = 0

training_dynamics = []

if cargs.local_rank not in [-1, 0]:
    torch.distributed.barrier()
if cargs.local_rank == 0:
    eval_loss = evaluate(args, model.module, eval_dataset)
    print("Eval loss at %d is %.2f" %(global_step, eval_loss))
    print("--- %s seconds ---" % (time.time() - start_time))
    training_dynamics.append([round(time.time() - start_time, 2), round(eval_loss, 2)])
    torch.distributed.barrier()


for epoch in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", position=0, leave=True, disable=cargs.local_rank not in [-1, 0])
    for step, batch in enumerate(epoch_iterator):
        batch.to(device)
        model.train()
        outputs = model(**batch)
        loss = outputs['loss']
        
        if args.n_gpu > 1:
            loss = loss.mean()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        global_step += 1
        
        if cargs.local_rank in [-1, 0] and global_step % args.save_steps == 0:
            output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                
        if global_step % args.eval_steps == 0:
            if cargs.local_rank not in [-1, 0]:
                torch.distributed.barrier()
            if cargs.local_rank == 0:
                eval_loss = evaluate(args, model.module, eval_dataset)
                print("Eval loss at %d is %.2f" %(global_step, eval_loss))
                print("--- %s seconds ---" % (time.time() - start_time))
                training_dynamics.append([round(time.time() - start_time, 2), round(eval_loss, 2)])
                torch.distributed.barrier()

if cargs.local_rank == 0:
    print("Training Dynamics", training_dynamics)


final_loss = evaluate(args, model, eval_dataset)
print("Final eval loss: %.2f" %final_loss)
print("--- %s seconds ---" % (time.time() - start_time))
            
            

                                 


