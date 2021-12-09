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
import torch.multiprocessing as mp
import argparse


def evaluate(args, model, eval_dataset, data_collator, device, n_gpu):
    
    model.to(device)
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
        if n_gpu > 1:
            tmp_loss = outputs['loss'].mean()
        else:
            tmp_loss = outputs['loss']
        eval_loss += tmp_loss.item()
        
    return eval_loss


## Training loop

def train(args, model, device, train_dataloader, eval_dataset, data_collator, rank, n_gpu):

    print("Number of availabel GPUs: %d" %n_gpu)
    training_dynamics = []
    torch.cuda.set_device(rank)

    train_batch_size = n_gpu * args.per_device_train_batch_size

    t_total = len(train_dataloader) * args.num_train_epochs

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)


    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    print("Local rank is: %d" %args.local_rank)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                      output_device=args.local_rank,
                                                      find_unused_parameters=True)


    print("***** Running training *****")
    print("  Num examples = %d" %len(train_dataloader))
    print("  Num Epochs = %d" %args.num_train_epochs)
    print("  Instantaneous batch size per GPU = %d" %args.per_device_train_batch_size)
    print("  Total optimization steps = %d" %t_total)

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    global_step = 0
    model.to(device)

    eval_loss = evaluate(args, model, eval_dataset, data_collator, device, n_gpu)
    print("Loss at %s is %.2f" %(time.time() - start_time, eval_loss))
    training_dynamics.append([round(time.time() - start_time, 2), round(eval_loss,2)])

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", position=0, leave=True)
        for step, batch in enumerate(epoch_iterator):
            batch.to(device)
            model.train()
            outputs = model(**batch)
            loss = outputs['loss']

            if n_gpu > 1:
                loss = loss.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1

            if global_step % args.save_steps == 0:
                output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    model.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))

            if global_step % args.eval_steps == 0:
                eval_loss = evaluate(args, model, eval_dataset, data_collator, device, n_gpu)
                print("Eval loss at %d is %.2f" %(global_step, eval_loss))
                print("--- %s seconds ---" % (time.time() - start_time))
                training_dynamics.append([round(time.time() - start_time, 2), round(eval_loss,2)])
    
    print("Training dynamics", training_dynamics) 



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
            description="Run asynchronous training"
    )
    parser.add_argument("--n_procs", type=int, default=1, help="Number of GPUs/asynchronous processes")
    cargs = parser.parse_args()
    
    mp.set_start_method('spawn')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
    n_gpu = 1   
    

    ## Asynch distributed training
    processes = []
    procs = cargs.n_procs
    for rank in range(procs):
        
        model.share_memory()
        train_batch_size = args.per_device_train_batch_size
        train_sampler = DistributedSampler(dataset=train_dataset,
                                           num_replicas=procs,
                                           rank=rank)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=train_batch_size,
                                      sampler=train_sampler,
                                      collate_fn=data_collator)
        
        p = mp.Process(target=train, args=(args, model, device, train_dataloader, eval_dataset, data_collator, rank, n_gpu))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
     
    final_loss = evaluate(args, model, eval_dataset, data_collator, device, n_gpu)
    
    print("Final eval loss: %.2f" %final_loss)
    print("--- %s seconds ---" % (time.time() - start_time))
    
        
        


