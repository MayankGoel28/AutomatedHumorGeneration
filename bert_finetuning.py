#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datasets import Dataset
from transformers import Trainer
from transformers import AutoModelForMaskedLM, BertForMaskedLM
from transformers import AutoTokenizer, BertTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd

with open('train.txt', 'r') as f:
    lines = f.readlines()
    
print("dataset loaded")


# In[ ]:


model_checkpoint = "bert-base-uncased"
model = BertForMaskedLM.from_pretrained(model_checkpoint)
tokenizer = BertTokenizer.from_pretrained(model_checkpoint, do_lower_case=False)


# In[ ]:


SEED = 42
BATCH_SIZE = 8
LEARNING_RATE = 2e-5 
LR_WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01


# ### Dataset prep

# In[ ]:





# In[ ]:


# comment this out
# lines = lines[:10]
    
df = pd.DataFrame({"text": [line for line in lines]})
df_train, df_valid = train_test_split(
    df, test_size=0.15, random_state=SEED
)

train_dataset = Dataset.from_pandas(df_train[['text']].dropna())
valid_dataset = Dataset.from_pandas(df_valid[['text']].dropna())


# In[ ]:


def tokenize_function(row):
    return tokenizer(
        row['text'],
        padding='max_length',
        return_special_tokens_mask=True)
  
column_names = train_dataset.column_names

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=column_names,
)

valid_dataset = valid_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=column_names,
)


# In[ ]:


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)
steps_per_epoch = int(len(train_dataset) / BATCH_SIZE)

training_args = TrainingArguments(
    output_dir='./bert-joke',
    logging_dir='./LMlogs',             
    num_train_epochs=2,
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=LR_WARMUP_STEPS,
    # save_steps=steps_per_epoch,
    weight_decay=WEIGHT_DECAY,
    learning_rate=LEARNING_RATE, 
    evaluation_strategy='steps',
    save_strategy='steps',
    eval_steps=5000,
    save_steps=5000,
    load_best_model_at_end=True,
    metric_for_best_model='loss', 
    greater_is_better=False,
    seed=SEED,
    disable_tqdm=False,
    save_total_limit=5,
    gradient_accumulation_steps=2
)


# In[ ]:


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    
)

trainer.train()
print("model trained")
trainer.save_model(".")
print("model saved")

