"""
#GOOGLE COLLAB CODE
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
import os
import pickle
import ast

class ChatDataset(Dataset):
    def __init__(self, inputs, masks, labels):
        self.inputs = inputs
        self.masks = masks
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx],
            'attention_mask': self.masks[idx],
            'labels': self.labels[idx]
        }

def save_tokenized_data(inputs, labels, filename):
    with open(filename, 'wb') as f:
        pickle.dump({'inputs': inputs, 'labels': labels}, f)

def load_tokenized_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def train_model():
    # Mount Google Drive
    from google.colab import drive
    #drive.mount('/content/drive/MyDrive/AI_Chatbot_Project/processed_synthetic_chatbot_data.csv')

    cleaned_file = "/content/drive/MyDrive/AI_Chatbot_Project/processed_synthetic_chatbot_data.csv"
    if not os.path.exists(cleaned_file):
        raise FileNotFoundError(f"Cleaned dataset not found at {cleaned_file}. Ensure file exists.")

    df = pd.read_csv(cleaned_file)
    if not all(col in df.columns for col in ['input', 'output']):
        raise ValueError(f"Dataset must contain 'input' and 'output' columns. Found: {list(df.columns)}")

    # Preprocess the input,output columns (stringified dicts)
    data = []
    for _, row in df.iterrows():
        try:
            input_msg = ast.literal_eval(row['input'])
            output_msg = ast.literal_eval(row['output'])
        except:
            continue

        # Extract 'value' and 'from' fields
        if isinstance(input_msg, dict) and isinstance(output_msg, dict):
            input_text = input_msg.get('value', '').strip()
            input_from = input_msg.get('from', '')
            output_text = output_msg.get('value', '').strip()
            output_from = output_msg.get('from', '')

            # Determine user and bot messages
            if input_from == 'human' and output_from == 'gpt':
                user_text = input_text
                bot_text = output_text
            elif input_from == 'gpt' and output_from == 'human':
                user_text = output_text
                bot_text = input_text
            else:
                continue  # Skip if 'from' fields don't match expected values

            if user_text and bot_text:
                data.append({'input': user_text, 'output': bot_text})

    if not data:
        raise ValueError("No valid input-output pairs extracted from input,output columns.")

    df_processed = pd.DataFrame(data)
    df_processed = df_processed.sample(n=80000, random_state=42)  # Use 50,000 samples
    train_df, val_df = train_test_split(df_processed, test_size=0.2, random_state=42)
    print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")

    # Debug: Print first 3 rows of train_df
    print(f"First 3 rows of train_df:\n{train_df.head(3)}")

    model_name = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"Device: {device}")
    print(f"Model on GPU: {next(model.parameters()).device}")

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        model.resize_token_embeddings(len(tokenizer))

    tokenized_train_file = "/content/drive/MyDrive/AI_Chatbot_Project/tokenized_train.pkl"
    tokenized_val_file = "/content/drive/MyDrive/AI_Chatbot_Project/tokenized_val.pkl"

    # Always retokenize to ensure full dataset is used
    train_inputs = tokenizer(
        list(train_df['input']),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    train_outputs = tokenizer(
        list(train_df['output']),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    train_labels = train_outputs['input_ids'].clone()
    train_labels[train_labels == tokenizer.pad_token_id] = -100
    

    val_inputs = tokenizer(
        list(val_df['input']),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    val_outputs = tokenizer(
        list(val_df['output']),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    val_labels = val_outputs['input_ids'].clone()
    val_labels[val_labels == tokenizer.pad_token_id] = -100

     # Debug: Check tensor sizes
    print(f"Train inputs shape: {train_inputs['input_ids'].shape}")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Val inputs shape: {val_inputs['input_ids'].shape}")
    print(f"Val labels shape: {val_labels.shape}")


    save_tokenized_data(train_inputs, train_labels, tokenized_train_file)
    save_tokenized_data(val_inputs, val_labels, tokenized_val_file)

    assert len(train_inputs['input_ids']) == len(train_labels), "Train input and label sizes mismatch"
    assert len(val_inputs['input_ids']) == len(val_labels), "Validation input and label sizes mismatch"

    train_dataset = ChatDataset(
        train_inputs['input_ids'],
        train_inputs['attention_mask'],
        train_labels
    )
    val_dataset = ChatDataset(
        val_inputs['input_ids'],
        val_inputs['attention_mask'],
        val_labels
    )

    # Debug: Print dataset sizes
    print(f"Size of train_dataset: {len(train_dataset)}")
    print(f"Size of val_dataset: {len(val_dataset)}")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    training_args = TrainingArguments(
        output_dir="/content/drive/MyDrive/AI_Chatbot_Project/results",
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=5000,
        logging_dir="/content/drive/MyDrive/AI_Chatbot_Project/logs",
        logging_strategy="steps",
        logging_steps=500,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=1,
        warmup_steps=200,
        weight_decay=0.01,
        learning_rate=5e-5,
        gradient_accumulation_steps=4,
        dataloader_pin_memory=False,
        dataloader_drop_last=True,
        report_to="none"  # Disable W&B
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )

    trainer.train()
    model.save_pretrained("/content/drive/MyDrive/AI_Chatbot_Project/output")
    tokenizer.save_pretrained("/content/drive/MyDrive/AI_Chatbot_Project/output")
    print("Model and tokenizer saved to /content/drive/MyDrive/AI_Chatbot_Project/output")

if __name__ == "__main__":
    train_model()


#VISUAL STUDIO CODE
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
import os
import pickle
import ast

class ChatDataset(Dataset):
    def __init__(self, inputs, masks, labels):
        self.inputs = inputs
        self.masks = masks
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx],
            'attention_mask': self.masks[idx],
            'labels': self.labels[idx]
        }

def save_tokenized_data(inputs, labels, filename):
    with open(filename, 'wb') as f:
        pickle.dump({'inputs': inputs, 'labels': labels}, f)

def load_tokenized_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def train_model():
    # Update path for your local machine
    cleaned_file = "C:/Users/hp/Downloads/AI_Chatbot_Mental_Health_Support/processed_synthetic_chatbot_data.csv"
    if not os.path.exists(cleaned_file):
        raise FileNotFoundError(f"Cleaned dataset not found at {cleaned_file}. Ensure file exists.")

    df = pd.read_csv(cleaned_file)
    if not all(col in df.columns for col in ['input', 'output']):
        raise ValueError(f"Dataset must contain 'input' and 'output' columns. Found: {list(df.columns)}")

    # Preprocess the input,output columns (stringified dicts)
    data = []
    for _, row in df.iterrows():
        try:
            input_msg = ast.literal_eval(row['input'])
            output_msg = ast.literal_eval(row['output'])
        except:
            continue

        if isinstance(input_msg, dict) and isinstance(output_msg, dict):
            input_text = input_msg.get('value', '').strip()
            input_from = input_msg.get('from', '')
            output_text = output_msg.get('value', '').strip()
            output_from = output_msg.get('from', '')

            if input_from == 'human' and output_from == 'gpt':
                user_text = input_text
                bot_text = output_text
            elif input_from == 'gpt' and output_from == 'human':
                user_text = output_text
                bot_text = input_text
            else:
                continue

            if user_text and bot_text:
                data.append({'input': user_text, 'output': bot_text})

    if not data:
        raise ValueError("No valid input-output pairs extracted from input,output columns.")

    df_processed = pd.DataFrame(data)
    df_processed = df_processed.sample(n=80000, random_state=42)  # Reduced to 20,000 samples
    train_df, val_df = train_test_split(df_processed, test_size=0.2, random_state=42)
    print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")

    print(f"First 3 rows of train_df:\n{train_df.head(3)}")

    model_name = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model on device: {next(model.parameters()).device}")
    model.to(device)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        model.resize_token_embeddings(len(tokenizer))

    tokenized_train_file = "C:/Users/hp/Downloads/AI_Chatbot_Mental_Health_Support/tokenized_train.pkl"
    tokenized_val_file = "C:/Users/hp/Downloads/AI_Chatbot_Mental_Health_Support/tokenized_val.pkl"

    train_inputs = tokenizer(
        list(train_df['input']),
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    train_outputs = tokenizer(
        list(train_df['output']),
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    train_labels = train_outputs['input_ids'].clone()
    train_labels[train_labels == tokenizer.pad_token_id] = -100

    val_inputs = tokenizer(
        list(val_df['input']),
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    val_outputs = tokenizer(
        list(val_df['output']),
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    val_labels = val_outputs['input_ids'].clone()
    val_labels[val_labels == tokenizer.pad_token_id] = -100

    print(f"Train inputs shape: {train_inputs['input_ids'].shape}")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Val inputs shape: {val_inputs['input_ids'].shape}")
    print(f"Val labels shape: {val_labels.shape}")

    save_tokenized_data(train_inputs, train_labels, tokenized_train_file)
    save_tokenized_data(val_inputs, val_labels, tokenized_val_file)

    assert len(train_inputs['input_ids']) == len(train_labels), "Train input and label sizes mismatch"
    assert len(val_inputs['input_ids']) == len(val_labels), "Validation input and label sizes mismatch"

    train_dataset = ChatDataset(
        train_inputs['input_ids'],
        train_inputs['attention_mask'],
        train_labels
    )
    val_dataset = ChatDataset(
        val_inputs['input_ids'],
        val_inputs['attention_mask'],
        val_labels
    )

    print(f"Size of train_dataset: {len(train_dataset)}")
    print(f"Size of val_dataset: {len(val_dataset)}")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=False)

    training_args = TrainingArguments(
        output_dir="C:/Users/hp/Downloads/AI_Chatbot_Mental_Health_Support/results",
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=5000,
        logging_dir="C:/Users/hp/Downloads/AI_Chatbot_Mental_Health_Support/logs",
        logging_strategy="steps",
        logging_steps=500,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        warmup_steps=200,
        weight_decay=0.01,
        learning_rate=5e-5,
        dataloader_pin_memory=False,
        dataloader_drop_last=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )

    trainer.train()
    model.save_pretrained("C:/Users/hp/Downloads/AI_Chatbot_Mental_Health_Support/output")
    tokenizer.save_pretrained("C:/Users/hp/Downloads/AI_Chatbot_Mental_Health_Support/output")
    print("Model and tokenizer saved to C:/Users/hp/Downloads/AI_Chatbot_Mental_Health_Support/output")

if __name__ == "__main__":
    train_model()

"""
#KAGGLE NOTEBOOK CODE
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
import os
import pickle
import ast

class ChatDataset(Dataset):
    def __init__(self, inputs, masks, labels):
        self.inputs = inputs
        self.masks = masks
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx],
            'attention_mask': self.masks[idx],
            'labels': self.labels[idx]
        }

def save_tokenized_data(inputs, labels, filename):
    with open(filename, 'wb') as f:
        pickle.dump({'inputs': inputs, 'labels': labels}, f)

def load_tokenized_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def train_model():
    cleaned_file = "/kaggle/input/mental-health-chatbot-data/processed_synthetic_chatbot_data.csv"
    if not os.path.exists(cleaned_file):
        raise FileNotFoundError(f"Cleaned dataset not found at {cleaned_file}. Ensure the dataset is added to the notebook.")

    df = pd.read_csv(cleaned_file)
    if not all(col in df.columns for col in ['input', 'output']):
        raise ValueError(f"Dataset must contain 'input' and 'output' columns. Found: {list(df.columns)}")

    data = []
    for _, row in df.iterrows():
        try:
            input_msg = ast.literal_eval(row['input'])
            output_msg = ast.literal_eval(row['output'])
        except:
            continue

        if isinstance(input_msg, dict) and isinstance(output_msg, dict):
            input_text = input_msg.get('value', '').strip()
            input_from = input_msg.get('from', '')
            output_text = output_msg.get('value', '').strip()
            output_from = output_msg.get('from', '')

            if input_from == 'human' and output_from == 'gpt':
                user_text = input_text
                bot_text = output_text
            elif input_from == 'gpt' and output_from == 'human':
                user_text = output_text
                bot_text = input_text
            else:
                continue

            if user_text and bot_text:
                data.append({'input': user_text, 'output': bot_text})

    if not data:
        raise ValueError("No valid input-output pairs extracted from input,output columns.")

    df_processed = pd.DataFrame(data)
    df_processed = df_processed.sample(n=100000, random_state=42)  # 80,000 samples for 8,000 steps
    train_df, val_df = train_test_split(df_processed, test_size=0.2, random_state=42)
    print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")

    print(f"First 3 rows of train_df:\n{train_df.head(3)}")

    model_name = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model on device: {next(model.parameters()).device}")
    model.to(device)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        model.resize_token_embeddings(len(tokenizer))

    tokenized_train_file = "/kaggle/working/tokenized_train.pkl"
    tokenized_val_file = "/kaggle/working/tokenized_val.pkl"

    train_inputs = tokenizer(
        list(train_df['input']),
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    train_outputs = tokenizer(
        list(train_df['output']),
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    train_labels = train_outputs['input_ids'].clone()
    train_labels[train_labels == tokenizer.pad_token_id] = -100

    val_inputs = tokenizer(
        list(val_df['input']),
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    val_outputs = tokenizer(
        list(val_df['output']),
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    val_labels = val_outputs['input_ids'].clone()
    val_labels[val_labels == tokenizer.pad_token_id] = -100

    print(f"Train inputs shape: {train_inputs['input_ids'].shape}")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Val inputs shape: {val_inputs['input_ids'].shape}")
    print(f"Val labels shape: {val_labels.shape}")

    save_tokenized_data(train_inputs, train_labels, tokenized_train_file)
    save_tokenized_data(val_inputs, val_labels, tokenized_val_file)

    assert len(train_inputs['input_ids']) == len(train_labels), "Train input and label sizes mismatch"
    assert len(val_inputs['input_ids']) == len(val_labels), "Validation input and label sizes mismatch"

    train_dataset = ChatDataset(
        train_inputs['input_ids'],
        train_inputs['attention_mask'],
        train_labels
    )
    val_dataset = ChatDataset(
        val_inputs['input_ids'],
        val_inputs['attention_mask'],
        val_labels
    )

    print(f"Size of train_dataset: {len(train_dataset)}")
    print(f"Size of val_dataset: {len(val_dataset)}")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=False)

    training_args = TrainingArguments(
        output_dir="/kaggle/working/results",
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=5000,
        logging_dir="/kaggle/working/logs",
        logging_strategy="steps",
        logging_steps=500,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        warmup_steps=200,
        weight_decay=0.01,
        learning_rate=5e-5,
        dataloader_pin_memory=False,
        dataloader_drop_last=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )

    trainer.train()
    model.save_pretrained("/kaggle/working/output")
    tokenizer.save_pretrained("/kaggle/working/output")
    print("Model and tokenizer saved to /kaggle/working/output")

if __name__ == "__main__":
    train_model()