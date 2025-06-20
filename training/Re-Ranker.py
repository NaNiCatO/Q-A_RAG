## ---------------------------------
## LIBRARIES & SETUP
## ---------------------------------
print("✅ Loading libraries...")
import pandas as pd
import random
import torch
import os
from torch.utils.data import DataLoader
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
print("Libraries loaded successfully.")

# Check if a CUDA-enabled GPU is available for faster training
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")

## ---------------------------------
## STEP 1: DATA PREPARATION (for Cross-Encoder)
## ---------------------------------
print("\n✅ Starting Step 1: Data Preparation for Cross-Encoder...")
csv_file_path = 'Dataset.xlsx'
try:
    df = pd.read_excel(csv_file_path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"❌ Error: Dataset file not found at '{csv_file_path}'.")
    exit()

# Define and validate column names from your CSV file
topic_col = 'Topic Name'
answer_col = 'Answer'
question_col = 'Question'
required_cols = [topic_col, answer_col, question_col]

if not all(col in df.columns for col in required_cols):
    print(f"❌ Error: Missing required columns. Found: {df.columns.tolist()}, Required: {required_cols}")
    exit()

# Clean data by dropping rows with any empty cells in the required columns
df.dropna(subset=required_cols, inplace=True)
print(f"Cleaned dataset has {len(df)} rows.")

cross_encoder_train_examples = []
all_contexts = [f"{row[topic_col]}: {row[answer_col]}" for index, row in df.iterrows()]
all_topics = df[topic_col].tolist()

for index, row in df.iterrows():
    topic = row[topic_col]
    positive_context = f"{row[topic_col]}: {row[answer_col]}"
    question = row[question_col]
    
    cross_encoder_train_examples.append(InputExample(texts=[question, positive_context], label=1))
    
    while True:
        random_index = random.randint(0, len(all_contexts) - 1)
        if all_topics[random_index] != topic:
            negative_context = all_contexts[random_index]
            break
            
    cross_encoder_train_examples.append(InputExample(texts=[question, negative_context], label=0))

print(f"Data preparation complete. Total training examples created: {len(cross_encoder_train_examples)}")

## ---------------------------------
## STEP 2: MODEL & TRAINING SETUP
## ---------------------------------
print("\n✅ Starting Step 2: Model and Training Setup...")
model_name = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
model = CrossEncoder(model_name, num_labels=1, device=device)
print(f"Model '{model_name}' loaded.")

batch_size = 32
train_dataloader = DataLoader(cross_encoder_train_examples, shuffle=True, batch_size=batch_size)
print(f"DataLoader is ready with batch size {batch_size}.")

## ---------------------------------
## STEP 3: FINE-TUNING THE MODEL
## ---------------------------------
print("\n✅ Starting Step 3: Fine-Tuning...")
num_epochs = 3
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
output_path = './output/my-finetuned-cross-encoder-v1'

print(f"""
Training Parameters:
- Epochs: {num_epochs}
- Batch Size: {batch_size}
- Warmup Steps: {warmup_steps}
- Output Path: {output_path}
""")

# Start the training!
model.fit(
    train_dataloader=train_dataloader,
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    show_progress_bar=True
)

## ---------------------------------
## STEP 4: SAVE THE MODEL (The Fix)
## ---------------------------------
print("\n✅ Starting Step 4: Saving the model...")
# Create the output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# This line explicitly saves the fine-tuned model to the specified path.
# This was the missing step.
model.save(output_path)
print("Model saved successfully.")


## ---------------------------------
## DONE!
## ---------------------------------
print("\n🎉🎉🎉 Re-ranker model fine-tuning and saving complete! 🎉🎉🎉")
print(f"Your new, fine-tuned cross-encoder is saved in: '{output_path}'")

