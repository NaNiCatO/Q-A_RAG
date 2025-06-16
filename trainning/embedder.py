## ---------------------------------
## LIBRARIES & SETUP
## ---------------------------------
print("✅ Loading libraries...")
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
print("Libraries loaded successfully.")

# Check if a CUDA-enabled GPU is available for faster training
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")

## ---------------------------------
## STEP 1: DATA PREPARATION
## ---------------------------------
print("\n✅ Starting Step 1: Data Preparation...")
# --- This is the data loading code we customized for your CSV ---
csv_file_path = 'Dataset.xlsx'
try:
    df = pd.read_excel(csv_file_path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"❌ Error: Dataset file not found at '{csv_file_path}'.")
    exit()

# Define the column names from your CSV file.
# PLEASE DOUBLE-CHECK THESE MATCH YOUR FILE EXACTLY.
topic_col = 'Topic Name'
answer_col = 'Answer'
question_col = 'Question'
required_cols = [topic_col, answer_col, question_col]

# Validate that all required columns exist in the DataFrame
if not all(col in df.columns for col in required_cols):
    print(f"❌ Error: Missing one or more required columns. Found: {df.columns.tolist()}, Required: {required_cols}")
    exit()

train_examples = []
# Iterate over each row and create the training pairs
for index, row in df.iterrows():
    topic = row[topic_col]
    answer = row[answer_col]
    question = row[question_col]
    
    # Ensure all required fields for a row are non-empty
    if pd.notna(topic) and pd.notna(answer) and pd.notna(question) and pd.notna(question):
        context_chunk = f"{topic}: {answer}"
        train_examples.append(InputExample(texts=[question, context_chunk]))

print(f"Data preparation complete. Total training pairs created: {len(train_examples)}")

## ---------------------------------
## STEP 2: MODEL & TRAINING SETUP
## ---------------------------------
print("\n✅ Starting Step 2: Model and Training Setup...")
# Load the pre-trained bi-encoder model from Hugging Face
model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
model = SentenceTransformer(model_name, device=device)
print(f"Model '{model_name}' loaded.")

# The DataLoader shuffles and batches the data.
# A larger batch size is better for this loss function. Adjust based on your GPU memory.
batch_size = 32

# The loss function will guide the model's learning process.
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
train_loss = losses.MultipleNegativesRankingLoss(model=model)
print(f"DataLoader and Loss Function (MultipleNegativesRankingLoss) are ready.")

## ---------------------------------
## STEP 3: FINE-TUNING THE MODEL
## ---------------------------------
print("\n✅ Starting Step 3: Fine-Tuning...")
# Define training parameters
num_epochs = 3  # For a real run, you might want 2-4 epochs
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1) # 10% of steps for warmup
output_path = './output/my-finetuned-model-v1'

print(f"""
Training Parameters:
- Epochs: {num_epochs}
- Batch Size: {batch_size}
- Warmup Steps: {warmup_steps}
- Output Path: {output_path}
""")

# Start the training!
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          output_path=output_path,
          show_progress_bar=True,
          save_best_model=True) # Saves the best model based on validation loss (if evaluator is provided)

## ---------------------------------
## DONE!
## ---------------------------------
print("\n🎉🎉🎉 Fine-tuning complete! 🎉🎉🎉")
print(f"Your new, fine-tuned model is saved in the directory: '{output_path}'")
print("\nYou can now load this model for your application using:")
print(f"my_new_model = SentenceTransformer('{output_path}')")