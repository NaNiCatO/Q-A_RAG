# How to Update the Coding Standards Knowledge Base

This guide provides instructions for processing updated or new `Standard Coding (.docx)` documents and generating the `knowledge_base.json` file required by our RAG application.

This process uses a powerful AI model to read the document, including text and images, and convert it into a structured format.

### Step 1: Go to an AI Chat Tool

Navigate to a web-based AI chat tool that supports multimodal inputs (text and image uploads), such as Google's AI Studio or another similar service.

### Step 2: Upload Your Document

These tools typically do not support direct `.docx` upload.

1.  Open your `.docx` file.
2.  Save it as a PDF, which will preserve the layout and images.
3.  Use the "upload file" or "upload image" feature in the AI tool to upload the PDF version of your document.

### Step 3: Use the Master Prompt

After the document is uploaded, you must provide the AI with a very specific set of instructions. Copy the entire prompt below and paste it into the chat window after your file upload.

```text
You are an expert technical writer and data processor. Your task is to analyze the following uploaded document, which contains coding standards, and convert it into a structured JSON format. The document includes both text and images of code.

Follow these instructions exactly:

1.  Read the entire document, including analyzing the code within any images, to identify each distinct coding rule. A rule often starts with a number (e.g., "1.", "2.").

2.  For each rule you find, create a JSON object. Each object must have two fields: "ref_id" and "content".

3.  For the "ref_id", create a unique identifier.
    - If the rule is for backend development (e.g., C#, SQL), use a 'BE-' prefix.
    - If the rule is for frontend development (e.g., Angular, HTML), use an 'FE-' prefix.
    - Follow the prefix with the rule number (e.g., "BE-1", "FE-12").

4.  For the "content", write a clear, concise, self-contained summary of the rule.
    - Start with a short title in bold (e.g., "**Logging Standards:**").
    - Combine all related information for that rule, including details from the text and code examples from the images, into a single, easy-to-understand paragraph.

5.  Your final output must be a single, valid JSON array `[...]` containing all the rule objects you found. Do not include any other text, explanations, or markdown formatting outside of the JSON array itself.
```

### Step 4: Generate, Verify, and Save

1.  Submit the prompt to the AI. It will generate a block of JSON text containing all the processed rules.
2.  **Verify the Output:** Carefully review the generated JSON. Ensure all rules have been captured, the `ref_id`s are correct, and the content is accurate. If the AI missed something, you can ask it to revise its output (e.g., "You missed rule #11, please add it and regenerate the JSON").
3.  **Save the File:** Once you are satisfied, copy the entire JSON array. Paste it into the `knowledge_base.json` file in our project, completely replacing the old content.

### Step 5: Load the New Data into the System

You are now ready to update the application's knowledge base.

1.  Open your terminal in the project directory.
2.  Run the ingestion script:
    ```bash
    python ingest_data.py
    ```

This will load your new, high-quality, AI-processed rules into the Redis database, and the assistant will now use this updated knowledge.
