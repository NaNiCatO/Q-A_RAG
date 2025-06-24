---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:190
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
widget:
- source_sentence: If a database column is VARCHAR(50), what attribute should be added
    to the corresponding entity property in the code?
  sentences:
  - 'การใช้ค่าคงที่: เพื่อให้คนอื่นทราบว่าค่าๆ นั้นหมายถึงอะไร'
  - 'Field Length Validation: The attribute [MaxLength(50)] should be added to the
    property.'
  - 'การบันทึก Log: โดยทั่วไปต้องมี 3 จุด ได้แก่ Logger ''Start'' ที่ส่วนแรกสุดใน
    try , Logger ''End'' ก่อนบรรทัด return  และ Logger ''Error'' ภายใน catch '
- source_sentence: A component needs to run code when it first loads and just before
    it is removed. What interfaces must it implement and what methods must it contain?
  sentences:
  - 'การคำนวณที่ Backend: เพราะมีความเสี่ยงที่ผู้ใช้อาจแก้ไขค่าที่คำนวณแล้ว (เช่น
    แก้ไขยอดรวมจาก 400 เป็น 999) แล้วส่งไปให้ Backend โดยตรง'
  - 'File Handling & Validation: 1. First, validate that the file is a legitimate
    Excel file using a function like IsValidExcelFile(file).  &lt;br> 2. Second, safely
    read the file into memory using a using (MemoryStream ms = new()) block to ensure
    resources are managed correctly. '
  - 'Component Structure: It must implement OnInit and OnDestroy.  It must have an
    ngOnInit() method for the initial code  and an override ngOnDestroy() method for
    the removal code.'
- source_sentence: ในตัวอย่างการ subscribe ของฟังก์ชัน save (source 9) โค้ดใน next
    block ที่มี if/else if/else กำลังตรวจสอบอะไร?
  sentences:
  - 'การตรวจสอบความยาวฟิลด์: ควรเพิ่ม Attribute [MaxLength(50)] ให้กับ Property นั้นๆ
    ในคลาส Entity'
  - 'โครงสร้างไฟล์ Master.cs: มี return type เป็น Task<MasterData>'
  - 'การ Subscribe: กำลังตรวจสอบค่า res?.status ที่ได้กลับมา โดยเทียบกับค่าคงที่ Status.success
    และ Status.dupp เพื่อจัดการกับผลลัพธ์ของการบันทึกที่แตกต่างกัน'
- source_sentence: How should I handle text labels and messages in the application?
    Should I write them directly in the HTML?
  sentences:
  - 'Text & Labels: You must not hardcode text. All general words and labels should
    be queried from the database and displayed using a translation system, such as
    `{{ "label.RDMS01.DefineDecoration"'
  - 'Try-Catch Blocks: Every method must be enclosed in a try-catch block.'
  - 'Required Field Validation: Backend validation is necessary to handle cases where
    a request is sent directly via tools like Postman, bypassing frontend checks. '
- source_sentence: How does a single Master.cs file handle requests for different
    kinds of data?
  sentences:
  - 'Subscriptions & Text Labels: มีการเรียก this.ms.success() ใน next block หลังจากลบสำเร็จ  และเรียก
    this.ms.error() ใน error block หากการทำงานล้มเหลว'
  - 'Master.cs File Structure: It uses a switch statement on a request parameter (e.g.,
    request.Select) to determine which data to fetch and return. '
  - 'Null Checks: For a collection, you must check if it is not null AND if it has
    any members: if (items != null && items.Any()). For a single object, you can simply
    check if it is not null before using its properties, often with the ?. operator:
    if (emp?.MainDept != null). '
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) <!-- at revision 86741b4e3f5cb7765a600d3a3d55a0f6a6cb443d -->
- **Maximum Sequence Length:** 128 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 128, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'How does a single Master.cs file handle requests for different kinds of data?',
    'Master.cs File Structure: It uses a switch statement on a request parameter (e.g., request.Select) to determine which data to fetch and return. ',
    'Subscriptions & Text Labels: มีการเรียก this.ms.success() ใน next block หลังจากลบสำเร็จ  และเรียก this.ms.error() ใน error block หากการทำงานล้มเหลว',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 190 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 190 samples:
  |         | sentence_0                                                                        | sentence_1                                                                          |
  |:--------|:----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                              |
  | details | <ul><li>min: 9 tokens</li><li>mean: 24.68 tokens</li><li>max: 85 tokens</li></ul> | <ul><li>min: 12 tokens</li><li>mean: 44.28 tokens</li><li>max: 105 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                  | sentence_1                                                                                                                                                                                                                                                                     |
  |:------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>An intern creates two files: orms04-add.component.ts and orms04-edit.component.ts. Why is this incorrect according to the standard?</code>            | <code>Page Naming: This is incorrect because the standard specifies that for a module's add and detail pages, only a single component should be created (e.g., orms04-detail). The mode should be handled internally with a flag. </code>                                      |
  | <code>In a -detail component, where should the logic to check for 'edit' mode versus 'add' mode be placed?</code>                                           | <code>Page Naming & Lifecycle: This logic should be placed inside the subscribe block of the route's data resolver. This subscription is typically called from within the ngOnInit lifecycle hook.</code>                                                                      |
  | <code>A function has 'Start' and 'End' logs, but is missing the SQL parameter and SQL query logs. What is the risk of omitting these when debugging?</code> | <code>Logging: Without the parameter log, you can't see what specific values were used in the query. Without the final SQL log, you can't see the exact query that was executed, making it very difficult to reproduce and fix bugs related to the data or query logic.</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Framework Versions
- Python: 3.12.10
- Sentence Transformers: 3.4.1
- Transformers: 4.48.3
- PyTorch: 2.5.1+cu121
- Accelerate: 1.4.0
- Datasets: 3.3.2
- Tokenizers: 0.21.0

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->