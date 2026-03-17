# flake8: noqa
"""
OpenCompass evaluation config for Qwen3 training checkpoints.

Evaluates on: math500, aime2025, ifeval, livecodebench
All evaluators are rule-based (no LLM judge required).
Thinking mode enabled: <think> tags stripped before evaluation.

Parameterized via environment variables:
  CKPT_PATH   - path to HuggingFace-format checkpoint directory (required)
  MODEL_ABBR  - short name for the model (used in output paths)
  TP_SIZE     - tensor parallel size (default: 1)
  BENCHMARKS  - comma-separated list of benchmarks to run
               (default: math500,aime2025,ifeval,livecodebench)
"""
# All imports use __import__ / importlib.import_module to bypass mmengine's
# lazy-import interception (which replaces every `from x import y` with a
# LazyAttr object, breaking isinstance checks, list concatenation, etc.).
_imp = __import__('importlib').import_module

# -- Environment variables ---------------------------------------------------
_env = __import__('os').environ
CKPT_PATH = _env['CKPT_PATH']
MODEL_ABBR = _env.get('MODEL_ABBR', 'qwen3-ckpt')
TP_SIZE = int(_env.get('TP_SIZE', '1'))
BENCHMARKS = set(
    b.strip()
    for b in _env.get('BENCHMARKS', 'math500,aime2025,ifeval,livecodebench').split(',')
    if b.strip()
)

# -- OpenCompass imports ------------------------------------------------------
VLLMwithChatTemplate      = _imp('opencompass.models').VLLMwithChatTemplate
NaivePartitioner          = _imp('opencompass.partitioners').NaivePartitioner
NumWorkerPartitioner      = _imp('opencompass.partitioners').NumWorkerPartitioner
LocalRunner               = _imp('opencompass.runners').LocalRunner
OpenICLEvalTask           = _imp('opencompass.tasks').OpenICLEvalTask
OpenICLInferTask          = _imp('opencompass.tasks').OpenICLInferTask
extract_non_reasoning_content = _imp('opencompass.utils.text_postprocessors').extract_non_reasoning_content

#######################################################################
#                          PART 0  Datasets                           #
#######################################################################

# -- math500 (MATHVerifyEvaluator, n=1) ----------------------------------
if 'math500' in BENCHMARKS:
    math_datasets = _imp('opencompass.configs.datasets.math.math_500_gen').math_datasets
else:
    math_datasets = []

# -- IFEval (IFEvaluator, n=1) -------------------------------------------
if 'ifeval' in BENCHMARKS:
    ifeval_datasets = _imp('opencompass.configs.datasets.IFEval.IFEval_gen_353ae7').ifeval_datasets
else:
    ifeval_datasets = []

# -- LiveCodeBench v6 academic (LCBCodeGenerationEvaluator, n=6) ----------
if 'livecodebench' in BENCHMARKS:
    lcb_datasets = [_imp('opencompass.configs.datasets.livecodebench.livecodebench_v6_academic').LCBCodeGeneration_dataset]
else:
    lcb_datasets = []

# -- AIME2025 with MATHVerifyEvaluator only (no LLM judge), n=32 ---------
if 'aime2025' in BENCHMARKS:
    PromptTemplate      = _imp('opencompass.openicl.icl_prompt_template').PromptTemplate
    ZeroRetriever       = _imp('opencompass.openicl.icl_retriever').ZeroRetriever
    GenInferencer       = _imp('opencompass.openicl.icl_inferencer').GenInferencer
    CustomDataset       = _imp('opencompass.datasets').CustomDataset
    MATHVerifyEvaluator = _imp('opencompass.evaluator').MATHVerifyEvaluator

    aime2025_datasets = [
        dict(
            type=CustomDataset,
            abbr='aime2025_repeat_32',
            path='opencompass/aime2025',
            reader_cfg=dict(input_columns=['question'], output_column='answer'),
            infer_cfg=dict(
                prompt_template=dict(
                    type=PromptTemplate,
                    template=dict(
                        round=[
                            dict(
                                role='HUMAN',
                                prompt='{question}\nRemember to put your final answer within \\boxed{}.',
                            ),
                        ],
                    ),
                ),
                retriever=dict(type=ZeroRetriever),
                inferencer=dict(type=GenInferencer),
            ),
            eval_cfg=dict(evaluator=dict(type=MATHVerifyEvaluator)),
            n=32,
        )
    ]
else:
    aime2025_datasets = []

#######################################################################
#                          PART 1  Dataset List                       #
#######################################################################
datasets = math_datasets + aime2025_datasets + ifeval_datasets + lcb_datasets

# Override max_out_len to 32768 for all datasets (academic standard for reasoning)
for dataset in datasets:
    dataset['infer_cfg']['inferencer']['max_out_len'] = 32768

#######################################################################
#                          PART 2  Summarizer                         #
#######################################################################
_dataset_abbrs = []

if 'math500' in BENCHMARKS or 'aime2025' in BENCHMARKS:
    _dataset_abbrs.append('')
    _dataset_abbrs.append('Math')
    if 'math500' in BENCHMARKS:
        _dataset_abbrs.append(['math-500', 'accuracy'])
    if 'aime2025' in BENCHMARKS:
        _dataset_abbrs.append(['aime2025_repeat_32', 'accuracy'])

if 'ifeval' in BENCHMARKS:
    _dataset_abbrs.append('')
    _dataset_abbrs.append('Instruction Following')
    _dataset_abbrs.append(['IFEval', 'Prompt-level-strict-accuracy'])

if 'livecodebench' in BENCHMARKS:
    _dataset_abbrs.append('')
    _dataset_abbrs.append('Code')
    _dataset_abbrs.append(['lcb_code_generation_repeat_6', 'pass@1'])

summarizer = dict(
    dataset_abbrs=_dataset_abbrs,
    summary_groups=[],
)

#######################################################################
#                          PART 3  Model                              #
#######################################################################
models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr=MODEL_ABBR,
        path=CKPT_PATH,
        model_kwargs=dict(tensor_parallel_size=TP_SIZE),
        max_out_len=32768,
        max_seq_len=32768,
        batch_size=16,
        generation_kwargs=dict(temperature=0.6, top_p=0.95, top_k=20),
        chat_template_kwargs=dict(enable_thinking=True),
        run_cfg=dict(num_gpus=TP_SIZE),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]

#######################################################################
#                   PART 4  Inference/Eval Runner                     #
#######################################################################
infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=8),
    runner=dict(
        type=LocalRunner,
        max_num_workers=16,
        retry=0,
        task=dict(type=OpenICLInferTask),
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner, n=10),
    runner=dict(
        type=LocalRunner,
        max_num_workers=16,
        task=dict(type=OpenICLEvalTask),
    ),
)
