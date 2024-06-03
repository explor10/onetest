# Use a pipeline as a high-level helper
import datasets
import torch
import transformers
from transformers import pipeline,set_seed
from transformers import GPT2Tokenizer, GPT2Model,GPT2ForQuestionAnswering
from datasets import load_dataset,Dataset
from trl import SFTTrainer

data=load_dataset("databricks/databricks-dolly-15k",split='train')
print(data)

model = GPT2ForQuestionAnswering.from_pretrained('gpt2')

#trainer=SFTTrainer(model,train_dataset=data,dataset_text_field="context",max_seq_length=30).train()


set_seed(300)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
print(f'{tokenizer}'+f'\n')

text = 'When did Virgin Australia start operating?'
encoded_input = tokenizer(text,return_tensors='pt')
with torch.no_grad():
    output = model(**encoded_input)
print(type(input))
print(f'{input}'+f'\n',output)
#help(dict)


import shap

# load the model
pmodel = transformers.pipeline("question-answering")


# define two predictions, one that outputs the logits for the range start,
# and the other for the range end
def f(questions, start):
    outs = []
    for q in questions:
        question, context = q.split("[SEP]")
        d = pmodel.tokenizer(question, context)
        out = pmodel.model.forward(**{k: torch.tensor(d[k]).reshape(1, -1) for k in d})
        logits = out.start_logits if start else out.end_logits
        outs.append(logits.reshape(-1).detach().numpy())
    return outs


def f_start(questions):
    return f(questions, True)


def f_end(questions):
    return f(questions, False)


# attach a dynamic output_names property to the models so we can plot the tokens at each output position
def out_names(inputs):
    question, context = inputs.split("[SEP]")
    d = pmodel.tokenizer(question, context)
    return [pmodel.tokenizer.decode([id]) for id in d["input_ids"]]


f_start.output_names = out_names
f_end.output_names = out_names
data = [
    "What is on the table?[SEP]When I got home today I saw my cat on the table, and my frog on the floor."
]

explainer_start = shap.Explainer(f_start, pmodel.tokenizer)
shap_values_start = explainer_start(data)

shap.plots.text(shap_values_start)
explainer_end = shap.Explainer(f_end, pmodel.tokenizer)
shap_values_end = explainer_end(data)
