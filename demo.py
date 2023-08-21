# coding=utf-8

import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings

from langchain import HuggingFacePipeline
from langchain import PromptTemplate,  LLMChain
from transformers import AutoModelForCausalLM
from langchain.document_loaders import PyPDFLoader
from transformers import pipeline
import json
import textwrap

pdf_file_path = "./attention is all you need.pdf"
model_path = "/mnt/h/Chinese-Llama-2-7b-4bit"


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT ):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

def cut_off_text(text, prompt):
    cutoff_phrase = prompt
    index = text.find(cutoff_phrase)
    if index != -1:
        return text[:index]
    else:
        return text

def remove_substring(string, substring):
    return string.replace(substring, "")



def generate(text):
    prompt = get_prompt(text)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
        outputs = model.generate(**inputs,
                                 max_new_tokens=512,
                                 eos_token_id=tokenizer.eos_token_id,
                                 pad_token_id=tokenizer.eos_token_id,
                                 )
        final_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        final_outputs = cut_off_text(final_outputs, '</s>')
        final_outputs = remove_substring(final_outputs, prompt)

    return final_outputs#, outputs

def parse_text(text):
        wrapped_text = textwrap.fill(text, width=100)
        print(wrapped_text +'\n\n')
        # return assistant_text


def main():
    pdf_loader = PyPDFLoader(pdf_file_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
    splitted_docs = text_splitter.split_documents(pdf_loader.load())

    EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cuda"},
    )
    vector_store = Chroma(
        collection_name='llama2_demo',
        embedding_function=embeddings,
        persist_directory='./'
    )
    vector_store.add_documents(splitted_docs)

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        max_new_tokens=512,
        do_sample=True,
        top_k=30,
        num_return_sequences=1,
        eos_token_id = tokenizer.eos_token_id
    )

    llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature':0})
    system_prompt = "You are an advanced assistant that excels at translation. "
    instruction = "Convert the following text from English to French:\n\n {text}"
    template = get_prompt(instruction, system_prompt)
    print(template)
    
    prompt = PromptTemplate(template=template, input_variables=["text"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    text = "how are you today?"
    output = llm_chain.run(text)

    parse_text(output)


if __name__=='__main__':
    main()




    
    



    
    
