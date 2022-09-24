#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#install pytorch


# In[1]:


#!pip install transformers ipywidgets gradio --upgrade


# In[11]:


#!pip install sentencepiece


# In[3]:


#!pip install sacremoses


# In[21]:


import gradio as gr                   # UI library
from transformers import pipeline     # Transformers pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM    #tokenizer for hub

#insert hub for translation model ex: Helsinki-NLP/opus-mt-en-es is english to spanish hub
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es") 

model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-es")


# In[22]:


translation_pipeline = pipeline(model="Helsinki-NLP/opus-mt-en-es")


# In[23]:


results = translation_pipeline('Hello, my name is Alex. What is yours?')


# In[24]:


results[0]['translation_text']


# In[25]:


def translate_transformers(from_text):
    results = translation_pipeline(from_text)
    return results[0]['translation_text']


# In[26]:


translate_transformers('This is my translation app.')


# In[19]:


interface = gr.Interface(fn=translate_transformers, 
                         inputs=gr.inputs.Textbox(lines=2, placeholder='Text to translate'),
                        outputs='text')


# In[20]:


interface.launch()

