import os
from openai import OpenAI
import json
from scipy.spatial import distance
import tiktoken

EMBEDDING_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-3.5-turbo"
OpenAI_API_KEY = 'Input key'
client = OpenAI(api_key=OpenAI_API_KEY)

def read_and_process_file(file_path, encoding='utf-8'):
    sections = []
    with open(file_path, 'r', encoding=encoding) as file:
        lines = file.readlines()
    
    link = None
    content = []
    for line in lines:
        line = line.strip()
        if line.startswith("http://") or line.startswith("https://"):
            if link and content:
                sections.append({"link": link, "content": " ".join(content)})
            link = line
            content = []
        else:
            content.append(line)
    
    if link and content:
        sections.append({"link": link, "content": " ".join(content)})

    return sections

def split_content_with_link(content, link, max_tokens=500):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(content)
    sections = [tokens[i:i+max_tokens] for i in range(0, len(tokens), max_tokens)]
    decoded_sections = [{"link": link, "content": tokenizer.decode(section)} for section in sections]
    return decoded_sections

def generate_embeddings(sections):
    embeddings = []
    for section in sections:
        response = client.embeddings.create(input = [section["content"]], model=EMBEDDING_MODEL)
        embeddings.append(response.data[0].embedding)
    return embeddings

file_path = 'web_content_new.txt'

raw_sections = read_and_process_file(file_path, encoding='utf-8')

all_sections = []
all_embeddings = []
for section in raw_sections:
    split_sections = split_content_with_link(section["content"], section["link"])
    embeddings = generate_embeddings(split_sections)
    for split_section, embedding in zip(split_sections, embeddings):
        all_sections.append(split_section)
        all_embeddings.append(embedding)

with open('embeddings_total.json', 'w') as f:
    json.dump({'sections': all_sections, 'embeddings': all_embeddings}, f)
