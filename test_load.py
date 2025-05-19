import os
from openai import OpenAI
import json
import streamlit as st
from scipy.spatial import distance
from streamlit_extras.mention import mention
import requests
from bs4 import BeautifulSoup

EMBEDDING_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-3.5-turbo"
OpenAI_API_KEY = 'Input Key'
client = OpenAI(api_key=OpenAI_API_KEY)

def load_embeddings(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    sections = data['sections']
    embeddings = data['embeddings']
    return sections, embeddings

def get_query_embedding(query):
    response = client.embeddings.create(input = [query], model=EMBEDDING_MODEL)
    return response.data[0].embedding

# def find_most_relevant_section(query_embedding, embeddings):
#     similarities = [1 - distance.cosine(query_embedding, emb) for emb in embeddings]
#     most_relevant_index = similarities.index(max(similarities))
#     return most_relevant_index


def find_top_n_relevant_sections(query_embedding, embeddings, n=3):
    similarities = [1 - distance.cosine(query_embedding, emb) for emb in embeddings]
    most_relevant_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:n]
    return most_relevant_indices

def ask_question(question, context):
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {context} \n\n Question: {question}"},
        ]
    )
    return response.choices[0].message.content


def fetch_title(url):
    try:
        response = requests.get(url, verify=False)
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.title.string if soup.title else url
    except:
        return url

sections, embeddings = load_embeddings('embeddings_total.json')

st.title("QA System")
st.write("Ask a question based on the content from multiple websites.")

query = st.text_input("Enter your query:")

if st.button('Submit Query'):
    query_embedding = get_query_embedding(query)

    top_n = 3  
    most_relevant_indices = find_top_n_relevant_sections(query_embedding, embeddings, top_n)
    most_relevant_sections = [sections[i] for i in most_relevant_indices]

    combined_context = " ".join([section["content"] for section in most_relevant_sections])
    answer = ask_question(query, combined_context)

    st.write("### Answer:")
    st.write(answer)
    st.write("### Sources:")
    displayed_links = set()
    for section in most_relevant_sections:
        if section['link'] not in displayed_links:
            mention(
                label= fetch_title(section['link']),
                icon="🔗",
                url=section['link'],
            )
            # st.write(f"- {section['link']}")
            displayed_links.add(section['link'])
