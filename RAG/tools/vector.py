import streamlit as st
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain.prompts.prompt import PromptTemplate
from solutions.llm import llm, embeddings
from langchain.chains import RetrievalQA

neo4jvector = Neo4jVector.from_existing_index(
    embedding = embeddings,                              # (1)
    url=st.secrets["NEO4J_URI"],             # (2)
    username=st.secrets["NEO4J_USERNAME"],   # (3)
    password=st.secrets["NEO4J_PASSWORD"],   # (4)
    index_name="EmbedFinder",                  # (5)
    node_label="Regulation",                       # (6)
    text_node_property="Description",       # (7)
    embedding_node_property="Embedding",     # (8)
    retrieval_query="""
RETURN
    node.Name AS name,
    node.Description AS Description, 
    node.Category AS category, 
    node.Section AS section, 
    node.SubSection AS sub_section, 
    node.ID AS id,
    node.Value AS value,
    {source: node.Regulation} AS metadata
"""
) 

retriever = neo4jvector.as_retriever()

QA_PROMPT = """
You have been asked to formulate a response to a user query based on the provided context. Use the context to generate a helpful answer.

EXAMPLE 1:
User Query: What is said in your database about `Life Cycle Assessment (LCA)`?
Expected Context: [{{'name': 'Life Cycle Assessment', 'Description': 'Introduce a plan for Life Cycle Assessment (LCA) in accordance with EU guidelines and internationally standardised methodologies.', 'Value': '0.5'}}]
Helpful Answer: Group Involvement is focused on maximizing energy efficiency.

If the provided information is empty, say that you don't know the answer.

Information:
{context}

Question: {question}
Helpful Answer:
"""

qa_prompt = PromptTemplate.from_template(QA_PROMPT)

kg_qa = RetrievalQA.from_chain_type(
    llm,                  # (1)
    chain_type="stuff",   # (2)
    chain_type_kwargs={"prompt": qa_prompt},
    retriever=retriever,  # (3)
)