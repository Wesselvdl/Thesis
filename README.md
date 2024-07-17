# README

Welcome to the GitHub page for the thesis of Wessel van der Linde (12857130). This project focuses on creating a Retrieval-Augmented Generation (RAG) model to answer the following question:

To what extent can an advanced Retrieval-Augmented Generation (RAG) system be designed to efficiently generate actionable compliance plans for Digital Infrastructure in Data Centers?

## Overview

This repository contains the code, results, and embeddings used in the research for the Retrieval-Augmented Generation (RAG) system designed to generate actionable compliance plans for data centers. The repository is organized into the following sections:

- **Results**: Contains the findings of the study, including analysis and evaluation metrics.
- **RAG**: Includes the codebase for the RAG system, covering data processing, model training, and inference.
- **Embeddings**: Stores the precomputed embeddings used for vector search in the system.

## Repository Structure

### Results

The `Results` folder contains the following items:

1. **EvulationScorer.ipynb**: Jupyter notebook for generating and analyzing the results.
2. **Results.csv**: A CSV file with the detailed analysis results of the Cypher query accuracy and completeness.
3. **Results.xlsx**: An Excel file containing the raw timings for various operations.

### RAG

The `RAG` folder contains the code necessary to replicate the RAG system. It includes:

1. **agent.py**: Script for the main agent responsible for handling queries.
2. **bot.py**: Script for the bot interface.
3. **neo4jKEYS.txt**: Configuration file containing Neo4j keys (ensure this is secured properly).
4. **requirements.txt**: List of dependencies required to run the code.
5. **temp.py**: Temporary scripts for testing purposes.
6. **utils.py**: Utility functions used across different scripts.
7. **solutions/**: Folder containing various solution scripts.
8. **tools/**: Folder with additional tools and helper scripts.
9. **README.adoc**: Documentation for the RAG codebase.

### Embeddings

The `Embeddings` folder includes:

1. **Embedding.ipynb**: Jupyter notebook for generating the embeddings.
2. **DataEmbeddings.csv**: CSV file containing the precomputed vector embeddings for the regulatory documents.

## Getting Started

To replicate the research and run the RAG system, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/rag-compliance.git
    cd rag-compliance
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Set Up Neo4j**:
    - Install Neo4j and set up the database.
    - Update the `neo4jKEYS.txt` file with the appropriate credentials.

4. **Run the RAG System**:
    - Run the following command to start the RAG system.
    ```bash
    cd RAG
    streamlit run bot.py
    ```

5. **Interact with the Bot**:
    - Open the provided URL in your browser to interact with the RAG system.

6. **Query the System**:
    - Enter a query in natural language to generate actionable compliance plans.

7. **Generate Results**:
    - Use the system to generate compliance plans and evaluate the results.

7. **Analyze the Results**:
    - Use the provided Jupyter notebooks to analyze the results and evaluate the system based on Expert Action Plan (EAP) and Cypher query accuracy and completeness.

## Results Analysis

To analyze the results and reproduce the findings:

1. **Run the Jupyter Notebook**:
    Open the `EvulationScorer.ipynb` notebook and execute the cells to generate and analyze the results.
    ```bash
    jupyter notebook Results/EvulationScorer.ipynb
    ```

## Embeddings

To generate and use embeddings for vector search:

1. **Run the Embeddings Notebook**:
    Open the `Embedding.ipynb` notebook and execute the cells to generate the embeddings.
    ```bash
    jupyter notebook Embeddings/Embedding.ipynb
    ```

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

This README file provides clear instructions on the repository structure and how to use the provided code and data to replicate the research. Feel free to customize it further based on any additional details or specific instructions you might have.