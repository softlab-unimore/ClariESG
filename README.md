# ClariGRI 🌱

ClariGRI is an end-to-end system designed to extract, clean, structure, and semantically query information contained in corporate sustainability reports (PDF). It integrates LLM-based language understanding, table extraction, numerical reasoning, and sector-aware contextualization through an interactive Gradio interface. The system supports ESG analysis, report comparison, automated table extraction, and RAG-based question answering.



## ✨ Main Features

### PDF Upload & Processing
ClariGRI automatically:
- extracts the company name  
- identifies, cleans, and standardizes GRI-related tables  
- generates metadata  
- stores structured tables in `table_dataset/`  
- inserts dense and sparse embeddings into PostgreSQL  

### Table Dataset Management
For each processed report, the system generates a dedicated folder inside `table_dataset/` containing:  
- cleaned CSV tables  
- extracted GRI indicators  
- metadata files   

### RAG-based Chatbot
The chatbot allows users to:
- query uploaded reports  
- query companies  
- query industrial sectors  
- perform numerical reasoning using Program-of-Thought  
- retrieve relevant tables and text segments  

Retrieval uses a hybrid dense + sparse strategy powered by OpenAI, LangChain, and pgvector.

### Fully Dockerized
A single Docker container includes the Python backend, the Gradio interface, PostgreSQL with pgvector, and the full processing pipeline.


# 📦 Installation & Setup (Docker Only) 

This is the only required setup method. No Git clone is necessary for normal usage.
Before running ClariGRI, you must install **Docker Desktop** on your machine.🐋

Download it from:  
https://www.docker.com/products/docker-desktop/

Docker Desktop is required in order to pull, run, and manage the ClariGRI container. Once installed, make sure it is running before executing any Docker commands.


## 1. Create a Local Folder

Create a directory on your Desktop, for example: ``` clarigri/ ```
Inside it, prepare the following structure:
```
clarigri/
│
├── .env
├── reports/
└── table_dataset/
```

### reports/
Copy the entire `reports/` folder from the GitHub repository into your local directory. From ```repo → Code button → Download ZIP``` extract and keep only the folder you need.
It contains example sustainability reports used by the demo. You may also add your own PDF reports inside this folder.

### table_dataset/
Create this folder empty.  
ClariGRI will automatically populate it as you process reports.


## 2. Create the `.env` File

Create a `.env` file inside your project folder with the following content. Make sure the file is named **exactly** `.env` and does **not** have extensions like `.txt` or similar.

```env
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=griqa
POSTGRES_PORT=5432
POSTGRES_EMB_TABLE_NAME=langchain_pg_embedding
POSTGRES_SPARSE_TABLE_NAME=sparse_table

DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5432/griqa

PYTHONHASHSEED=0

OPENAI_API_KEY=YOUR_OPENAI_KEY_HERE
OPENAI_MODEL_NAME=gpt-4o-mini
OPENAI_TEMPERATURE=0.2
```

Replace `YOUR_OPENAI_KEY_HERE` with your actual OpenAI key.

## 3. Pull the Docker Image

Open a shell and run the following command:
```console
docker pull martasantacroce/clarigri:latest
```

## 4. Run the Docker Container

From inside your ```clarigri/``` folder open a shell and digit:
```console
docker run --name clarigri_container \
  --env-file .env \
  -v ./reports:/app/reports \
  -v ./table_dataset:/app/table_dataset \
  -p 7860:7860 \
  -p 5432:5432 \
  -p 8080:8080 \
  martasantacroce/clarigri:latest
```
This command:

- loads your `.env`  
- mounts `reports/` as input  
- mounts `table_dataset/` as output  
- exposes Gradio (7860) and PostgreSQL (5432)

After the first build and creation of the container, you can start it again at any time with:

```console
docker start clarigri_container
```



## 5. Access the Web Interface

Open: [http://localhost:7860](http://localhost:7860)

## ▶️ Usage Summary

1. Create a folder (`clarigri/`).  
2. Copy the entire `reports/` folder from GitHub.  
3. Add your own PDF reports if desired.  
4. Create an empty `table_dataset/` folder.  
5. Add the `.env` file.  
6. Pull the Docker image.  
7. Run the container.  
8. Use the Gradio interface to upload, process, and query ESG data.

# 📄 License

This project is released under the **MIT License**.
