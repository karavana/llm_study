## 📂 Project Structure

The project is organized as follows:

```
llm_case_study/
├── data/
│   ├── dr_voss_diary.pdf   # The document to process
│   ├── questions.txt       # Questions to answer
│   ├── answers.txt         # Answers to the questions, for evaluation/testing purposes
│   └── ...                 # Any other data files you create in your pipeline can go here
├── scripts/
│   ├── prepare_data.py     # Document processing pipeline (TO BE IMPLEMENTED)
│   └── eval.py             # Evaluation pipeline (TO BE IMPLEMENTED)
├── src/                    # Your custom modules and packages go here (OPTIONAL)
│   └── ...
├── app.py                  # FastAPI server implementation (TO BE IMPLEMENTED)
├── README.md               # Project documentation (TO BE UPDATED)
└── ...                     # Any other files your project needs
```


---

## 🔧 Technical Requirements

- **Vector Database:** Milvus Lite (required)
- **API Framework:** FastAPI (required)
- **Models:**  
  - All models used in your pipeline (LLMs, embedding models, or any others) are **open source**.  
  - We recommend using the following models:  
    - **LLM:** TinyLlama-1.1B-Chat-v1.0
    - **Embedding model:** snowflake-arctic-embed-s

---

## 📝 Report Guidelines

### 1. **Installation & Setup Instructions**  


- **Environment Setup:**  
  - For the docker image, we are using Python 3.10-slim
  - For the host computer, we need to use any Python > 3.9
  - We are using a docker container so there is no need spend time on the environment setup for the container. We have some assumptions though;
		1. Docker is downloaded and installed, WSL is configured, Linux (Ubuntu) is installed on the Docker system.
  - For the host computer (we are going to run the eval.py on our computer, not on the docker container), we also need to install several libraries
		1. open a CMD (do not close it, we are going to use it in the other steps)
		2. type; pip install google-genai
		3. extract the ZIP file I have sent as an attachment
  - Thanks to the YAML and Dockerfile configurations, we are just going to;
		1. Open the docker application as admin
		2. In the first CMD we opened, change the current directory into the folder llm_study-main (the one we extracted from ZIP)
		3. type; docker-compose build
		4. Wait for almost 15 minutes because it will setup the entire environment with transformers, torch, langchain, etc. 
  - After completing the above steps;
		1. Switch to the first CMD we opened(the one we build the docker container)
		2. type; docker-compose run --rm app python scripts/prepare_data.py
		3. make sure the data preparation step is completed (a message like "pipeline completed successfully!" should appear)
		4. Open another CMD and change the current directory into the folder llm_study-main again (do not close this for other steps as well)
		5. In this new CMD we opened, type; docker-compose up
		6. Wait until the container is up, we should see a message like  
INFO:     Application startup complete. 
INFO:     Uvicorn running on http://0.0.0.0:8000(Press CTRL+C to quit)
  

- **Dependency Installation:**  
  - Dependencies are listed inside the requirements.txt, but this is directly installed when we setup the environment, thanks to Docker container build
  - For the host computer, we need to use the pip commands from the Environment Setup title.

- **Model Downloads & Setup (if required):**  
  - We don't need to download or configure anything, the docker container has everything set up and the models are downloaded into RAM during run-time. 
  - No external files or programs are necessary, but if we need to test some of the individual questions (not the entire question set) we can use CURL
  and we need to set that up. Here is the link for that (https://kb.naverisk.com/en/articles/5569958-how-to-install-curl-in-windows).

- **Running the Scripts & Application:**  
  - Doing ‘docker-compose up’ already runs our fastapi, and the prepare_data.py was explained previously, so the only thing left is the evaluation phase.
  - In order to test a specific question, in the first opened CMD (the one we prepared the data) we can type;
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d "{\"question\":\"What is the Veridian policy towards space colonization?\"}"
  - If we want to test every question in the questions.txt set, then;
		1. In the first CMD (the one we prepared the data), we should be already in the llm_study-main folder and we simply call;
		‘python scripts/eval.py’
		and it will start evaluating "based on string matching techniques". Fortunately, our eval.py is equipped with a better evaluation criteria; LLM-as-a-judge.
		In order to use this though, Gemini gives free API KEYs but its quota is limited (we can obtain it from https://aistudio.google.com/apikey and click
		"Create API Key"). After obtaining our key, we have to set it in our environment for safe access. Just do;
		setx GOOGLE_API_KEY "your api key"
		and it should be set in your environment.
		After this, we can again do
		python scripts/eval.py
		and it will start using Gemini as a judge, which can evaluate our answers more fairly. We can only evaluate half of the answers before we exhaust our quota. 



### 2. **Technical Discussion:**  

   - **Model Selection:**
     - snowflake/snowflake-arctic-embed-s
It is a task requirement. Chosen for its compact size (384-dim), reasonable performance, and HuggingFace availability. Works well in resource-constrained environments and integrates cleanly with sentence-transformers.
     - The choice of LLM is TinyLlama-1.1B-Chat-v1.0, because I was rejected from using the required Llama-3.2-1B model unfortunately. Only open source and light option is this.
There are many great open source models, but they are either gated models (require permission from the authors) or they are too big for my RAM, I am out of memory suddenly.
The only thing I had was this model, and this has many restrictions. There are hallucinations and it does not obey the custom prompts I am injecting.
   - **Data Processing:**
     - Parsing is easily done with pymupdf. When I started using langchain, I directly used its pymupdf loader to prevent importing two of the same libraries.
     - I used a custom chunking strategy with sliding windows. It extracted many irrelevant text and the sentences were cut-short, making us lose the context.
After this, I adapted langchain's RecursiveCharacterTextSplitter where we are able to split the text into sentences, and we could get less text (avoid overwhelming the tiny model)
   - **Retrieval System:**
     - We used Milvus Lite (which was a requirement)
     - We have chosen L2 similarity, Top-k retrieval (k=5 by default) and we retrieved relevant context using Milvus db, concatenated and passed as input to the LLM with a hard custom prompt.
     - In the custom chunking technique, almost half of the questions were incorrectly answered or hallucinated. After adapting the recursivecharactertextsplitter
it slightly got better. String matching evaluation is not enough to evaluate and expect every generated answer to be exactly like the expected answer. If we use the power of
LLM-as-a-judge technique, we can see better (and correct) evaluation statistics. If we inspect the output from eval.py, we can clearly see that some of the answers are semantically correct
but will be evaluated as incorrect in the string matching evaluation, unfortunately.
	 and being creative, and on top of that I have had to adapt the ParentDocumentRetriever. 
   - **Results and Analysis:**
	  - The model is very tiny, and it can only answer correctly if there are no confusions in the context and the context should be very small because of the token memory.
		If we use a bigger model, then we need more RAM, also access permissions from the authors.
	  - We needed a more complex retrieval strategy to compensate the tiny model's incapabilities. I have reduced the temperature to avoid paraphrasing 
	 The retrieved documents were drastically more relevant after the ParentDocumentRetriever.
	 I used dual splitting, parent and child with different chunks. These children chunks are better for semantic search since they are smaller. Each parent is assigned
	 a unique ID, and children of these parent chunks will have the same id to make them traceable. Large parent chunk is in the disk, and search index chunks will have
	 focused vectors. While retrieving in the app, we use the same sentence-transformer model, convert it into vectors, we search semantically, retrieve and combine.
	 There are disadvantages to this approach. If the answers resided in separate parts and if we had to combine them, then we could have lost the connections. If we ever need this
	 we have to swap the child for their parent (since we can trace it) and give the entire parent to LLM, to expect a combined answer.