### DS-11

👏 **DocuRAG**

Our objective is to create a chatbot model that can read and interpret the data from all the PDF, excel(still in progress) and word documents that would be provided to it from the manufacturing department and is to allow the chatbot to respond to highly specific questions related to the company's products without getting puzzled quickly and effecienty.

![Capture213](https://github.com/jwild3/DS-11/assets/169072725/1b9140f9-1b35-4c4e-8dda-0ab90d13d39b)

source: https://www.leewayhertz.com/llamaindex/


The diagram above illustrates the functionality of our project. Our model processes files in PDF format, extracting relevant information and performing indexing to facilitate easy retrieval. This indexed data is stored and used to respond to user queries. When a query is received, the system searches the indexed information and generates an appropriate response.

✔ **Prerequisites:**

- Ollama.

- Notebook.

- LlamIndex.

- Docker.

 **Usage:**

- Install the libraries:  `pip install -r requirement.txt`

- Install the used llm: `ollama install *llm*`

- Start the chroma db `docker compose up`

- Run it with `python ./src/app.py`

**LlamaIndex:**

LlamaIndex is a data framework specifically designed to work with large language models. In our project, it enhances the system by analyzing files and locating text based on the semantic meaning of query terms, rather than relying on simple keyword matching. This approach increases the accuracy and relevance of the responses, improving the chances of getting the correct answer.

**Links:**

Ollama:

https://www.ollama.com/

LlamaIndex:

https://www.llamaindex.ai/

Elastic Search:

https://www.elastic.co/
