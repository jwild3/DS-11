### DS-11

👏 **DocuRAG**

Our objective is to create a chatbot model that can read and interpret the data from all the PDF, excel(still in progress) and word documents that would be provided to it from the manufacturing department and is to allow the chatbot to respond to highly specific questions related to the company's products without getting puzzled quickly and effecienty.

![Capture213](https://github.com/jwild3/DS-11/assets/169072725/1b9140f9-1b35-4c4e-8dda-0ab90d13d39b)

source: https://www.leewayhertz.com/llamaindex/


The Above-Mentioned diagram explains the functionality of our project as our model will get the files in PDF format and will extract the relevant information from the files and will do the indexing so that it would be easier for the model to find the specific information and the indexed data will be stored to be used in exchange of a user's query and the response will be generated.

✔ **Installations:**

- Ollama.

- Notebook.

- LlamIndex.

- Docker.

 **Process:**

- Install the library:  `pip install -r requirement.txt`

- `docker compose up`

- Run it with `python ./src/app.py`

- First offers data connectors to ingest your existing data sources and data formats (APIs, PDFs, docs, SQL, etc.). This explains the usage of elastic search as a database.

- Can easily connect the OpenAI.


**ELASTIC SEARCH:**

Elastic search basically used to store strings etc at the very beginning and we are using it to store vector embeddings (a vector embedding, often just called an embedding, is a numerical representation of the semantics, or meaning of your text. Two pieces of text with similar meanings will have mathematically similar embeddings, even if the actual text is quite different). So in this database it will store the PDF files that will contain the detailed information about a product and while responding to a customer's query, our system will go through all the PDF files and will compare the results and will respond to the user's query for the better understanding.

**LlamaIndex:**

LlamaIndex is a data framework specifically designed to work with large language models, including Ollama. It makes the data structured. In our project it will work in a way that it will go through the files and locate text that is related to the meaning of the query terms rather than simple keyword matching this will give increase the ratio of getting the correct answer.

**Links:**

Ollama:

https://www.ollama.com/

LlamaIndex:

https://www.llamaindex.ai/

Elastic Search:

https://www.elastic.co/
