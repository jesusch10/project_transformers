from dotenv import load_dotenv
from PyPDF2 import PdfReader
import oracledb
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_community.llms import OCIGenAI
from langchain.llms import Cohere
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import oraclevs
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.embeddings import OCIGenAIEmbeddings


########## VARIABLES ##########


load_dotenv()
username = " "
password = " "
dsn = ''' '''
COMPARTMENT_OCID = " "
PATH_TO_PDF = " "
user_question = (" ")
YOUR_MODEL_ID = " "
YOUR_SERVICE_ENDPOINT = " "
TABLE_NAME = " " # to create a new table to store data extracted from the document
TEMPERATURE = 0.7
COHERE_API_KEY = " "


########## FUNCTIONS ##########


def chunks_to_docs_wrapper(row: dict) -> Document:
    """
    Converts text into a Document object suitable for ingestion into Oracle Vector Store.
    - row (dict): A dictionary representing a row of data with keys for 'id', 'link', and 'text'.
    """
    metadata = {'id': str(row['id']), 'link': row['link']}
    print(metadata)
    return Document(page_content=row['text'], metadata=metadata)


########## RAG ##########


# 1. Loading database and transforming the document to text

print(f"Database user name: {username} | Database connection information: {dsn}")
try: 
    conn23c = oracledb.connect(user=username, password=password, dsn=dsn)
    print("Connection successful!")
except Exception as e:
    print("Connection failed!")

try:
    pdf = PdfReader(PATH_TO_PDF)
    print(f"Document loaded! {len(pdf.pages)} pages. First page:")
    print(pdf.pages[0].extract_text())
    text = "".join([page.extract_text() for page in pdf.pages])
    print("PDF document transformed to text format")
except FileNotFoundError:
    print("Error: doc.pdf not found!")
    exit(1)

# 2. Chunking the text document
text_splitter = CharacterTextSplitter(separator="\n",chunk_size=800,chunk_overlap=100,length_function=len)
chunks = text_splitter.split_text(text)
print(f"Text chunked. Example: {chunks[0]}")
docs = [chunks_to_docs_wrapper({'id': page_num, 'link': f'Page {page_num}', 'text': text}) for page_num, text in enumerate(chunks)]
print("Created metadata wrapper with the chunks")

# 3. Embedding chunks as vectors into Oracle Database 23ai.
model_4db = OCIGenAIEmbeddings(model_id=YOUR_MODEL_ID, service_endpoint=YOUR_SERVICE_ENDPOINT, compartment_id=COMPARTMENT_OCID)
knowledge_base = OracleVS.from_documents(docs, model_4db, client=conn23c, table_name=TABLE_NAME, distance_strategy=DistanceStrategy.DOT_PRODUCT)     
print("Chunks have been embedded as vectors in the database")

# 4. Building the query prompt
cohere_api_key = COHERE_API_KEY
llmOCI = Cohere(
    model="command", 
    cohere_api_key=cohere_api_key, 
    max_tokens=1000, 
    temperature=TEMPERATURE
)
template = """Answer the question based only on the following context:
            {context} Question: {user_question}"""
prompt = PromptTemplate.from_template(template)
retriever = knowledge_base.as_retriever()

# 5. Chaining the entire process together
print("Sending the prompt and RAG context to the LLM...")
chain = (
  {"context": retriever, "user_question": RunnablePassthrough()}
     | prompt
     | llmOCI
     | StrOutputParser()
)
response = chain.invoke(user_question)
print(f"User question: {user_question}")
print(f"Response: {response}")
