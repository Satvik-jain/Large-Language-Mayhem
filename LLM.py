from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from huggingface_hub import login
import signal
from functools import wraps

load_dotenv()

# Only login if API key exists
if "HUGGING_FACE_API_KEY" in os.environ:
    try:
        login(token=os.environ["HUGGING_FACE_API_KEY"])
    except:
        pass
os.environ['CURL_CA_BUNDLE'] = ''

load_dotenv()

# Timeout decorator
def timeout_handler(signum, frame):
    raise TimeoutError("API call timed out")

def with_timeout(seconds=30):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Set signal handler for timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            except TimeoutError:
                return f"Error: Request timed out after {seconds} seconds. The API might be slow or unavailable."
            finally:
                signal.alarm(0)  # Cancel the alarm
            return result
        return wrapper
    return decorator

class Bot():
    def __init__(self):
        self.groq_models = ['gemma-7b-it', 'llama3-70b-8192',\
                            'llama3-8b-8192', 'mixtral-8x7b-32768']
        self.hf_models = ["01-ai/Yi-1.5-34B-Chat", "google/gemma-1.1-2b-it",\
                          "google/gemma-1.1-7b-it"]
        self.google_models = ["gemini-pro", "gemini-1.5-flash"]
        self.models = ["gemini-pro", "gemini-1.5-flash", "01-ai/Yi-1.5-34B-Chat", "google/gemma-1.1-2b-it",\
                          "google/gemma-1.1-7b-it", 'gemma-7b-it', 'llama3-70b-8192', 'llama3-8b-8192', 'mixtral-8x7b-32768']

    def call_groq(self, model, temp = 0.7, given_prompt = "Hi"):
        try:
            if "GROQ_API_KEY" not in os.environ:
                return "Error: GROQ_API_KEY not found in environment variables. Please set it in your .env file or Hugging Face Space secrets."
            
            llm = ChatGroq(
                temperature=temp,
                model=model,
                timeout=30,
                max_retries=2
            )

            system = "You are a helpful assistant."
            human = "{text}"
            prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

            chain = prompt | llm | StrOutputParser()
            result = chain.invoke({"text": given_prompt})
            return result if result else "Error: Empty response from model"

        except TimeoutError as e:
            return f"Error: Request timed out - {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"

    def call_hf(self, model, temp = 0.7, given_prompt = "Hi"):
        try:
            if "HUGGING_FACE_API_KEY" not in os.environ:
                return "Error: HUGGING_FACE_API_KEY not found in environment variables. Please set it in your .env file or Hugging Face Space secrets."
            
            llm = HuggingFaceEndpoint(
                repo_id=model,
                temperature=temp,
                timeout=60,
                max_retries=2
            )

            chat = ChatHuggingFace(llm=llm, verbose=False)

            template = """
                You are a helpful assistant

                User: {query}

                Answer:
            """

            prompt = PromptTemplate(
                template=template,
                input_variables=["query"]
            )

            chain = prompt | chat | StrOutputParser()
            result = chain.invoke({"query": given_prompt})
            return result if result else "Error: Empty response from model"

        except TimeoutError as e:
            return f"Error: Request timed out - {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"

    def call_google(self, model, temp=0.7, given_prompt = "Hi"):
        try:
            if "GOOGLE_API_KEY" not in os.environ:
                return "Error: GOOGLE_API_KEY not found in environment variables. Please set it in your .env file or Hugging Face Space secrets."
            
            llm = ChatGoogleGenerativeAI(
                model=model, 
                temperature=temp,
                timeout=30,
                max_retries=1
            )
            system = "You are a helpful assistant."
            human = "{text}"
            prompt = ChatPromptTemplate.from_messages([("human", human)])
            chain = prompt | llm | StrOutputParser()
            result = chain.invoke({"text": given_prompt})
            return result if result else "Error: Empty response from model"
        except TimeoutError as e:
            return f"Error: Request timed out - {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"

    def response(self, model, prompt="Hi", temperature = 0.7):
        if model in self.groq_models:
            res_show = self.call_groq(temp = temperature, given_prompt = prompt, model= model)
        elif model in self.hf_models:
            res_show = self.call_hf(given_prompt = prompt, temp = temperature, model = model)
        elif model in self.google_models:
            res_show = self.call_google(given_prompt = prompt, temp = temperature, model = model)
        else:
            return "Sorry! App not working properly"
        return res_show





