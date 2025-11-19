from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from huggingface_hub import login
load_dotenv()

# Only login if API key exists
if "HUGGING_FACE_API_KEY" in os.environ:
    try:
        login(token=os.environ["HUGGING_FACE_API_KEY"])
    except:
        pass
os.environ['CURL_CA_BUNDLE'] = ''

load_dotenv()

class Bot():
    def __init__(self):
        # Updated Groq models (2024-2025)
        self.groq_models = ['llama-3.1-8b-instant', 'llama-3.3-70b-versatile']
        # Hugging Face models
        self.hf_models = ["01-ai/Yi-1.5-34B-Chat", "google/gemma-1.1-2b-it",\
                          "google/gemma-1.1-7b-it"]
        # Updated Google Gemini models (2024-2025) - using Google AI Studio models
        # Using gemini-2.5-flash and gemini-2.5-flash-lite as specified
        # If these exact names don't work, try: gemini-2.0-flash-exp, gemini-1.5-flash, or gemini-1.5-pro
        self.google_models = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
        # Combined list of all models
        self.models = ["gemini-2.5-flash", "gemini-2.5-flash-lite", 
                      "01-ai/Yi-1.5-34B-Chat", "google/gemma-1.1-2b-it", "google/gemma-1.1-7b-it", 
                      'llama-3.1-8b-instant', 'llama-3.3-70b-versatile']

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





