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
        # Groq models (2024-2025) - Various providers
        self.groq_models = [
            # Llama models
            'llama-3.1-8b-instant',
            'llama-3.3-70b-versatile',
            # DeepSeek models
            'deepseek-r1-distill-llama-8b',
            'deepseek-r1-distill-qwen-7b',
            'deepseek-chat',
            # Qwen models
            'qwen-2.5-7b-instruct',
            'qwen-2.5-14b-instruct',
            'qwen-2.5-32b-instruct',
            'qwen-2.5-72b-instruct',
            # Gemma models
            'gemma-2-9b-it',
            'gemma-2-27b-it',
            # Mistral models
            'mistral-large-2407',
            'pi-3-mini',
            'pi-3'
        ]
        # Hugging Face models - Removed as requested
        self.hf_models = []
        # Google Gemini models (2024-2025) - using Google AI Studio models
        self.google_models = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-1.5-pro"]
        # Combined list of all models (15 total)
        self.models = [
            # Google Gemini (3 models)
            "gemini-2.5-flash", 
            "gemini-2.5-flash-lite",
            "gemini-1.5-pro",
            # Groq models (12 models)
            'llama-3.1-8b-instant',
            'llama-3.3-70b-versatile',
            'deepseek-r1-distill-llama-8b',
            'deepseek-r1-distill-qwen-7b',
            'deepseek-chat',
            'qwen-2.5-7b-instruct',
            'qwen-2.5-14b-instruct',
            'qwen-2.5-32b-instruct',
            'qwen-2.5-72b-instruct',
            'gemma-2-9b-it',
            'gemma-2-27b-it',
            'mistral-large-2407',
            'pi-3-mini',
            'pi-3'
        ]

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





