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
        # Groq models (2024-2025) - Verified available models
        self.groq_models = [
            # Llama models (verified on Groq)
            'llama-3.1-8b-instant',
            'llama-3.3-70b-versatile',
            # Qwen models (verified on Groq)
            'qwen/qwen3-32b',
            # OpenAI models (verified on Groq)
            'openai/gpt-oss-120b',
            'openai/gpt-oss-20b'
        ]
        # Hugging Face models - Using correct repo IDs from model cards
        self.hf_models = [
            # DeepSeek models
            'deepseek-ai/DeepSeek-V3',
            'deepseek-ai/DeepSeek-R1',
            'deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct',
            # Qwen models
            'Qwen/Qwen2.5-7B-Instruct',
            'Qwen/Qwen2.5-14B-Instruct',
            'Qwen/Qwen2.5-32B-Instruct',
            'Qwen/Qwen2.5-72B-Instruct',
            # Gemma models
            'google/gemma-2-9b-it',
            'google/gemma-2-27b-it',
            # Mistral models
            'mistralai/Mistral-7B-Instruct-v0.3',
            'mistralai/Mixtral-8x7B-Instruct-v0.1'
        ]
        # Google Gemini models (2024-2025) - using Google AI Studio models
        self.google_models = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-1.5-pro"]
        # Combined list of all models (17 total)
        self.models = [
            # Google Gemini (3 models)
            "gemini-2.5-flash", 
            "gemini-2.5-flash-lite",
            "gemini-1.5-pro",
            # Groq models (5 models)
            'llama-3.1-8b-instant',
            'llama-3.3-70b-versatile',
            'qwen/qwen3-32b',
            'openai/gpt-oss-120b',
            'openai/gpt-oss-20b',
            # Hugging Face models (10 models)
            'deepseek-ai/DeepSeek-V3',
            'deepseek-ai/DeepSeek-R1',
            'deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct',
            'Qwen/Qwen2.5-7B-Instruct',
            'Qwen/Qwen2.5-14B-Instruct',
            'Qwen/Qwen2.5-32B-Instruct',
            'Qwen/Qwen2.5-72B-Instruct',
            'google/gemma-2-9b-it',
            'google/gemma-2-27b-it',
            'mistralai/Mistral-7B-Instruct-v0.3',
            'mistralai/Mixtral-8x7B-Instruct-v0.1'
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





