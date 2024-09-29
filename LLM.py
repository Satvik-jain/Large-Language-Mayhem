from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain.schema.output_parser import StrOutputParser
from langchain_huggingface import ChatHuggingFace
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from huggingface_hub import login

load_dotenv()

login(token=os.environ["HUGGING_FACE_API_KEY"])
os.environ['CURL_CA_BUNDLE'] = ''

load_dotenv()

class Bot():
    def __init__(self):
        self.groq_models = ['gemma-7b-it', 'llama3-70b-8192',\
                            'llama3-8b-8192', 'mixtral-8x7b-32768']
        self.hf_models = ["01-ai/Yi-1.5-34B-Chat", "google/gemma-1.1-2b-it",\
                          "google/gemma-1.1-7b-it"]
        self.google_models = ["gemini-1.0-pro", "gemini-1.5-flash",\
                              "gemini-1.5-pro"]
        self.models = ["gemini-1.0-pro", "gemini-1.5-flash", "gemini-1.5-pro", "01-ai/Yi-1.5-34B-Chat", "google/gemma-1.1-2b-it",\
                          "google/gemma-1.1-7b-it", 'gemma-7b-it', 'llama3-70b-8192', 'llama3-8b-8192', 'mixtral-8x7b-32768']

    def call_groq(self, model, temp = 0.7, given_prompt = "Hi"):
        try:
            llm = ChatGroq(
                temperature=temp,
                model= model
            )

            system = "You are a helpful assistant."
            human = "{text}"
            prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

            chain = prompt | llm | StrOutputParser()
            return chain.invoke({"text": given_prompt})

        except Exception as e:
            return f"Error: {str(e)}"

    def call_hf(self,model, temp = 0.7, given_prompt = "Hi"):
        try:
            llm = HuggingFaceEndpoint(
                repo_id=model,
                temperature=temp
                )

            chat = ChatHuggingFace(llm=llm, verbose=True)

            template = """
                You are a helpful assistant

                User: {query}

                Answer:
            """

            prompt = PromptTemplate(
                template=template,
                input_variables=["query"]
            )

            chain =prompt | chat | StrOutputParser()

            return chain.invoke({"query": given_prompt})

        except Exception as e:
            return f"Error: {str(e)}"

    def call_google(self,model, temp=0.7, given_prompt = "Hi"):
        try:
            model = ChatGoogleGenerativeAI(model = model, temprature = temp)
            system = "You are a helpful assistant."
            human = "{text}"
            prompt = ChatPromptTemplate.from_messages([("human", human)])
            chain = prompt | model | StrOutputParser()
            return chain.invoke({"text": given_prompt})
        except Exception as e:
            return f"Error: {str(e)}"

    def response(self, model, prompt="Hi", temprature = 0.7):
        if model in self.groq_models:
            res_show = self.call_groq(temp = temprature, given_prompt = prompt, model= model)
        elif model in self.hf_models:
            res_show = self.call_hf(given_prompt = prompt, temp = temprature, model = model)
        elif model in self.google_models:
            res_show = self.call_google(given_prompt = prompt, temp = temprature, model = model)
        else:
            return "Sorry! App not working properly"
        return res_show





