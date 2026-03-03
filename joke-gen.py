from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os   

load_dotenv()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Tell me a joke about {topic}."),
    ]
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)   

chain = prompt | llm | StrOutputParser()   

app = FastAPI(title="Joke Generator API", description="An API that generates jokes based on a given topic.", version="1.0")

#add_routes(app, chain,path="/chat")

class JokeRequest(BaseModel):
    topic:str

@app.get("/chat")
async def generate_joke(request: JokeRequest):
    try:
        result = await chain.ainvoke({"topic": request.topic})
        return {"joke": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message" : "Welcome to JokeGenAI"
    }

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)