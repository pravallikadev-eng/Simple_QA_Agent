from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain.prompts import ChatPromptTemplate
#from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def main():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key="AIzaSyBK-XrmqhzmeLQFozBJDDuVy_3WGpHDGZE", temperature=0.9)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("human","{question}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({"question": "What's the meaning of life to a 30 year old married girl.Explain it simply in few words?"})
    print(response)

if __name__ == "__main__":
    main()