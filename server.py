import os
import json
from typing import Annotated
import openai
import pandas as pd
import numpy as np
import tiktoken
from fastapi import FastAPI, Body, Depends, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from databases import Database
from openai.embeddings_utils import get_embedding, cosine_similarity

app = FastAPI()
openai.api_key = os.environ.get("OPENAI_API_KEY")
# CORS settings
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static and views directories
app.mount("./static", StaticFiles(directory="static"), name="static")
app.mount("./views", StaticFiles(directory="views"), name="views")


@app.get("/")
async def read_index():
    return FileResponse('panel.html')


@app.get("/add")
async def read_index2():
    return FileResponse('add.html')


@app.get("/templater.js")
async def read_index3():
    return FileResponse('templater.js')
# Database initialization
database = Database("sqlite:///code.db")


async def create_table_with_filenames():
    await database.connect()
    await database.execute(
        """
        CREATE TABLE IF NOT EXISTS code (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE,
            code TEXT,
            embedding TEXT
        )
        """
    )
    await database.disconnect()


def num_tokens_from_messages(messages, model="gpt-3.5-turbo"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        num_tokens = 0
        for message in messages:
            num_tokens += 4
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += -1
        num_tokens += 2
        return num_tokens
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not presently implemented for model {model}.""")


@app.post("/code/add/")
async def add_code(code_file: UploadFile = File(...)):
    code_bytes = await code_file.read()
    code_str = code_bytes.decode()
    description = generate_description(code_str)
    embedding = get_embedding(description, engine='text-embedding-ada-002')
    embedding_str = json.dumps(embedding)

    query = """
        INSERT INTO code (filename, code, embedding)
        VALUES (:filename, :code, :embedding)
        ON CONFLICT (filename) DO UPDATE SET code = :code, embedding = :embedding
    """
    values = {"filename": code_file.filename,
              "code": description, "embedding": embedding_str}
    await database.execute(query, values)

    path = os.path.join("static", code_file.filename)
    with open(path, "wb") as f:
        f.write(code_bytes)

    return JSONResponse(content={"message": "Code added successfully", "file_link": f"/static/{code_file.filename}"})


@app.post("/code/add/raw")
async def add_code(code: Annotated[str, Form()] = "", filename: Annotated[str, Form()] = ""):
    fileName = filename
    if fileName == "" or code == "":
        return {"message": "Invalid file name"}, 500
    code_str = code
    description = generate_description(code_str)
    embedding = get_embedding(description, engine='text-embedding-ada-002')
    embedding_str = json.dumps(embedding)

    query = """
        INSERT INTO code (filename, code, embedding)
        VALUES (:filename, :code, :embedding)
        ON CONFLICT (filename) DO UPDATE SET code = :code, embedding = :embedding
    """
    values = {"filename": fileName,
              "code": description, "embedding": embedding_str}
    await database.execute(query, values)

    path = os.path.join("static", fileName)
    with open(path, "wb") as f:
        f.write(code_str.encode())

    return JSONResponse(content={"message": "Code added successfully", "file_link": f"/static/{fileName}"})


@app.get("/code/search/")
async def search_code(code_description: str, n: int = 3):
    query = """
        SELECT filename, code, embedding 
        FROM code
    """
    values = await database.fetch_all(query)

    code_df = pd.DataFrame(values, columns=['filename', 'code', 'embedding'])
    code_df['embedding'] = code_df['embedding'].apply(json.loads)

    embedding = get_embedding(
        code_description, engine='text-embedding-ada-002')
    code_df['similarities'] = code_df.embedding.apply(
        lambda x: cosine_similarity(np.array(x), embedding))

    res = code_df.sort_values('similarities', ascending=False).head(n)
    res["file_link"] = "/static/" + res["filename"]
    data = res[["file_link", "similarities", "code"]].to_dict(orient='records')

    return JSONResponse(content=data)


def generate_description(code: str) -> str:
    prompt = f"Translate the following code into a human-readable description:\n\n{code}\n\n"
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    tokens = num_tokens_from_messages(msgs) + 512
    cost = tokens / 1000.0 * 0.004

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=msgs,
        max_tokens=512
    )

    description = response.choices[0].message['content'].strip()
    return description


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0")
