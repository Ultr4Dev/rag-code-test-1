import os
import openai
import pandas as pd
from fastapi import FastAPI, Body, Depends, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai.embeddings_utils import get_embedding, cosine_similarity
from databases import Database
import json
import numpy as np
import tiktoken
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
openai.api_key = "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"
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/views", StaticFiles(directory="views"), name="views")
@app.get("/")
async def read_index():
    return FileResponse('panel.html')
@app.get("/add")
async def read_index2():
    return FileResponse('add.html')
@app.get("/templater.js")
async def read_index3():
    return FileResponse('templater.js')

database = Database("sqlite:///code.db")
@app.on_event("startup")
async def startup():
    await database.connect()
    # Call modified table creation function
    await create_table_with_filenames()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()
def num_tokens_from_messages(messages, model="gpt-3.5-turbo"):
  """Returns the number of tokens used by a list of messages."""
  try:
      encoding = tiktoken.encoding_for_model(model)
  except KeyError:
      encoding = tiktoken.get_encoding("cl100k_base")
  if model == "gpt-3.5-turbo":  # note: future models may deviate from this
      num_tokens = 0
      for message in messages:
          num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
          for key, value in message.items():
              num_tokens += len(encoding.encode(value))
              if key == "name":  # if there's a name, the role is omitted
                  num_tokens += -1  # role is always required and always 1 token
      num_tokens += 2  # every reply is primed with <im_start>assistant
      return num_tokens
  else:
      raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
  See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
# Modified table creation function
async def create_table_with_filenames():
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

# Modified add_code function
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
    values = {"filename": code_file.filename, "code": description, "embedding": embedding_str}
    await database.execute(query, values)

    path = os.path.join("static", code_file.filename)
    with open(path, "wb") as f:
        f.write(code_bytes)

    return JSONResponse(content={"message": "Code added successfully", "file_link": f"/static/{code_file.filename}"})
@app.post("/code/add/raw")
async def add_code(code:Annotated[str, Form()]="", filename:Annotated[str, Form()]=""):
    print(filename)
    fileName = filename
    if fileName== ""or code== "":
        return {"message":"Invalid file name"}, 500
    code_str = code
    description = generate_description(code_str)
    embedding = get_embedding(description, engine='text-embedding-ada-002')
    embedding_str = json.dumps(embedding)

    query = """
        INSERT INTO code (filename, code, embedding)
        VALUES (:filename, :code, :embedding)
        ON CONFLICT (filename) DO UPDATE SET code = :code, embedding = :embedding
    """
    values = {"filename": fileName, "code": description, "embedding": embedding_str}
    await database.execute(query, values)

    path = os.path.join("static", fileName)
    with open(path, "wb") as f:
        f.write(code_str.encode())

    return JSONResponse(content={"message": "Code added successfully", "file_link": f"/static/{fileName}"})
# Modified search_code function
@app.get("/code/search/")
async def search_code(code_description: str, n: int = 3):
    query = """
        SELECT filename, code, embedding 
        FROM code
    """
    values = await database.fetch_all(query)

    code_df = pd.DataFrame(values, columns=['filename', 'code', 'embedding'])
    code_df['embedding'] = code_df['embedding'].apply(json.loads)

    embedding = get_embedding(code_description, engine='text-embedding-ada-002')
    code_df['similarities'] = code_df.embedding.apply(lambda x: cosine_similarity(np.array(x), embedding))

    res = code_df.sort_values('similarities', ascending=False).head(n)
    res["file_link"] = "/static/" + res["filename"]
    data = res[["file_link", "similarities", "code"]].to_dict(orient='records')
  
    return JSONResponse(content=data)

# Function to generate description using code
def generate_description(code: str) -> str:
    prompt = f"Translate the following code into a human-readable description:\n\n{code}\n\n"
    msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    tokens = num_tokens_from_messages(msgs)+512
    cost = tokens/1000.0*0.004  # The cost per tokens is $20.0 per 4096 tokens for the 'gpt-3.5-turbo' model
    print("Price:", "$"+str(round(cost)))

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=msgs,
        max_tokens=512
    )
    
    description = response.choices[0].message['content'].strip()
    return description
