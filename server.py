import glob
import os
import json
import pandas as pd
import numpy as np
import requests
import tiktoken
import pika
from typing import Annotated
from fastapi import FastAPI, Body, Depends, UploadFile, File, Form
from fastapi.concurrency import asynccontextmanager
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from databases import Database
from openai.embeddings_utils import get_embedding, cosine_similarity
import openai

# Define RabbitMQ connection parameters
rabbitmq_host = os.environ.get("MSG_PRCCS_ADDR", "hitler.ultr4.io")
rabbitmq_queue = 'code_generation_requests'
credentials = pika.PlainCredentials("codegen_user", "codegen_pass")

# Create a connection to RabbitMQ


# Global variable for queue


@asynccontextmanager
async def lifespan(app: FastAPI):
    await database.connect()

    await create_table_with_filenames()
    yield
    await database.disconnect()


app = FastAPI(lifespan=lifespan)
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
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/views", StaticFiles(directory="views"), name="views")


@app.get("/")
async def read_index():
    return FileResponse('panel.html')


@app.post("/generate_code/")
async def generate_code(prompt: str):
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=rabbitmq_host, credentials=credentials, virtual_host="/"))
    channel = connection.channel()
    # Generate code as before
    channel.queue_declare(queue=rabbitmq_queue, durable=True)
    # Publish the generated code to RabbitMQ
    channel.basic_publish(
        exchange='',
        routing_key=rabbitmq_queue,
        body=json.dumps({"prompt": prompt}),
    )
    connection.close()
    response = requests.get(
        f'http://{credentials.username}:{credentials.password}@{rabbitmq_host}:15672/api/queues/%2f/{rabbitmq_queue}')
    data = response.json()
    print('The queue has {0} messages'.format(data['messages']))
    # Close the connection
    return {"message": "Code generation request has been added to the queue.", "queue_position": data['messages']}


@app.get("/queue/size")
async def get_queue_size():
    # Get the size of the RabbitMQ queue
    response = requests.get(
        f'http://{credentials.username}:{credentials.password}@{rabbitmq_host}:15672/api/queues/%2f/{rabbitmq_queue}')
    data = response.json()
    print('The queue has {0} messages'.format(data['messages']))

    # Close the connection
    return {"queue": data['messages']}


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
    # Returns the number of tokens used by a list of messages
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
async def add_raw_code(code: Annotated[str, Form()] = "", filename: Annotated[str, Form()] = ""):
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
    uvicorn.run("server:app", host="0.0.0.0", port=1337)
