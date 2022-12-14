from typing import Union
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Request, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from linear_regression import generate_linear_regression
from confusion_matrix import generate_confusion_matrix
from scatter import generate_scatter
import os
import time
import asyncio



app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None



# async def remove_graph(filename):
#     time.sleep(60 * 60 * 24)
#     os.remove(filename)



@app.get("/")
def read_root(request: Request):
    print(os.listdir("static"))
    return {"URL": request.base_url._url}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}


@app.get("/static")
def list_assets(request: Request):
    assets = os.listdir("static")
    response = []
    for a in assets:
        response.append(f"{request.url._url}/{a}")

    return response


@app.post("/regression/linear")
def linear_regression(request: Request, file: UploadFile = File(...), det: str = "", non_det: str = ""):
    df = pd.read_csv(file.file)
    filename = generate_linear_regression(df, det, non_det)
    file.file.close()
    return f"{request.base_url._url}{filename}"


@app.post("/scatter")
async def scatter(request: Request,  file: UploadFile = File(...), columns: Union[str, None]= None):
    df = pd.read_csv(file.file)
    filename = generate_scatter(df, columns)
    file.file.close()
    return f"{request.base_url._url}{filename}"


@app.post("/confusion/matrix")
def confusion_matrix(request: Request, file: UploadFile = File(...), target: str = "", result: str = "", label: Union[str, None] = None):
    df = pd.read_csv(file.file)
    filename = generate_confusion_matrix(df, target, result)
    file.file.close()
    return f"{request.base_url._url}{filename}"
