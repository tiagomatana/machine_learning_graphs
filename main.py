from typing import Union
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.staticfiles import StaticFiles
from linear_regression import generate_linear_regression
from confusion_matrix import generate_confusion_matrix
from scatter import generate_scatter
import os


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/static")
def list_assets(request: Request):
    assets = os.listdir("static")
    response = []
    for a in assets:
        response.append(f"{request.url}/{a}")

    return response


@app.post("/regression/linear")
def linear_regression(request: Request, file: UploadFile = File(...), det: str = "", non_det: str = ""):
    df = pd.read_csv(file.file)
    filename = generate_linear_regression(df, det, non_det)
    file.file.close()
    return f"{request.base_url}{filename}"


@app.post("/scatter")
async def scatter(request: Request,  file: UploadFile = File(...), columns: Union[str, None]= None):
    df = pd.read_csv(file.file)
    filename = generate_scatter(df, columns)
    file.file.close()
    return f"{request.base_url}{filename}"


@app.post("/confusion/matrix")
def confusion_matrix(request: Request, file: UploadFile = File(...), target: str = "", result: str = "", label: Union[str, None] = None):
    df = pd.read_csv(file.file)
    filename = generate_confusion_matrix(df, target, result)
    file.file.close()
    return f"{request.base_url}{filename}"
