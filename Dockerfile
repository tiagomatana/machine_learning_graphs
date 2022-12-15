FROM python:3.9
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY ./main.py /code/main.py
COPY ./linear_regression.py /code/linear_regression.py
COPY ./scatter.py /code/scatter.py
COPY ./confusion_matrix.py /code/confusion_matrix.py
RUN mkdir -p /code/static
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--reload"]
