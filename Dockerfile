FROM tiangolo/uvicorn-gunicorn:python3.8

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ./models /app/models
COPY ./main.py /app/main.py 

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
