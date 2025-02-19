FROM python:3.11
WORKDIR /app
COPY ./requirements.txt /app
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY . /app
CMD ["python", "main.py"]