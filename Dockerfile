FROM python:3.7-slim

WORKDIR /app

RUN pip install -U scikit-learn numpy

ADD boston.py .

ENTRYPOINT ["python", "boston.py"]