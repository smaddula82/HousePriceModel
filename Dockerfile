FROM python:latest
WORKDIR /root/Projects/Docker
COPY . .
RUN pip install --upgrade pip
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
CMD ["python","/root/Projects/Docker/ModelPredictor.py"]