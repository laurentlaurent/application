FROM ubuntu:22.04
<<<<<<< HEAD

# Install Python
RUN apt-get -y update && \
    apt-get install -y python3-pip

# Install project dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src
COPY train.py .
COPY app ./app

CMD ["bash", "-c", "./app/run.sh"]
=======
# Install Python
RUN apt-get -y update && \
    apt-get install -y python3-pip
# Install project dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY main.py .
COPY src ./src
CMD ["python3", "main.py"]
>>>>>>> 1df1704ada53842583cb4c9130d24a4796f7d198
