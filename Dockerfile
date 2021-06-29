FROM jinaai/jina:2.0.0rc8

# install git
RUN apt-get -y update && apt-get install -y git && apt-get install libgomp1

# install requirements before copying the workspace
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt

# setup the workspace
COPY . /workspace
WORKDIR /workspace

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
