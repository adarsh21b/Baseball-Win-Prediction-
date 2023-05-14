FROM ubuntu:kinetic

WORKDIR /app

RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
     build-essential \
     python3 \
     python3-pip \
     python3-dev \
  &&  pip install --upgrade pip \
  &&  apt-get install -y libmariadb-dev-compat libmariadb-dev --yes \
  &&  apt-get install mariadb-client --yes \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip3 install --compile --no-cache-dir -r requirements.txt

COPY bash_scrpt.sh .
RUN chmod +x bash_scrpt.sh

