FROM mariadb:latest

WORKDIR /app

RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
     build-essential \
  &&  apt-get install mariadb-client --yes \
  && rm -rf /var/lib/apt/lists/*

COPY baseball.sql   .
COPY hw_02.sql   .
COPY bash_scrpt.sh .

RUN chmod +x bash_scrpt.sh

COPY records.csv   .