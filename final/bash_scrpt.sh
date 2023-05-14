#!/bin/bash

echo 'we are inside the script!'

sleep 45

if mysql -u root -p9501 -h mariadb_db -e "select * from baseball.final_baseball limit 2;"
then
  echo 'the database exists. we are gonna load our file from hw_02'
  mysql -u root -p9501 -h mariadb_db  baseball < hw_05.sql
  echo 'the script ran successfully'

else
  echo 'the database does not exists. we are gonna load the database first'
  mysql -u root -p9501 -h mariadb_db  baseball < baseball.sql
  echo 'the database exists. we are gonna load our file from hw_02'
  mysql -u root -p9501 -h mariadb_db  baseball < hw_05.sql
  echo 'the script ran successfully'
fi

echo 'running the python code'

python3 hw_05.py

