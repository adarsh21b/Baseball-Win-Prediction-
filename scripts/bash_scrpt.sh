#!/bin/bash

echo 'we are inside the script!'

sleep 10

if mysql -u root -p9501 -h mariadb_db -e "select * from baseball.pitcher_counts limit 5;"
then
  echo 'the database exists. Next we will load hw_02 sql file'
  mysql -u root -p9501 -h mariadb_db  baseball < hw_02.sql
  echo 'the output needs to be stored in csv now'
  mysql -u root -p9501 -h mariadb_db  -e "select * from baseball.Rolling_AVG_100_day;" > records.csv
  echo 'records recorded in records.csv'

else
  echo 'the database does not exists. we are gonna load the database first'
  mysql -u root -p9501 -h mariadb_db  baseball < baseball.sql
  echo 'baseball db loaded. gonna run the hw_02 file'
  mysql -u root -p9501 -h mariadb_db  baseball < hw_02.sql
  echo 'the output needs to be stored in csv now'
  mysql -u root -p9501 -h mariadb_db  -e "select * from baseball.Rolling_AVG_100_day;" > records.csv
  echo 'records recorded in records.csv'

fi



