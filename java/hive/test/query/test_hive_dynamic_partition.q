drop table if exists hive_dynamic_partition_data;
create table hive_dynamic_partition_data(
    src_id int,
    dst_id int,
    year int)
stored as TEXTFILE
location "file:///tmp/hive_dynamic_partition_data";
insert into table hive_dynamic_partition_data values (1, 2, 2018),(3, 4, 2018),(2, 3, 2017);

drop table if exists hive_dynamic_partition_test;
create table hive_dynamic_partition_test
(
    src_id int,
    dst_id int
)partitioned by(mounth int, year int);
insert into table hive_dynamic_partition_test partition(mounth=1, year) select src_id,dst_id,year from hive_dynamic_partition_data;

insert overwrite directory 'file:///tmp/out/test_hive_dynamic_partition/'
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
select * from hive_dynamic_partition_test
order by src_id asc;

drop table hive_dynamic_partition_test;
drop table hive_dynamic_partition_data;