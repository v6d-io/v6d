drop table if exists hive_example;
create table hive_example(field_1 string,field_2 int);

insert into hive_example values('a', 1), ('b', 2), ('c', 3);

insert overwrite directory '/tmp/out/test_insert/'
row format delimited fields terminated by ','
select * from hive_example;

drop table hive_example;
