drop table if exists hive_static_partition;

create table hive_static_partition(
    src_id int,
    dst_id int
) partitioned by (value int);
insert into table hive_static_partition partition(value=666) values (3, 4);
insert into table hive_static_partition partition(value=666) values (999, 2), (999, 2), (999, 2);
insert into table hive_static_partition partition(value=114514) values (1, 2);

drop table if exists result;
create table result(
    field_1 int,
    field_2 int,
    field_3 int
);
insert into result
select * from hive_static_partition
union all
select * from hive_static_partition where value=666
union all
select * from hive_static_partition where value=114514;

insert overwrite directory 'file:///tmp/out/test_hive_static_partition/'
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
select * from result
order by field_1 asc;

drop table hive_static_partition;
drop table result;
