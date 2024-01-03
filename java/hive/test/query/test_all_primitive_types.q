drop table if exists test_all_primitive_types;
create table test_all_primitive_types (
    field_1 tinyint,
    field_2 smallint,
    field_3 bigint,
    field_4 int,
    field_5 double,
    field_6 float,
    field_7 string,
    field_9 varchar(20),
    field_10 char(20),
    field_8 binary,
    field_11 date,
    field_12 boolean,
    field_13 timestamp,
    field_14 decimal(6, 2)
);

insert into test_all_primitive_types select
    tinyint(1),
    smallint(1),
    42,
    bigint(1),
    double(2.0),
    float(1.0),
    'hello world1!',
    'hello world2!',
    'hello world3!',
    cast('hello world4!' as binary),
    date('2023-12-31'),
    true,
    timestamp('2023-12-31 23:59:59'),
    cast(1234.56 as decimal);

insert overwrite directory 'file:///tmp/out/test_all_primitive_types/'
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
select * from test_all_primitive_types;
drop table test_all_primitive_types;