drop table if exists nested_table;
create table nested_table (
    field_1 map<int,
                array<struct<field_1:int,
                                field_2:string>>>
);

insert into nested_table select
    map(
        42,
        array(named_struct('field_1', 1,
                            'field_2', 'hello'),
                named_struct('field_1', 2,
                            'field_2', 'world!')));

insert overwrite directory 'file:///tmp/out/test_nested_types/'
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
select * from nested_table;
drop table nested_table;