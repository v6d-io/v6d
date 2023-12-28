FROM registry.cn-hangzhou.aliyuncs.com/bigdata_cloudnative/hadoop:v1

COPY ./vineyard-hive-0.1-SNAPSHOT.jar ${HADOOP_HOME}/share/hadoop/common/

# prepare hdoop config
COPY ./hadoop-config/* ${HADOOP_HOME}/etc/hadoop/

# prepare tez
RUN sudo mkdir -p /opt/apache/tez && sudo chown -R hadoop:hadoop /opt/apache/tez
ENV TEZ_HOME=/opt/apache/tez
COPY tez /opt/apache/tez/
COPY ./tez-config/tez-site.xml ${HADOOP_HOME}/etc/hadoop/
COPY ./tez-config/tez-site.xml ${HIVE_HOME}/conf/
COPY tez/share/tez.tar.gz /

# prepare hive
RUN sudo rm -r ${HIVE_HOME}
COPY hive /opt/apache/hive/
COPY hive-config/ /hive-config
COPY hive-config-distributed/ /hive-config-distributed

COPY bootstrap.sh /opt/apache/
COPY mysql-connector-java-5.1.49/mysql-connector-java-5.1.49-bin.jar ${HIVE_HOME}/lib/

RUN sudo yum -y install unzip
RUN wget http://www.vuln.cn/wp-content/uploads/2019/08/libstdc.so_.6.0.26.zip -q && unzip libstdc.so_.6.0.26.zip
RUN cp libstdc++.so.6.0.26 /usr/lib64
RUN rm /usr/lib64/libstdc++.so.6
RUN ln -s /usr/lib64/libstdc++.so.6.0.26 /usr/lib64/libstdc++.so.6

RUN sudo yum -y install vim

#RUN yum -y install which
