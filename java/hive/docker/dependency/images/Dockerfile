FROM apache/hadoop:3.3.5

RUN mkdir -p /opt/apache/; \
    mv /opt/hadoop/ /opt/apache/
ENV HADOOP_HOME=/opt/apache/hadoop \
    HADOOP_CONF_DIR=/opt/apache/hadoop/etc/hadoop \
    HIVE_HOME=/opt/apache/hive
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

# prepare spark
COPY spark /opt/apache/spark/
ENV SPARK_HOME=/opt/apache/spark
COPY spark-config/ /spark-config
COPY spark-config-distributed/ /spark-config-distributed
COPY ./vineyard-hive-0.1-SNAPSHOT.jar ${SPARK_HOME}/jars/

ENV PATH=${HADOOP_HOME}/bin:${HADOOP_HOME}/sbin:${HIVE_HOME}/bin:${SPARK_HOME}/bin:${PATH}

COPY bootstrap.sh /opt/apache/
COPY mysql-connector-java-5.1.49/mysql-connector-java-5.1.49-bin.jar ${HIVE_HOME}/lib/

RUN sudo yum -y install unzip; \
    wget http://www.vuln.cn/wp-content/uploads/2019/08/libstdc.so_.6.0.26.zip -q && unzip libstdc.so_.6.0.26.zip; \
    sudo cp libstdc++.so.6.0.26 /usr/lib64; \
    sudo rm /usr/lib64/libstdc++.so.6; \
    sudo ln -s /usr/lib64/libstdc++.so.6.0.26 /usr/lib64/libstdc++.so.6; \
    sudo yum -y install vim; \
    rm libstdc.so_.6.0.26.zip libstdc++.so.6.0.26; \
    sudo yum install -y net-tools
