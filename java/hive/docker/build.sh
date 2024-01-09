WORK_DIR=~/hive-workdir
mkdir -p "$WORK_DIR"

find ~/hive-workdir -maxdepth 1 -mindepth 1 ! \( -name '*.tar.gz' -o -name '*.tgz' \) -exec rm -rf {} \;

TEZ_VERSION=${TEZ_VERSION:-"0.9.1"}
HIVE_VERSION=${HIVE_VERSION:-"2.3.9"}
SPARK_VERSION=${SPARK_VERSION:-"3.4.1"}

if [ -f "$WORK_DIR/apache-tez-$TEZ_VERSION-bin.tar.gz" ]; then
    echo "Tez exists, skipping download..."
else
    echo "Download Tez..."
    TEZ_URL=${TEZ_URL:-"https://archive.apache.org/dist/tez/$TEZ_VERSION/apache-tez-$TEZ_VERSION-bin.tar.gz"}
    echo "Downloading Tez from $TEZ_URL..."
    if ! curl --fail -L "$TEZ_URL" -o "$WORK_DIR/apache-tez-$TEZ_VERSION-bin.tar.gz"; then
        echo "Failed to download Tez, exiting..."
        exit 1
    fi
fi

if [ -f "$WORK_DIR/apache-hive-$HIVE_VERSION-bin.tar.gz" ]; then
    echo "Hive exists, skipping download..."
else
    echo "Download Hive..."
    if [ -n "$HIVE_VERSION" ]; then
        HIVE_URL=${HIVE_URL:-"https://archive.apache.org/dist/hive/hive-$HIVE_VERSION/apache-hive-$HIVE_VERSION-bin.tar.gz"}
        echo "Downloading Hive from $HIVE_URL..."
        if ! curl --fail -L "$HIVE_URL" -o "$WORK_DIR/apache-hive-$HIVE_VERSION-bin.tar.gz"; then
            echo "Failed to download Hive, exiting..."
            exit 1
        fi
        hive_package="$WORK_DIR/apache-hive-$HIVE_VERSION-bin.tar.gz"
    else
        HIVE_VERSION=$(mvn -f "$SOURCE_DIR/pom.xml" -q help:evaluate -Dexpression=project.version -DforceStdout)
        HIVE_TAR="$SOURCE_DIR/packaging/target/apache-hive-$HIVE_VERSION-bin.tar.gz"
        if  ls $HIVE_TAR || mvn -f $SOURCE_DIR/pom.xml clean package -DskipTests -Pdist; then
            cp "$HIVE_TAR" "$WORK_DIR/"
        else
            echo "Failed to compile Hive Project, exiting..."
            exit 1
        fi
    fi
fi

if [ -f "$WORK_DIR/spark-$SPARK_VERSION-bin-hadoop3.tgz" ]; then
    echo "Spark exists, skipping download..."
else
    echo "Download Spark..."
    SPARK_URL=${SPARK_URL:-"https://archive.apache.org/dist/spark/spark-$SPARK_VERSION/spark-$SPARK_VERSION-bin-hadoop3.tgz"}
    echo "Downloading Spark from $SPARK_URL..."
    if ! curl --fail -L "$SPARK_URL" -o "$WORK_DIR/spark-$SPARK_VERSION-bin-hadoop3.tgz"; then
        echo "Failed to download Spark, exiting..."
        exit 1
    fi
fi

cp -R ./dependency/images/ "$WORK_DIR/"
cp ../target/vineyard-hive-0.1-SNAPSHOT.jar "$WORK_DIR/images/"

tar -xzf "$WORK_DIR/apache-hive-$HIVE_VERSION-bin.tar.gz" -C "$WORK_DIR/"
tar -xzf "$WORK_DIR/apache-tez-$TEZ_VERSION-bin.tar.gz" -C "$WORK_DIR/"
tar -xzf "$WORK_DIR/spark-$SPARK_VERSION-bin-hadoop3.tgz" -C "$WORK_DIR/"
mv "$WORK_DIR/apache-hive-$HIVE_VERSION-bin" "$WORK_DIR/images/hive"
mv "$WORK_DIR/apache-tez-$TEZ_VERSION-bin" "$WORK_DIR/images/tez"
mv "$WORK_DIR/spark-$SPARK_VERSION-bin-hadoop3" "$WORK_DIR/images/spark"

network_name="hadoop-network"

if [[ -z $(docker network ls --filter name=^${network_name}$ --format="{{.Name}}") ]]; then
    echo "Docker network ${network_name} does not exist, creating it..."
    docker network create hadoop-network
else
    echo "Docker network ${network_name} already exists"
fi

docker build \
        "$WORK_DIR/images" \
        -f "$WORK_DIR/images/Dockerfile" \
        -t "apache/hadoop_hive:v1" \
        --no-cache

# rm -r "${WORK_DIR}"