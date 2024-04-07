#!/bin/sh
set -ex

SOCKET_NAME="vineyard-worker.sock"
SOCKET_FILE="$FUSE_DIR/vineyard-worker.sock"
RPC_CONFIG_FILE="$RPC_CONF_DIR/VINEYARD_RPC_ENDPOINT"
VINEYARD_YAML_FILE="$FUSE_DIR/vineyard-config.yaml"

# Write the IPCSocket and RPCEndpoints to the vineyard configurations YAML file
write_yaml_config() {
    echo "Vineyard:" > $VINEYARD_YAML_FILE
    echo "  IPCSocket: $SOCKET_NAME" >> $VINEYARD_YAML_FILE
    echo "  RPCEndpoint: $1" >> $VINEYARD_YAML_FILE
}

# start the standalone vineyardd
if [ "$CACHE_SIZE" != "0" ]; then
    vineyardd --socket=$FUSE_DIR/vineyard-local.sock \
            --size=$CACHE_SIZE \
            --etcd_endpoint=$ETCD_ENDPOINT &

    # wait for the local vineyard socket to be created
    timeout=60
    count=0
    while [ ! -S $FUSE_DIR/vineyard-local.sock ]; do
        sleep 1
        count=$((count+1))
        if [ $count -eq $timeout ]; then
            echo "Timeout waiting for $FUSE_DIR/vineyard-local.sock"
            exit 1
        fi
    done
    SOCKET_NAME="vineyard-local.sock"
    echo "Local vineyardd is started."
fi

mkdir -p $FUSE_DIR
while true; do
    # check if prestop marker exists, if so, skip creating the hard link
    if [ -f $PRESTOP_MARKER ]; then
        echo "PreStop hook is in progress, skip creating the hard link of vineyard socket."
        break
    fi
    # before creating the hard link of vineyard socket, store the rpc endpoint to a variable
    if [ -f $RPC_CONFIG_FILE ]; then
        VINEYARD_RPC_ENDPOINT=$(cat $RPC_CONFIG_FILE)
    else
        echo "rpc config file $RPC_CONFIG_FILE does not exist."
    fi

    echo "write vineyard ipc socket and rpc endpoint to vineyard configuration YAML..."
    write_yaml_config "$VINEYARD_RPC_ENDPOINT"
    echo "check whether vineyard socket symlink is created..."
    if [ ! -S $SOCKET_FILE ] && [ -S $MOUNT_DIR/vineyard.sock ]; then
        echo "create a hard link of vineyard socket..."
        ln $MOUNT_DIR/vineyard.sock $SOCKET_FILE
    else
        echo "$SOCKET_FILE exists."
    fi
    # wait for a minute so that the hard link of vineyard socket can be checked again
    sleep 60
done
