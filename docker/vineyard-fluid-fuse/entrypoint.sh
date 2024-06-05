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

# Start with mandatory arguments
args=(
  "--socket=$FUSE_DIR/vineyard-local.sock"
  "--size=$SIZE"
  "--etcd_endpoint=$ETCD_ENDPOINT"
)

# Optional arguments
[ -n "$RESERVE_MEMORY" ]   && args+=("--reserve_memory=$RESERVE_MEMORY")
[ -n "$ALLOCATOR" ]        && args+=("--allocator=$ALLOCATOR")
[ -n "$COMPRESSION" ]      && args+=("--compression=$COMPRESSION")
[ -n "$COREDUMP" ]         && args+=("--coredump=$COREDUMP")
[ -n "$META_TIMEOUT" ]     && args+=("--meta_timeout=$META_TIMEOUT")
[ -n "$ETCD_PREFIX" ]      && args+=("--etcd_prefix=$ETCD_PREFIX")
[ -n "$SPILL_PATH" ]       && args+=("--spill_path=$SPILL_PATH")
[ -n "$SPILL_LOWER_RATE" ] && args+=("--spill_lower_rate=$SPILL_LOWER_RATE")
[ -n "$SPILL_UPPER_RATE" ] && args+=("--spill_upper_rate=$SPILL_UPPER_RATE")

# start the standalone vineyardd
if [ "$SIZE" != "0" ]; then
    # Run vineyardd with constructed argument list
    vineyardd "${args[@]}" &

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
    if [ ! -S $SOCKET_FILE ] && [ -S $MOUNT_DIR/vineyard-worker.sock ]; then
        echo "create a hard link of vineyard socket..."
        ln $MOUNT_DIR/vineyard-worker.sock $SOCKET_FILE
    else
        echo "$SOCKET_FILE exists."
    fi

    # avoid vineyardd restart
    echo "check whether the inode number is same..."
    if [ -S $SOCKET_FILE ] && [ -S $MOUNT_DIR/vineyard-worker.sock ]; then
        SOCKET_INODE=$(ls -i $SOCKET_FILE | awk '{print $1}')
        MOUNT_INODE=$(ls -i $MOUNT_DIR/vineyard-worker.sock | awk '{print $1}')
        if [ "$SOCKET_INODE" != "$MOUNT_INODE" ]; then
            echo "inode number is different, remove the hard link of vineyard socket"
            rm -f $SOCKET_FILE
        fi
    fi

    # avoid vineyard worker crash
    echo "check whether vineyard worker socket exists..."
    if [ ! -S $MOUNT_DIR/vineyard-worker.sock ]; then
        echo "vineyard worker socket does not exist, remove the hard link of vineyard socket"
        rm -f $SOCKET_FILE
    fi

    # wait for 5 seconds before checking again
    sleep 5
done
