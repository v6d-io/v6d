#!/bin/sh
set -ex

SOCKET_FILE="$FUSE_DIR/vineyard.sock"
RPC_CONFIG_FILE="$RPC_CONF_DIR/VINEYARD_RPC_ENDPOINT"
VINEYARD_YAML_FILE="$FUSE_DIR/vineyard.yaml"

# Write the IPCSocket and RPCEndpoints to the vineyard configurations YAML file
write_yaml_config() {
    echo "Vineyard:" > $VINEYARD_YAML_FILE
    echo "  IPCSocket: vineyard.sock" >> $VINEYARD_YAML_FILE
    echo "  RPCEndpoint: $1" >> $VINEYARD_YAML_FILE
}

mkdir -p $FUSE_DIR
while true; do
    # check if prestop marker exists, if so, skip mounting
    if [ -f $PRESTOP_MARKER ]; then
        echo "PreStop hook is in progress, skip mounting."
        break
    fi
    # before mounting, store the rpc endpoint to a variable
    if [ -f $RPC_CONFIG_FILE ]; then
        VINEYARD_RPC_ENDPOINT=$(cat $RPC_CONFIG_FILE)
    else
        echo "rpc config file $RPC_CONFIG_FILE does not exist."
    fi

    if [ ! -S $SOCKET_FILE ]; then
        echo "Checking if vineyard-fuse is already a mount point..."
        if ! mountpoint -q $FUSE_DIR; then
            echo "mount vineyard socket..."
            mount --bind $MOUNT_DIR $FUSE_DIR
            echo "write vineyard ipc socket and rpc endpoint to vineyard configuration YAML..."
            write_yaml_config "$VINEYARD_RPC_ENDPOINT"
        else
            echo "$FUSE_DIR is already a mount point."
        fi
    else
        echo "$SOCKET_FILE exists."
    fi
    # wait for a minute so that the fuse mount point can be checked again
    sleep 60
done
