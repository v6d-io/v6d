import json

with open("/etc/fluid/config.json", "r") as f:
    lines = f.readlines()

rawStr = lines[0]
print(rawStr)


script = """
#!/bin/sh
set -ex

mkdir -p $targetPath
while true; do
    if [ ! -S "$targetPath/vineyard.sock" ]; then
        mount --bind $socketPath $targetPath
    fi
    sleep 10
done
"""

obj = json.loads(rawStr)

with open("mount-vineyard-socket.sh", "w") as f:
    f.write("targetPath=\"%s\"\n" % obj['targetPath'])
    if obj['mounts'][0]['mountPoint'].startswith("local://"):
      f.write("socketPath=\"%s\"\n" % obj['mounts'][0]['mountPoint'][len("local://"):])
    else:
      f.write("socketPath=\"%s\"\n" % obj['mounts'][0]['mountPoint'])

    f.write(script)
