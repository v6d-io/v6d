import vineyard

client = vineyard.connect("/opt/v6d/build/vineyard.sock")

target = vineyard.ObjectID("o003f29fd1f72415c")

print(client.get_meta(target))
# print(client.get(target))

table = client.get(target)
k = client.put(table)
r = client.get(k)
s = client.put(r)
print(client.get(s))
