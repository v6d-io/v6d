from enum import Enum
import os
from signal import SIGINT
import sys
from time import sleep
from pyrfc3339 import generate
import vineyard as vy
import pandas as pd
import pyarrow as pa
import numpy as np
import subprocess as sp
import asyncio
import logging as logger
import time
socket_name = ""
timestamp = int(time.time())
socket_name = "/var/run/%dvineyard.sock"%(timestamp) 

test_mount_dir = "/tmp/vyfs-test%d"%(timestamp)


class Type(Enum):
    STRING = 1
    INT64 = 2
    DOUBLE = 3
async def start_vineyard_server():
    print("vineyard started")

    cmd = ["sudo","-E","python3","-m","vineyard","--socket=%s"%socket_name]
    cmd = ' '.join(cmd)
    print("initilize vineyard by "+cmd)

    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=sys.stdout)
        # stderr=sys.subPIPEprocess.)

    # out, err = await proc.communicate()
    # print(out)
    
    return proc
async def start_fuse_server():
    print("fuse started")
    fuse_bin  = [os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..','..', '..','build', 'bin',"vineyard-fusermount"
    )]
    fuse_param = ["-f","-s","--vineyard-socket="+socket_name]
    os.mkdir(test_mount_dir)
    fuse_dir = [test_mount_dir]
    cmd  = fuse_bin+ fuse_param+ fuse_dir
    cmd = ' '.join(cmd)
    logger.debug("initilize fuse by " + cmd)
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=sys.stdout)
    return proc

def connect_to_server():
    client = vy.connect(socket_name)
    return client
def interrupt_proc(proc:asyncio.subprocess.Process):
    print("interrupt")
    proc.send_signal(SIGINT)
def generate_dataframe(size = (15,4)):
    height,width = size
    df = pd.DataFrame(np.random.randint(0,100,size=(height, width)), columns=list('ABCD'))
    return df
def generate_string_array(length = 20):
    res = []
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']

    for i in range(length):
        l = np.random.randint(1,length)
        res.append(''.join(np.random.choice(alphabet,l)))

    return res

def generate_array(type:Type, length = 20):
    f = {
        Type.INT64:lambda x: np.random.randint(0,1000,x),
        Type.DOUBLE:lambda x:np.random.uniform(low=0, high=1000, size=x),
        Type.STRING:lambda x:generate_string_array(x),
 
    }
    return pa.array(f[type](length))
    


def assert_dataframe(stored_df:pd.DataFrame, extracted_df:pa.Table):
    pdf = pa.Table.from_pandas(stored_df)
    assert extracted_df.equals(pdf),"data frame unmatch"
def assert_array(stored_arr:pa.Array, extracted_array:pa.Array):
    assert stored_arr.equals(extracted_array), "array unmatch"

def read_data_from_fuse(vid):
    with open(os.path.join(test_mount_dir,vid), 'rb') as source:
        with pa.ipc.open_stream(source) as reader:
            data = reader.read_all()
            return data

def test_fuse_array(data,client):
    id = client.put(data)
    extracted_data = read_data_from_fuse(str(id)[11:28])
    print("data: ")
    print(data)
    print("extracted data: ")
    extracted_data = extracted_data.column("a").chunk(0)
    print(extracted_data)
    assert_array(data,extracted_data)

def test_fuse_string_array(data,client):
    id = client.put(data)
    extracted_data = read_data_from_fuse(str(id)[11:28])
    print("data: ")
    print(data)
    print("extracted data: ")
    extracted_data = extracted_data.column("a").chunk(0)
    print(extracted_data)
    assert compare_two_string_array(data,extracted_data), "string array not the same"

def compare_two_string_array(arr_str_1,arr_str_2):
    a = arr_str_1
    b = arr_str_2
    if(len(a) != len(b)):
        return False
    else:
        for i,j in zip(a,b):
            if str(i)!= str(j):
                print("they are different")
                print(i)
                print(j)
                return False
    return True

def test_fuse_df(data,client):
    id = client.put(data)
    extracted_data = read_data_from_fuse(str(id)[11:28])
    assert_dataframe(data,extracted_data)
# string_array = generate_array(Type.STRING)
# int_array =generate_array(Type.INT64)
# print(int_array)
if __name__ == "__main__":
    
    logger.basicConfig(filename='my.log', level=logger.DEBUG)
    print("started")

    # vineyard_server = start_vineyard_server()
    vineyard_server = asyncio.run(start_vineyard_server())
    print(vineyard_server)
    sleep(2)
    fuse_server = asyncio.run(start_fuse_server())
    sleep(2)
    print("server started")
    client = connect_to_server()
# test array
    int_array =generate_array(Type.INT64)
    test_fuse_array(int_array,client)
    double_array = generate_array(Type.DOUBLE)
    test_fuse_array(double_array,client)
    
    string_array = generate_array(Type.STRING)
    test_fuse_string_array(string_array,client) 
    print("array_test passed")
# test df
    int_df = generate_dataframe()
    test_fuse_df(int_df,client)
    print("df_test_passed")
    interrupt_proc(fuse_server)
    interrupt_proc(vineyard_server)

