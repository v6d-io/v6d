import fluid

from fluid import constants
from fluid import models

# Use the default kubeconfig file to connect to the Fluid control plane 
# and create a Fluid client instance
client_config = fluid.ClientConfig()
fluid_client = fluid.FluidClient(client_config)

# Create a dataset named "vineyard" in the default namespace
fluid_client.create_dataset(
    dataset_name="vineyard",
    mount_name="dummy-mount-name",
    mount_point="dummy-mount-point"
)

# Get the dataset instance of the "vineyard" dataset
dataset = fluid_client.get_dataset(dataset_name="vineyard")

# Init vineyard runtime configuration and bind the vineyard dataset instance to the runtime.
# Replicas is 2, and the memory is 30Gi
dataset.bind_runtime(
    runtime_type=constants.VINEYARD_RUNTIME_KIND,
    replicas=2,
    cache_capacity_GiB=30,
    cache_medium="MEM",
    wait=True
)

# define the script of data preprocessing
preprocess_data_script = """
pip3 install numpy pandas pyarrow requests vineyard scikit-learn==1.4.0 joblib==1.3.2
#!/bin/bash
set -ex

cat <<EOF > ./preprocess.py
from sklearn.model_selection import train_test_split

import pandas as pd
import vineyard

df = pd.read_pickle('/data/df.pkl')

# Preprocess Data
df = df.drop(df[(df['GrLivArea']>4800)].index)
X = df.drop('SalePrice', axis=1)  # Features
y = df['SalePrice']  # Target variable

del df

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

del X, y

vineyard.put(X_train, name="x_train", persist=True)
vineyard.put(X_test, name="x_test", persist=True)
vineyard.put(y_train, name="y_train", persist=True)
vineyard.put(y_test, name="y_test", persist=True)

EOF

python3 ./preprocess.py
"""

# define the script of model training
train_data_script = """
pip3 install numpy pandas pyarrow requests vineyard scikit-learn==1.4.0 joblib==1.3.2
#!/bin/bash
set -ex

cat <<EOF > ./train.py
from sklearn.linear_model import LinearRegression

import joblib
import pandas as pd
import vineyard

x_train_data = vineyard.get(name="x_train", fetch=True)
y_train_data = vineyard.get(name="y_train", fetch=True)

model = LinearRegression()
model.fit(x_train_data, y_train_data)

joblib.dump(model, '/data/model.pkl')

EOF
python3 ./train.py
"""

# define the script of model testing
test_data_script = """
pip3 install numpy pandas pyarrow requests vineyard scikit-learn==1.4.0 joblib==1.3.2
#!/bin/bash
set -ex

cat <<EOF > ./test.py
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import vineyard
import joblib
import pandas as pd

x_test_data = vineyard.get(name="x_test", fetch=True)
y_test_data = vineyard.get(name="y_test", fetch=True)

model = joblib.load("/data/model.pkl")
y_pred = model.predict(x_test_data)

err = mean_squared_error(y_test_data, y_pred)

with open('/data/output.txt', 'a') as f:
    f.write(str(err))

EOF

python3 ./test.py
"""

from kubernetes.client import models as k8s_models
# define the template of the task processor and mount the OSS Volume
def create_processor(script):
    return models.Processor(
        # When enabling fuse affinity scheduling, add the following label
        # to achieve the best performance of data processing
        # pod_metadata=models.PodMetadata(
        #     labels={"fuse.serverful.fluid.io/inject": "true"},
        # ),
        script=models.ScriptProcessor(
            command=["bash"],
            source=script,
            image="python",
            image_tag="3.10",
            volumes=[k8s_models.V1Volume(
                name="data",
                persistent_volume_claim=k8s_models.V1PersistentVolumeClaimVolumeSource(
                    claim_name="pvc-oss"
                )
            )],
            volume_mounts=[k8s_models.V1VolumeMount(
                name="data",
                mount_path="/data"
            )],
        )   
    )

preprocess_processor = create_processor(preprocess_data_script)
train_processor = create_processor(train_data_script)
test_processor = create_processor(test_data_script)

# Create a linear regression model task workflow: data preprocessing -> model training -> model testing
# The following mount path "/var/run" is the default path of the vineyard configuration file
flow = dataset.process(processor=preprocess_processor, dataset_mountpath="/var/run") \
              .process(processor=train_processor, dataset_mountpath="/var/run") \
              .process(processor=test_processor, dataset_mountpath="/var/run")

# Submit the linear regression model task workflow to the Fluid platform and start execution
run = flow.run(run_id="linear-regression-with-vineyard")
run.wait()

# Clean up all resources
dataset.clean_up(wait=True)
