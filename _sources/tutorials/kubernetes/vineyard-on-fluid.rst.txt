Vineyard + Fluid in Action: Train a Linear Regression Model on ACK
==================================================================

In this tutorial, we will demonstrate how to train a linear regression 
model on ACK (Alibaba Cloud Kubernetes) using VineyardRuntime, 
please follow the steps below.

Step 1: Create a dataset and upload it to OSS
---------------------------------------------

The ``prepare-dataset.py`` code introduced below shows how to use the 
numpy and pandas libraries in Python to generate a data set of about 22GB 
and upload the files to the OSS service through `ossutil`_. The whole process is 
divided into two main parts:

.. note::

    If your ACK node machine memory is insufficient to generate a 22GB dataset,
    please reduce the number of rows (num_rows) in the dataset.

1. **Dataset creation**: This code uses the pandas library to create a DataFrame 
containing multiple random number columns that simulate various attributes of
the real estate market.

2. **Serialize the dataset**: This line of code serializes the dataset into
the local file ``df.pkl``.

.. code:: python

    import numpy as np
    import pandas as pd

    # generate the dataframe with a size of about 22G
    num_rows = 6000 * 10000
    df = pd.DataFrame({
        'Id': np.random.randint(1, 100000, num_rows),
        'MSSubClass': np.random.randint(20, 201, size=num_rows),
        'LotFrontage': np.random.randint(50, 151, size=num_rows),
        'LotArea': np.random.randint(5000, 20001, size=num_rows),
        'OverallQual': np.random.randint(1, 11, size=num_rows),
        'OverallCond': np.random.randint(1, 11, size=num_rows),
        'YearBuilt': np.random.randint(1900, 2022, size=num_rows),
        'YearRemodAdd': np.random.randint(1900, 2022, size=num_rows),
        'MasVnrArea': np.random.randint(0, 1001, size=num_rows),
        'BsmtFinSF1': np.random.randint(0, 2001, size=num_rows),
        'BsmtFinSF2': np.random.randint(0, 1001, size=num_rows),
        'BsmtUnfSF': np.random.randint(0, 2001, size=num_rows),
        'TotalBsmtSF': np.random.randint(0, 3001, size=num_rows),
        '1stFlrSF': np.random.randint(500, 4001, size=num_rows),
        '2ndFlrSF': np.random.randint(0, 2001, size=num_rows),
        'LowQualFinSF': np.random.randint(0, 201, size=num_rows),
        'GrLivArea': np.random.randint(600, 5001, size=num_rows),
        'BsmtFullBath': np.random.randint(0, 4, size=num_rows),
        'BsmtHalfBath': np.random.randint(0, 3, size=num_rows),
        'FullBath': np.random.randint(0, 5, size=num_rows),
        'HalfBath': np.random.randint(0, 3, size=num_rows),
        'BedroomAbvGr': np.random.randint(0, 11, size=num_rows),
        'KitchenAbvGr': np.random.randint(0, 4, size=num_rows),
        'TotRmsAbvGrd': np.random.randint(0, 16, size=num_rows),
        'Fireplaces': np.random.randint(0, 4, size=num_rows),
        'GarageYrBlt': np.random.randint(1900, 2022, size=num_rows),
        'GarageCars': np.random.randint(0, 5, num_rows),
        'GarageArea': np.random.randint(0, 1001, num_rows),
        'WoodDeckSF': np.random.randint(0, 501, num_rows),
        'OpenPorchSF': np.random.randint(0, 301, num_rows),
        'EnclosedPorch': np.random.randint(0, 201, num_rows),
        '3SsnPorch': np.random.randint(0, 101, num_rows),
        'ScreenPorch': np.random.randint(0, 201, num_rows),
        'PoolArea': np.random.randint(0, 301, num_rows),
        'MiscVal': np.random.randint(0, 5001, num_rows),
        'TotalRooms': np.random.randint(2, 11, num_rows),
        "GarageAge": np.random.randint(1, 31, num_rows),
        "RemodAge": np.random.randint(1, 31, num_rows),
        "HouseAge": np.random.randint(1, 31, num_rows),
        "TotalBath": np.random.randint(1, 5, num_rows),
        "TotalPorchSF": np.random.randint(1, 1001, num_rows),
        "TotalSF": np.random.randint(1000, 6001, num_rows),
        "TotalArea": np.random.randint(1000, 6001, num_rows),
        'MoSold': np.random.randint(1, 13, num_rows),
        'YrSold': np.random.randint(2006, 2022, num_rows),
        'SalePrice': np.random.randint(50000, 800001, num_rows),
    })

    # Save the dataframe to the current directory
    df.to_pickle("df.pkl")

3. **Upload the data set to OSS**: Follow the following command to use ossutil to 
upload the current file to the OSS service.

.. code:: bash

    # Upload the current dataset df.pkl to the OSS service through the ossutil cp command.
    # Refer to https://help.aliyun.com/zh/oss/developer-reference/upload-objects-6?spm=a2c4g.11186623.0.i3
    $ ossutil cp ./df.pkl oss://your-bucket-name/your-path


Step 2: Install the Fluid control plane and Fluid Python SDK in the ACK cluster.
--------------------------------------------------------------------------------

Option 1: Install ack-fluid. Refer to the document: `Install the cloud native AI suite`_

Option 2: Using the open-source version, we will use `Kubectl`_ to create a 
namespace named ``fluid-system``, and then use `Helm`_ to install Fluid.
This process only needs to be completed through the following simple Shell commands.

.. code:: bash

    # Create the fluid-system namespace
    $ kubectl create ns fluid-system

    # Add the Fluid repository to the Helm repository
    $ helm repo add fluid https://fluid-cloudnative.github.io/charts
    # Get the latest Fluid repository
    $ helm repo update
    # Find the development version in the Fluid repository
    $ helm search repo fluid --devel
    # Deploy the corresponding version of the Fluid chart on ACK
    $ helm install fluid fluid/fluid --devel

After we deploy the Fluid platform on ACK, we need to execute the following pip 
command to install the Fluid Python SDK.

.. code:: bash

    $ pip install git+https://github.com/fluid-cloudnative/fluid-client-python.git


Step 3: Enable collaborative scheduling of data and tasks (optional)
--------------------------------------------------------------------

In cloud environments, end-to-end data operation pipelines often contain multiple subtasks.
When these tasks are scheduled by Kubernetes, the system only considers the required resource
constraints and cannot guarantee that two consecutively executed tasks can run on the same node.
This results in additional network overhead due to data migration when the two use Vineyard to
share intermediate results.

If you want to schedule tasks and vineyard to the same node to achieve the best performance,
you can modify the configmap configuration as follows to enable fuse affinity scheduling.
In this way, system scheduling will give priority to associated tasks to access memory on the
same node to reduce data migration. The network overhead incurred.

.. code:: bash

    # Update the webhook-plugins configuration according to the following command 
    # and enable fuse affinity scheduling
    $ kubectl edit configmap webhook-plugins -n fluid-system
    data:
    pluginsProfile: |
        pluginConfig:
        - args: |
            preferred:
            # Enable fuse affinity scheduling
            - name: fluid.io/fuse
                weight: 100
        ...

    # Restart the fluid-webhook pod
    $ kubectl delete pod -lcontrol-plane=fluid-webhook -n fluid-system

Step 4: Use Fluid Python SDK to build and deploy linear regression data operation pipeline
------------------------------------------------------------------------------------------

In the ``linear-regression-with-vineyard.py`` script below, we will explore an example
of building and deploying a machine learning workflow using Python and the Fluid library.
The dataset is generated by the code in the appendix. The workflow covers data preprocessing,
The whole process of model training and model testing.

.. code:: python

    import fluid

    from fluid import constants
    from fluid import models

    # Create a Fluid client instance by connecting to the Fluid control plane 
    # using the default kubeconfig file
    client_config = fluid.ClientConfig()
    fluid_client = fluid.FluidClient(client_config)

    # Create a dataset named vineyard in the default namespace
    fluid_client.create_dataset(
        dataset_name="vineyard",
    )

    # Get the vineyard dataset instance
    dataset = fluid_client.get_dataset(dataset_name="vineyard")

    # Initialize the configuration of the vineyard runtime and bind the 
    # vineyard dataset instance to the runtime.
    # The number of replicas is 2, and the memory is 30Gi
    dataset.bind_runtime(
        runtime_type=constants.VINEYARD_RUNTIME_KIND,
        replicas=2,
        cache_capacity_GiB=30,
        cache_medium="MEM",
        wait=True
    )

    # define the data preprocessing task
    def preprocess():
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

    # define the model training task
    def train():
        from sklearn.linear_model import LinearRegression

        import joblib
        import pandas as pd
        import vineyard

        x_train_data = vineyard.get(name="x_train", fetch=True)
        y_train_data = vineyard.get(name="y_train", fetch=True)

        model = LinearRegression()
        model.fit(x_train_data, y_train_data)

        joblib.dump(model, '/data/model.pkl')

    # define the model testing task
    def test():
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

    packages_to_install = ["numpy", "pandas", "pyarrow", "requests", "vineyard", "scikit-learn==1.4.0", "joblib==1.3.2"]
    pip_index_url = "https://pypi.tuna.tsinghua.edu.cn/simple"

    preprocess_processor = create_processor(preprocess, packages_to_install, pip_index_url)
    train_processor = create_processor(train, packages_to_install, pip_index_url)
    test_processor = create_processor(test, packages_to_install, pip_index_url)

    # Create a linear regression model task workflow: data preprocessing -> model training -> model testing
    # The following mount path "/var/run/vineyard" is the default path of the vineyard configuration file
    flow = dataset.process(processor=preprocess_processor, dataset_mountpath="/var/run/vineyard") \
                .process(processor=train_processor, dataset_mountpath="/var/run/vineyard") \
                .process(processor=test_processor, dataset_mountpath="/var/run/vineyard")

    # Submit the data processing task workflow of the linear regression model and wait for it to run to completion
    run = flow.run(run_id="linear-regression-with-vineyard")
    run.wait()

Here's an overview of each part of the code:

1. **Create Fluid client**: This code is responsible for establishing
a connection with the Fluid control platform using the default kubeconfig file and
creating a Fluid client instance.

2. **Create and configure the vineyard dataset and runtime environment**: Next, the code
creates a dataset named ``Vineyard``, then obtains the dataset instance, initializes the vineyard
runtime configuration, and sets up a copy number and memory size to bind the dataset to the
runtime environment.

3. **Define the data preprocessing function**: This part defines a python function for data
preprocessing, which includes splitting the training set and the test set, as well as
data filtering and other operations.

4. **Define model training function**: As the name suggests, this code defines another
python function for training a linear regression model.

5. **Define the model testing function**: This section contains the model testing logic
for evaluating the trained model.

6. **Create a task template and define task workflow**: The code encapsulates a task
template function named create_processor, which uses the previously defined python functions
to build data preprocessing, model training and model testing steps respectively.
These steps are designed to be executed sequentially, forming a complete workflow in which
data preprocessing is the first step, followed by model training, and finally model testing.
This serial execution sequence ensures that the output of each stage can be used as the input
of the next stage, thereby achieving a coherent and orderly machine learning process.

7. **[Optional] Enable data affinity scheduling**: After enabling fuse affinity scheduling,
add the tag ``"fuse.serverful.fluid.io/inject": "true"`` to ensure that related tasks run on the
same node first through scheduling. to achieve the best performance in data processing.

8. **Submit and execute the task workflow**: Submit the entire linear regression model task
workflow to the Fluid platform for execution through the run command.

9. **Resource Cleanup**: Finally, clean up all resources created on the Fluid platform.

.. _Install the cloud native AI suite: https://help.aliyun.com/zh/ack/cloud-native-ai-suite/user-guide/deploy-the-cloud-native-ai-suite?spm=a2c4g.11186623.0.i14#task-2038811
.. _ossutil: https://help.aliyun.com/zh/oss/developer-reference/ossutil
.. _Kubectl: https://github.com/kubernetes/kubectl
.. _Helm: https://github.com/helm/helm