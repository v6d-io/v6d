Distributed Learning with Vineyard
==================================

With the growth of data, distributed learning is becoming a must in real-world machine learning
applications, as the data size can easily exceed the memory limit of a single machine.
Thus, many distributed systems addressing different workloads are developed
and they share the same objective of extending users' single machine prototypes 
to distributed settings with as few modifications to the code as possible.

For example, **dask.dataframe** mimics the API of **pandas** which is the de-facto standard
library for single-machine structured data processing, so that users can apply their
pandas code for data preprocessing in the dask cluster with few modifications.
Similarly, **horovod** provides easy-to-use APIs for users to transfer their single-machine
code in machine learning frameworks (e.g., TensorFlow, PyTorch, MXNet) to the distributed settings
with only a few additional lines of code.

However, when extending to distributed learning, the data sharing between libraries within the same
python process (e.g., pandas and tensorflow) becomes inter-process sharing between engines (e.g.,
dask and horovod), not to mention in the distributed fashion. Existing solutions using external
distributed file systems are less than optimal for the huge I/O overheads.

Vineyard shares the same design principle with the aforementioned distributed systems, which aims to
provide efficient cross-engine data sharing with few modifications to the existing code.
Next, we demonstrate how to transfer a single-machine learing example in **keras** to distributed learning
with dask, horovod and Vineyard.

An Example from Keras
---------------------
This example_ uses the Covertype dataset from the UCI Machine Learning Repository.
The task is to predict forest cover type from cartographic variables.
The dataset includes 506,011 instances with 12 input features:
10 numerical features and 2 categorical features.
Each instance is categorized into 1 of 7 classes.

The solution contains three steps:

1. preprocess the data in pandas to extract the 12 features and the label

2. store the preprocessed data in files

3. define and train the model in keras


Mapping the solution to distributed learning, we have:

1. preprocess the data in dask.dataframe

2. share the preprocessed data using Vineyard

3. train the model in horovod.keras


We will walk through the code as follows.

Preprocessing the data
----------------------

Suppose we have a much larger dataset that does not fit into
the memory of a single machine. To read the data, we replace
**pd.read_csv** by **dd.read_csv**, which will automatically
read the data in parallel.

.. code:: python

    import dask.dataframe as dd
    raw_data = dd.read_csv(data_path, header=None)

Then we preprocess the data using the same code from the example_,
except the replacement of **pd.concat** to **dd.concat** only.

.. code:: python

    """
    The two categorical features in the dataset are binary-encoded.
    We will convert this dataset representation to the typical representation, where each
    categorical feature is represented as a single integer value.
    """

    soil_type_values = [f"soil_type_{idx+1}" for idx in range(40)]
    wilderness_area_values = [f"area_type_{idx+1}" for idx in range(4)]

    soil_type = raw_data.loc[:, 14:53].apply(
        lambda x: soil_type_values[0::1][x.to_numpy().nonzero()[0][0]], axis=1
    )
    wilderness_area = raw_data.loc[:, 10:13].apply(
        lambda x: wilderness_area_values[0::1][x.to_numpy().nonzero()[0][0]], axis=1
    )

    CSV_HEADER = [
        "Elevation",
        "Aspect",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points",
        "Wilderness_Area",
        "Soil_Type",
        "Cover_Type",
    ]

    data = dd.concat(
        [raw_data.loc[:, 0:9], wilderness_area, soil_type, raw_data.loc[:, 54]],
        axis=1,
        ignore_index=True,
    )
    data.columns = CSV_HEADER

    # Convert the target label indices into a range from 0 to 6 (there are 7 labels in total).
    data["Cover_Type"] = data["Cover_Type"] - 1

Finally, instead of saving the preprocessed data into files, we store them in Vineyard.

.. code:: python

    import vineyard
    import vineyard.contrib.dask.dask # register the dask builders

    gdf_id = vineyard.connect().put(data, dask_scheduler='tcp://localhost:8786')

.. code:: bash

    ObjectID <"o00d60aba46eaf536">

We saved the preprocessed data as a global dataframe
in Vineyard with the ObjectID of **o00d60aba46eaf536**.

Training the model
------------------
In the single machine solution from the example_. A **get_dataset_from_csv** function 
is defined to load the dataset from the files of the preprocessed data as follows:

.. code:: python

    def get_dataset_from_csv(csv_file_path, batch_size, shuffle=False):

        dataset = tf.data.experimental.make_csv_dataset(
            csv_file_path,
            batch_size=batch_size,
            column_names=CSV_HEADER,
            column_defaults=COLUMN_DEFAULTS,
            label_name=TARGET_FEATURE_NAME,
            num_epochs=1,
            header=True,
            shuffle=shuffle,
        )
        return dataset.cache()

while in the training procedure, it loads the train_dataset and test_dataset
seperately from two files as:

.. code:: python

    def run_experiment(model):

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )

        train_dataset = get_dataset_from_csv(train_data_file, batch_size, shuffle=True)

        test_dataset = get_dataset_from_csv(test_data_file, batch_size)

        print("Start training the model...")
        history = model.fit(train_dataset, epochs=num_epochs)
        print("Model training finished")

        _, accuracy = model.evaluate(test_dataset, verbose=0)

        print(f"Test accuracy: {round(accuracy * 100, 2)}%")

In our solution, we provide a function to load dataset from the global dataframe
generated in the last step.

.. code:: python

    import vineyard.contrib.ml.tensorflow  # register tf data resolvers

    def get_dataset_from_vineyard(object_id, batch_size, shuffle=False):
        
        ds = vineyard.connect().get(object_id, label=TARGET_FEATURE_NAME) # specify the label column

        if shuffle:
            ds = ds.shuffle(len(ds))

        len_test = int(len(ds) * 0.15)
        test_dataset = ds.take(len_test).batch(batch_size)
        train_dataset = ds.skip(len_test).batch(batch_size)

        return train_dataset, test_dataset


And modify the training procedure with a few lines of horovod code.

.. code:: python

    def run_experiment(model):

        hvd.init()

        model.compile(
            optimizer=hvd.DistributedOptimizer(keras.optimizers.Adam(learning_rate=learning_rate)),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )

        callbacks = [
            # Horovod: broadcast initial variable states from rank 0 to all other processes.
            # This is necessary to ensure consistent initialization of all workers when
            # training is started with random weights or restored from a checkpoint.
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        ]

        train_dataset, test_dataset = get_dataset_from_vineyard(sys.argv[1], batch_size, shuffle=True)

        print("Start training the model...")
        history = model.fit(train_dataset, epochs=num_epochs, callbacks=callbacks)
        print("Model training finished")

        _, accuracy = model.evaluate(test_dataset, verbose=0)

        print(f"Test accuracy: {round(accuracy * 100, 2)}%")


Then we can execute the distributed training with the command:

.. code:: bash

    horovodrun -np 4 -H h1:1,h2:1,h3:1,h4:1 python train.py o00d60aba46eaf536

All the other parts of training procedure are the same as the single machine solution.

Conclusion
----------

From this example, we can see that with the help of Vineyard, users can easily extend
their single machine solutions to distributed learning using dedicated systems without
worrying about the cross-system data sharing issues.

.. _example: https://keras.io/examples/structured_data/wide_deep_cross_networks/