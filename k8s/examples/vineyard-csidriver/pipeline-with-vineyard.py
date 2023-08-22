from kfp import dsl

def PreProcess():
    #############################################################
    # user need to add the volume Op for vineyard object manully
    v1op = dsl.VolumeOp(name="vineyard-object1",
                    resource_name="vineyard-object1-pvc", size="1Mi",
                    storage_class="vineyard-system.vineyardd-sample.csi",
                    modes=dsl.VOLUME_MODE_RWM,
                    set_owner_reference=True)

    v2op = dsl.VolumeOp(name="vineyard-object2",
                resource_name="vineyard-object2-pvc", size="1Mi",
                storage_class="vineyard-system.vineyardd-sample.csi",
                modes=dsl.VOLUME_MODE_RWM,
                set_owner_reference=True)

    v3op = dsl.VolumeOp(name="vineyard-object3",
            resource_name="vineyard-object3-pvc", size="1Mi",
            storage_class="vineyard-system.vineyardd-sample.csi",
            modes=dsl.VOLUME_MODE_RWM,
            set_owner_reference=True)

    v4op = dsl.VolumeOp(name="vineyard-object4",
                    resource_name="vineyard-object4-pvc", size="1Mi",
                    storage_class="vineyard-system.vineyardd-sample.csi",
                    modes=dsl.VOLUME_MODE_RWM,
                    set_owner_reference=True)
    #############################################################

    return dsl.ContainerOp(
        name='Preprocess Data',
        image = 'preprocess-data',
        container_kwargs={'image_pull_policy':"IfNotPresent"},
        pvolumes={
            "/data": dsl.PipelineVolume(pvc="benchmark-data"),
            '/data/x_train.pkl': v1op.volume,
            '/data/y_train.pkl': v2op.volume,
            '/data/x_test.pkl': v3op.volume,
            '/data/y_test.pkl': v4op.volume
        },
        command = ['python3', 'preprocess.py']
    )

def Train(comp1):
    return dsl.ContainerOp(
        name='Train Data',
        image='train-data',
        container_kwargs={'image_pull_policy':"IfNotPresent"},
        pvolumes={
            "/data": comp1.pvolumes['/data'],
            '/data/x_train.pkl': comp1.pvolumes['/data/x_train.pkl'],
            '/data/y_train.pkl': comp1.pvolumes['/data/y_train.pkl']
        },
        command = ['python3', 'train.py'],
    )

def Test(comp1, comp2):
    return dsl.ContainerOp(
        name='Test Data',
        image='test-data',
        container_kwargs={'image_pull_policy':"IfNotPresent"},
        pvolumes={
            "/data": comp2.pvolumes['/data'],
            '/data/x_test.pkl': comp1.pvolumes['/data/x_test.pkl'],
            '/data/y_test.pkl': comp1.pvolumes['/data/y_test.pkl']
        },
        command = ['python3', 'test.py']
    )

@dsl.pipeline(
   name='Machine learning Pipeline',
   description='An example pipeline that trains and logs a regression model.'
)
def pipeline():
    comp1 = PreProcess()
    comp2 = Train(comp1)
    comp3 = Test(comp1, comp2)

if __name__ == '__main__':
    from kfp import compiler
    compiler.Compiler().compile(pipeline, __file__[:-3]+ '.yaml')
