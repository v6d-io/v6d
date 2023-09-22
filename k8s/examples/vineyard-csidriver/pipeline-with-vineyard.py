from kfp import dsl

def PreProcess():
    #############################################################
    # user need to add the volume Op for vineyard object manully
    vop = dsl.VolumeOp(name="vineyard-objects",
                    resource_name="vineyard-objects-pvc", size="1Mi",
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
            "/vineyard/data": vop.volume
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
            "/vineyard/data": comp1.pvolumes['/vineyard/data'],
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
            "/vineyard/data": comp1.pvolumes['/vineyard/data']
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
