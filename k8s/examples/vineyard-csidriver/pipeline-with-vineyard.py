from kfp import dsl

def PreProcess(data_multiplier: int):
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
        command = ['python3', 'preprocess.py'],
        arguments=[f'--data_multiplier={data_multiplier}', '--with_vineyard=True'],
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
        arguments=['--with_vineyard=True'],
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
        command = ['python3', 'test.py'],
        arguments=['--with_vineyard=True'],
    )

@dsl.pipeline(
   name='Machine learning Pipeline',
   description='An example pipeline that trains and logs a regression model.'
)
def pipeline(data_multiplier: int):
    comp1 = PreProcess(data_multiplier=data_multiplier)
    comp2 = Train(comp1)
    comp3 = Test(comp1, comp2)

if __name__ == '__main__':
    from kfp import compiler
    compiler.Compiler().compile(pipeline, __file__[:-3]+ '.yaml')
