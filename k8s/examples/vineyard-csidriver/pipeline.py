from kfp import dsl

def PreProcess():
    return dsl.ContainerOp(
        name='Preprocess Data',
        image = 'preprocess-data',
        container_kwargs={'image_pull_policy':"IfNotPresent"},
        command = ['python3', 'preprocess.py'],

        # add the existing volume to the pipeline
        pvolumes={"/data": dsl.PipelineVolume(pvc="benchmark-data")},
    )

def Train(comp1):
    return dsl.ContainerOp(
        name='Train Data',
        image='train-data',
        container_kwargs={'image_pull_policy':"IfNotPresent"},
        command = ['python3', 'train.py'],
        
        pvolumes={"/data": comp1.pvolumes['/data']},
    )

def Test(comp2):
    return dsl.ContainerOp(
        name='Test Data',
        image='test-data',
        container_kwargs={'image_pull_policy':"IfNotPresent"},
        command = ['python3', 'test.py'],

        pvolumes={"/data": comp2.pvolumes['/data']},
    )

@dsl.pipeline(
   name='Machine Learning Pipeline',
   description='An example pipeline that trains and logs a regression model.'
)
def pipeline():
    comp1 = PreProcess()
    comp2 = Train(comp1)
    comp3 = Test(comp2)

if __name__ == '__main__':
    from kfp import compiler
    compiler.Compiler().compile(pipeline, __file__[:-3]+ '.yaml')
