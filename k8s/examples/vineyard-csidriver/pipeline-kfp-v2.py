from kfp import dsl
from kfp import kubernetes

@dsl.container_component
def PreProcess(data_multiplier: int):
    return dsl.ContainerSpec(
        image = 'ghcr.io/v6d-io/v6d/csidriver-example/preprocess-data',
        command = ['python3', 'preprocess.py'],
        args=[f'--data_multiplier={data_multiplier}'],
    )

@dsl.container_component
def Train():
    return dsl.ContainerSpec(
        image='ghcr.io/v6d-io/v6d/csidriver-example/train-data',
        command = ['python3', 'train.py'],
    )

@dsl.container_component
def Test():
    return dsl.ContainerSpec(
        image='ghcr.io/v6d-io/v6d/csidriver-example/test-data',
        command = ['python3', 'test.py'],
    )

def mount_pvc(component, pvc_name):
    kubernetes.mount_pvc(
        component,
        pvc_name=pvc_name,
        mount_path='/data',
    )

@dsl.pipeline(
   name='Machine Learning Pipeline',
   description='An example pipeline that trains and logs a regression model.'
)
def pipeline(data_multiplier: int):
    comp1 = PreProcess(data_multiplier=data_multiplier)
    mount_pvc(comp1, "benchmark-data")
    comp2 = Train().after(comp1)
    mount_pvc(comp2, "benchmark-data")
    comp3 = Test().after(comp2)
    mount_pvc(comp3, "benchmark-data")

if __name__ == '__main__':
    from kfp import compiler
    compiler.Compiler().compile(pipeline, __file__[:-3]+ '.yaml')
