from kfp import dsl
from kubernetes.client.models import V1EnvVar
import kubernetes as k8s

def PreProcess(data_multiplier: int, registry: str):
    vineyard_volume = dsl.PipelineVolume(
        volume=k8s.client.V1Volume(
            name="vineyard-socket",
            host_path=k8s.client.V1HostPathVolumeSource(
                path="/var/run/vineyard-kubernetes/vineyard-system/vineyardd-sample"
            )
        )
    )

    op = dsl.ContainerOp(
        name='Preprocess Data',
        image = f'{registry}/preprocess-data',
        container_kwargs={
            'image_pull_policy': "Always",
            'env': [V1EnvVar('VINEYARD_IPC_SOCKET', '/var/run/vineyard.sock')]
        },
        pvolumes={
            "/data": dsl.PipelineVolume(pvc="benchmark-data"),
            "/var/run": vineyard_volume,
        },
        command = ['python3', 'preprocess.py'],
        arguments=[f'--data_multiplier={data_multiplier}', '--with_vineyard=True'],
    )
    op.add_pod_label('scheduling.k8s.v6d.io/vineyardd-namespace', 'vineyard-system')
    op.add_pod_label('scheduling.k8s.v6d.io/vineyardd', 'vineyardd-sample')
    op.add_pod_label('scheduling.k8s.v6d.io/job', 'preprocess-data')
    op.add_pod_annotation('scheduling.k8s.v6d.io/required', '')
    return op

def Train(comp1, registry: str):
    op = dsl.ContainerOp(
        name='Train Data',
        image=f'{registry}/train-data',
        container_kwargs={
            'image_pull_policy': "Always",
            'env': [V1EnvVar('VINEYARD_IPC_SOCKET', '/var/run/vineyard.sock')]
        },
        pvolumes={
            "/data": comp1.pvolumes['/data'],
            "/var/run": comp1.pvolumes['/var/run'],
        },
        command = ['python3', 'train.py'],
        arguments=['--with_vineyard=True'],
    )
    op.add_pod_label('scheduling.k8s.v6d.io/vineyardd-namespace', 'vineyard-system')
    op.add_pod_label('scheduling.k8s.v6d.io/vineyardd', 'vineyardd-sample')
    op.add_pod_label('scheduling.k8s.v6d.io/job', 'train-data')
    op.add_pod_annotation('scheduling.k8s.v6d.io/required', 'preprocess-data')
    return op

def Test(comp2, registry: str):
    op = dsl.ContainerOp(
        name='Test Data',
        image=f'{registry}/test-data',
        container_kwargs={
            'image_pull_policy': "Always",
            'env': [V1EnvVar('VINEYARD_IPC_SOCKET', '/var/run/vineyard.sock')]
        },
        pvolumes={
            "/data": comp2.pvolumes['/data'],
            "/var/run": comp2.pvolumes['/var/run']
        },
        command = ['python3', 'test.py'],
        arguments=['--with_vineyard=True'],
    )
    op.add_pod_label('scheduling.k8s.v6d.io/vineyardd-namespace', 'vineyard-system')
    op.add_pod_label('scheduling.k8s.v6d.io/vineyardd', 'vineyardd-sample')
    op.add_pod_label('scheduling.k8s.v6d.io/job', 'test-data')
    op.add_pod_annotation('scheduling.k8s.v6d.io/required', 'train-data')
    return op

@dsl.pipeline(
   name='Machine Learning Pipeline',
   description='An example pipeline that trains and logs a regression model.'
)
def pipeline(data_multiplier: int, registry: str):
    comp1 = PreProcess(data_multiplier=data_multiplier, registry=registry)
    comp2 = Train(comp1, registry=registry)
    comp3 = Test(comp2, registry=registry)

if __name__ == '__main__':
    from kfp import compiler
    compiler.Compiler().compile(pipeline, __file__[:-3]+ '.yaml')
