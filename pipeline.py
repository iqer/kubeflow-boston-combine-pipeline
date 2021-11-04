import kfp
from kfp import dsl


def train_mnist_op():
    return dsl.ContainerOp(name='boston', image='shanau2/boston_pipeline_combine:v0.1.0')


@dsl.pipeline(
    name='boston combine train',
    description='test'
)
def pipeline():
    op = train_mnist_op()


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(pipeline, __file__ + '.zip')
