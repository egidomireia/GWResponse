from setuptools import setup, find_packages

setup(
    name='gwresponse',
    version='0.1.0',
    packages=find_packages(),
    description='LISA waveform and response modeling toolkit',
    author='Mireia Egido',
    install_requires=[
        'jax',
        'jaxlib',
        'numpy',
        'matplotlib',
        'ipykernel'],
    include_package_data=True,
    package_data={
        'gwresponse': ['WFfiles/*.txt', 'WFfiles/*.h5'],
    },
)