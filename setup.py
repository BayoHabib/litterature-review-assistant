
from setuptools import setup, find_packages

setup(
    name='literature_review_assistant',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pyyaml',
        'pytest',
        'torch',
        'transformers',
        'langchain',
        'llamaindex',
    ],
)
