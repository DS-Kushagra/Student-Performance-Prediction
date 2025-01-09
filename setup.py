from setuptools import find_packages,setup
from typing import List


E_DOT = "-e ."
def get_requires(file_path:str)->List[str]:
    requirements = []
    with open(file_path) as f:
        requirements = f.readlines()
        requirements=[req.replace("\n", "") for req in requirements]
        
        if E_DOT in requirements:
            requirements.remove(E_DOT)

    return requirements 


setup(
    name="ML-Project",
    version='0.0.1',
    author="Kushagra",
    author_email="kushagraagrawal128@gmail.com",
    packages= find_packages(),
    install_requires= get_requires('requirements.txt')
)