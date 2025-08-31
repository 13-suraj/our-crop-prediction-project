from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path:str) -> List[str]:
    '''
    This function will return a list of moduleswhich is required for the project.
    '''
    requirements = []
    HYPEN_E_DOT = '-e .'
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
    name = 'our-crop-prediction',
    version = '1.0.0',
    author = 'Suraj Keshari, Abhishek Singh, Aniket Chakraborty',
    author_email = 'surajkeshari260@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)