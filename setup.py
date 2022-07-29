from setuptools import setup, find_packages
from pathlib import Path

with open('requirements.txt') as f:
    required = f.read().splitlines()

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='rescompy',
      version='0.1.0',
      description='REServoir COMputing for PYthon',
      long_description=long_description,
      long_description_content_type='text/markdown',
      classifiers=[],
      keywords='Machine Learning, Reservoir Computing',
      url='https://github.com/PotomacResearch/rescompy',
      author='Daniel Canaday',
      author_email='daniel.marcus.canaday@gmail.com',
      license='Proprietary',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=required,
      )