from setuptools import setup, find_packages

from .version import __version__

setup(
    name='ca_model',
    version=__version__,

    url='https://github.com/COSMOS-UALR/CA_Word_Characterisation.git',
    author='Ridwan Amure',
    author_email='raamure@ualr.edu',

    packages=find_packages(),

    install_requires = [
    "numpy==1.26.4",
    "pandas==2.2.2",
    "pdfkit==1.0.0",
    "spacy==3.7.4",
    "nltk==3.8.1",
    "transformers==4.39.3",
    "spacy",
    "tqdm==4.66.2",
    "requests==2.31.0",
    "unidecode==1.3.8",
    "PyYAML==6.0.1"
],
include_package_data=True,
package_data= {"ca_model":["data/updated_verbs_array.json"]}
)
