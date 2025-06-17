from setuptools import find_packages, setup
import os

# Function to read requirements from requirements.txt
def parse_requirements(filename="requirements.txt"):
    """Load requirements from a pip requirements file."""
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

# Get the long description from the README file if you have one
# here = os.path.abspath(os.path.dirname(__file__))
# try:
#     with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
#         long_description = f.read()
# except FileNotFoundError:
#     long_description = "ESMM Recommender Pipeline using EasyRec"


# Read requirements
requirements = parse_requirements()

setup(
    name='esmm_recommender',
    version='0.1.0',
    description='ESMM Recommender Pipeline using EasyRec',
    # long_description=long_description,
    # long_description_content_type='text/markdown',
    author='AI SWE Agent', # Placeholder
    author_email='agent@example.com', # Placeholder
    url='http://localhost/', # Placeholder for project URL
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True, # To include files specified in MANIFEST.in
    install_requires=requirements,
    python_requires='>=3.7, <3.11', # Example range, adjust based on EasyRec/TF compatibility
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License', # Example, choose your license
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10', # Check EasyRec compatibility for newer Python versions
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    entry_points={
        'console_scripts': [
            'esmm_split_data=data_processing.split_dataset:main',
            'esmm_generate_features=data_processing.auto_feature_config_generator:main', # Assuming auto_feature_config_generator also has a main()
            'esmm_generate_train_config=training.generate_and_run_train:main',
            'esmm_evaluate_model=evaluation.evaluate_model:main',
        ]
    },
    # If you have data files that need to be included outside of packages, like config templates
    # data_files=[('configs', ['configs/esmm_pipeline_template.config'])], # This is often for system-wide installs.
                                                                      # MANIFEST.in is better for sdist.
)
