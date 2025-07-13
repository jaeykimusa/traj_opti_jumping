from setuptools import setup, find_packages

setup(
    name='go2', # This is the name your package will be known by
    version='0.1.0', # A version number for your package
    author='Jaeyoon Kim',
    author_email='jkim232@ncsu.edu',
    packages=find_packages(), # Automatically finds all packages and subpackages
    # You can add other metadata here if you want, like:

    # description='A Python package for Go2 robot simulations',
    # long_description=open('README.md').read(),
    # long_description_content_type='text/markdown',
    url='https://github.com/jaeykimusa/traj_opti_jumping',
    # classifiers=[
    #     'Programming Language :: Python :: 3',
    #     'License :: OSI Approved :: MIT License',
    #     'Operating System :: Ububtu 22.04.4 LTS (Jammy Jellyfish)',
    # ],
    # python_requires='>=3.8', # Specify minimum Python version
    # install_requires=[ # List any external dependencies your code needs
    #     'pinocchio',
    #     'numpy',
    #     # 'jax', # If you use JAX, list it here
    #     # etc.
    # ],
)