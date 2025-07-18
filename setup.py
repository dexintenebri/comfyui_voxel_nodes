from setuptools import setup, find_packages

setup(
    name='comfyui-voxel-nodes',
    version='0.1.0',
    description='Custom voxel/3D nodes for ComfyUI including depth-to-voxel and WFC terrain generation.',
    author='dexintenebri',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'Pillow',
        'midvoxio',
        'trimesh',
    ],
    python_requires='>=3.8',
    include_package_data=True,
    url='https://github.com/dexintenebri/comfyui-voxel-nodes',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License'
    ],
)