from setuptools import setup, find_packages

setup(
    name='comfyui-voxel-nodes',
    version='0.1.0',
    description='Custom voxel/3D nodes for ComfyUI.',
    author='dexintenebri',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'Pillow',
        'midvoxio',
        'trimesh',
        'scikit-image',
        'scikit-learn',
        'perlin-noise',
        'scipy',
    ],
    python_requires='>=3.8',
    include_package_data=True,
    url='https://github.com/dexintenebri/comfyui_voxel_nodes',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License'
    ],
)
