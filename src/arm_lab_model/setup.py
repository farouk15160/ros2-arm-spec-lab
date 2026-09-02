from glob import glob

from setuptools import find_packages, setup

package_name = 'arm_lab_model'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'numpy', 'pyyaml'],
    zip_safe=True,
    maintainer='farouk',
    maintainer_email='farouk15160@gmail.com',
    description='Config-driven rover arm model, analysis and URDF generation.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'spec_report = arm_lab_model.spec_report:main',
            'urdf_gen = arm_lab_model.cli:urdf_main',
            'controllers_gen = arm_lab_model.cli:controllers_main',
            'sweep = arm_lab_model.sweep:main',
        ],
    },
)
