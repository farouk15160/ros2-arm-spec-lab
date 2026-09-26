from glob import glob
from pathlib import Path

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
        ('share/' + package_name + '/config/pipeline', glob('config/pipeline/*.yaml')),
        *[(str(Path('share') / package_name / directory),
           [str(path) for path in Path(directory).iterdir() if path.is_file()])
          for directory in sorted({str(path.parent) for path in Path('config/benchmarks').rglob('*') if path.is_file()})],
    ],
    install_requires=['setuptools', 'numpy', 'pyyaml'],
    extras_require={'simulation': ['mujoco>=3.2,<4'], 'reports': ['matplotlib>=3.6,<4']},
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
            'verify_physics = arm_lab_model.verification:main',
            'engineering_report = arm_lab_model.engineering_report:main',
            'robot_test = arm_lab_model.mujoco_backend:main',
            'project_check = arm_lab_model.project_config:main',
            'robot_pipeline = arm_lab_model.pipeline_cli:main',
        ],
    },
)
