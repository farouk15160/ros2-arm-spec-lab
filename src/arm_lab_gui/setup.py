from setuptools import find_packages, setup

package_name = 'arm_lab_gui'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='farouk',
    maintainer_email='farouk15160@gmail.com',
    description='Capability dashboard and analysis nodes for the rover arm.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dashboard = arm_lab_gui.dashboard:main',
            'config_editor = arm_lab_gui.config_editor:main',
            'capability_node = arm_lab_gui.capability_node:main',
            'speed_test = arm_lab_gui.speed_test:main',
        ],
    },
)
