from setuptools import find_packages, setup

package_name = 'arm_lab_kinematics'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy', 'pyyaml'],
    zip_safe=True,
    maintainer='farouk',
    maintainer_email='farouk15160@gmail.com',
    description='Kinematics, workspace, accuracy and motion analysis for the rover arm.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ik_check = arm_lab_kinematics.cli:ik_check_main',
            'workspace = arm_lab_kinematics.cli:workspace_main',
            'singularity = arm_lab_kinematics.cli:singularity_main',
            'iso9283 = arm_lab_kinematics.cli:iso9283_main',
            'cartesian_plan = arm_lab_kinematics.cli:cartesian_main',
            'topp = arm_lab_kinematics.cli:topp_main',
            'collision_check = arm_lab_kinematics.cli:collision_main',
            'moveit_gen = arm_lab_kinematics.cli:moveit_main',
            'safety_map = arm_lab_kinematics.cli:safety_main',
            'pick_place = arm_lab_kinematics.pick_place:main',
            'cartesian_move = arm_lab_kinematics.cartesian_node:main',
            'workspace_markers = arm_lab_kinematics.workspace_markers:main',
        ],
    },
)
