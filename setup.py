from setuptools import setup

setup(
    setup_requires=['setuptools_scm'],
    name='presumm-server',
    entry_points={
        'console_scripts': [
            'presumm-server=src.serve:serve'
        ],
    },
    install_requires=[
        "captum",
        "click",
        "click_completion",
        "logbook",
        "flask",
        "torch",
        "transformers",
    ],
)
