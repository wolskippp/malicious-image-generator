# malicious-image-generator
Intro to evolutionary algorithms

How to run the program:

...to generate an image which is no longer recognizable by Inception-V3:
1. Insert image recognizable by Inception-V3 into test_images folder.
2. Setup problem parameters in Config.py
3. If any evolutionary algorithm parameters must be changed - change them in src\main.py.
4. Run `python src\main.py` using Python3.

...to analyze different sets of parameters:
1. Insert image recognizable by Inception-V3 into test_images folder.
2. Setup problem parameters in Config.py
3. Setup evolutionary algorithm parameters to test in src\params_analysis.py.
4. Run `python src\params_analysis.py` using Python3.

Output (generated images, charts and execution summary) will be placed in the folder output\[running_datetime].