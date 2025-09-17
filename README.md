# Neural_Processing
Program to perform spike sorting for Neuralynx (.ncs) files, handling single session recordings as well as spike template comparison across multiple recording sessions.

Set Up:
You will need to download Neo (NeuralEnsemble) from the GitHub link execute the following in your terminal:

pip install git+https://github.com/NeuralEnsemble/python-neo.git

***Note: pip installing Neo directly will not work because PyPI hasn’t updated the version to work with our type of .ncs files yet***
You can pip install the remaining required libraries.

How To Use:
There are 2 scripts: SpikeSorting.py handles single recordings with multiple channels, and compare_templates.py correlates the spike templates that SpikeSorting.py creates for each channel of each recording session.

SpikeSorting.py:
In quotes, write the directory of the folder containing the .ncs files for a recording session into line 466 where it says “data_folder = ”. Run the program. It will begin by performing common average referencing (CAR) across all channels (you will be able to exclude channels from consideration later if you wish). It will create a folder named “results_initial” and save outputs there. It will look for spikes across all channels, and save all spike times to a CSV. It will also save spike templates to a .npy file (you will use this for the compare_templates.py script), and save quality metrics to a JSON file. It will then begin the quality control phase and open a new window with spike graphs and quality metrics for each channel. Use the slider on the bottom of the window to change channel, and click “accept” or “reject” for each channel after evaluating the graphs and quality metrics. After accepting or rejecting all channels, click “save all”. Once you exit the quality control window, the program will continue, and if you rejected any channels it will offer to run the spike sorting again excluding the rejected channel(s) and you will see the quality control window again but this time the outputs will be saved to a folder named “results_clean”. 

Compare_templates.py:
At the bottom of the code, under “compare_templates_across_sessions([”, write the directories of the template files you want to compare (there are commented examples, make sure to end the file name with .npy). Run the program. It will create a folder called “template_comparison” and will save images which show superimposed spike templates as well as correlations between templates of the same channel across multiple sessions.

