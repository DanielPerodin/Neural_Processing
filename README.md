# Neural_Processing
Program to perform spike sorting for Neuralynx (.ncs) files, handling single session recordings as well as spike template comparison across multiple recording sessions.

Set Up:

You will need to download Neo (NeuralEnsemble) from the GitHub link execute the following in your terminal:

pip install git+https://github.com/NeuralEnsemble/python-neo.git

***Note: pip installing Neo directly will not work because PyPI hasn’t updated the version to work with our type of .ncs files yet***

You can pip install the remaining required libraries.

How To Use:

There are 2 scripts: clustering6.py handles single recordings (what we’re calling “days”) with multiple channels, and compare_templates.py correlates the spike templates that clustering6.py creates for each cluster on all channels for each recording session (day).

clustering6.py:

In quotes, write the directory of the folder containing the .ncs files for a recording session into line 673 where it says “data_folder = ”. Run the program. It will begin by asking if you would like to reject any channels from processing. It will then ask if you want to make groups for CAR. It will then begin performing common average referencing (CAR) in the specified groups or across all channels depending on your choices. It will create a folder named “clustering_results” and save outputs there.


Compare_templates.py:

At the bottom of the code, under “compare_templates_across_sessions([”, write the directories of the template files you want to compare (there are commented examples, make sure to end the file name with .npy). Run the program. It will create a folder called “template_comparison” and will save images which show superimposed spike templates as well as correlations between templates of the same channel across multiple sessions.

