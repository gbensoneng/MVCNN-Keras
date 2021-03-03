# MVCNN-Keras
My attempt at implementing a multi-view CNN in Keras. The only implementation I could find for this was using pytorch.

Research paper on topic [here](http://vis-www.cs.umass.edu/mvcnn/).

## Important Notes:
- Project is tested against **Python 3.6** and **TensorFlow 2.1.0**
- This implementation deviates from the research paper in that it assumes individual views are well defined and predictable. This means we opt to train each singular view model on *only* images from that view.
- This implementation assumes use of the [dataset](http://maxwell.cs.umass.edu/mvcnn-data/modelnet40v1png/) used by the authors of the umass research paper. Any alteration to the dataset (i.e. file names, folder structure, etc.) will likely cause runtime errors. 
- There is conda environment file that can be used to [create the development environment](https://docs.conda.io/projects/conda/en/4.6.0/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) on your machine
	> conda env create -f env.yml

## How to Run:
1. Clone this repository to your machine
2. Recreate the conda environment from the .yml file
3. Download the umass dataset to your machine
4. Configure the **settings.ini** file to your liking
5. With your environment activated, run the **run.py** file
