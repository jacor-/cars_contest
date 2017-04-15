# Start the virtualenvironment
export ENV_NAME=kgcars
conda env create -n $ENV_NAME -f environment.yml 2> /dev/null
source activate $ENV_NAME

## Be sure jupyter is available
## conda install jupyter
ipython kernel install --name "Virtulenv_Python_3" --user

# System path variables
export CARS_PATH=/home/jose/tech/ml_projects/kaggle/cars_contest
export DATA_PATH=/home/jose/tech/ml_projects/datasets/cars_contest
export DATAMODEL_PATH=/home/jose/tech/ml_projects/data_models/cars_contest
