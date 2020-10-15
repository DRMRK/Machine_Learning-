export ORIGINAL_DATA=data/original/Trips2016_10.pkl
export CLEANED_DATA=data/processed/cleaned1.csv
export REPORT1=results/scores_voting.png
#export TEST_DATA=input/test.csv
#export FOLD=0
export MODEL=$1
#python -m src.clean_data
python -m src.train