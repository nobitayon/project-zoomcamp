# Description

FIFA is video game series about football produced by EA SPORTS .
This game provide estimation of player's attribute,that correspond 
to technical abilities  like dribbling,shooting etc or non technical
thing like international reputation , work rate.Based on some weighting
on attribute ,that depend on position , overall rating of player calculated 
.On this project , i want to predict overall rating of player based on some
technical and non technical abilities that provided. The data can downloaded from 
[kaggle](https://www.kaggle.com/karangadiya/fifa19).

# How to run project

## Starter
1. Download project 
2. Install dependencies with `pipenv install` 
3. Build docker image in directory by `docker build -t zoomcamp-project .`

## To run notebook.ipynb
1. Activate virtual environment in directory by `pipenv shell` and open jupyter notebook
2. Open notebook.ipynb
3. Run first section 1(Table of contents ) to section 5(EDA) . 
4. For section 6.1 and 6.2 , you can run 6.2 first than 6.1 , but in each subsection of 6.1 and 6.2 , must run sequentially
5. For section 6.2 to 9.4 , run first section between 6.2 and 6.3.No need to run sequentially in section 6.2 to 9.4 , but need to run sequentially in each section.

## To run train.py
1. Activate virtual environment in directory by `pipenv shell`
2. run `python train.py` , it will save model_chosen.bin

## Deploy locally using docker
1. run docker image using `docker run -it --rm zoomcamp-project`
2. run `python predict-test.py` on another command prompt 
3. Player data that specified in predict-test.py can modified if you want to try another player 

