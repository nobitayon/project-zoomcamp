# Description

FIFA19 is video game series about football produced by EA SPORTS .
This game provide estimation of player's attribute,that correspond 
to technical abilities  like dribbling,shooting etc or non technical
thing like international reputation , work rate.Based on some weighting
on attribute ,that depend on player's position , overall rating of player calculated 
.On this project , i want to predict overall rating of player based on some
technical and non technical abilities that provided. The data can be downloaded from 
https://www.kaggle.com/karangadiya/fifa19. Information about attribute can be seen on 
https://fifauteam.com/fifa-19-attributes-guide/#22 . Position of football player can be seen on
https://sofifa.com/calculator

File in repository :
* train.py : to train the best model i choose , after run it save model on model_chosen.bin
* predict.py  : to deploy web service locally 
* notebook.ipynb : Model selection process
* predict-test.py : to try web service that deployed locally
* deploy-test.py  : to try web service that deployed on pythonanywhere
* prep.py : Module that needed on predict.py 
* Dockerfile : to running the service on docker
* Summary-model.xlsx : summary of model selection process
* Data.csv : Data i used to train and test on this project

# How to run project

## Starter
1. Download project 
2. Install dependencies with `pipenv install` 

## To run notebook.ipynb
1. Activate virtual environment in directory by `pipenv shell` and open jupyter notebook
2. Open notebook.ipynb
3. Run first section 1(Table of contents ) to section 5(EDA) . 
4. For section 6.1 and 6.2 , you can run 6.2 first than 6.1 , but in each subsection of 6.1 and 6.2 , must run sequentially
5. For section 6.2 to 9.4 , run first section between 6.2 and 6.3.No need to run sequentially in section 6.2 to 9.4 , but need to run sequentially in each section.

## To run train.py
1. Activate virtual environment in directory by `pipenv shell`
2. run `python train.py` , it will save model_chosen.bin in directory

## Deploy locally using docker
1. run docker image using `docker run -it --rm -p 9696:9696 zoomcamp-project`
2. run `python predict-test.py` on another command prompt 
3. Player data that specified in predict-test.py can modified if you want to try another player 

# Deploy the service on cloud
I deploy this project on pythonanywhere
[![](https://res.cloudinary.com/marcomontalbano/image/upload/v1635820806/video_to_markdown/images/google-drive--1C0cyzeLFY09PNtOt9RTxw2DTkOS-4ys2-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://drive.google.com/file/d/1C0cyzeLFY09PNtOt9RTxw2DTkOS-4ys2/view?usp=sharing "")


