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



1. Download this repository
2. Install dependencies with `pipenv install` on directory 
3. Activate the virtual environment by `pipenv shell'
4. If you want to see notebook.ipynb , run `jupyter notebook` on virtual environment , and open the notebook.ipynb
5. Build docker image that included in repository by run `docker build -t zoomcamp-project .`
6. After building docker image , run it with `docker run -it --rm -p 9696:9696 zoomcamp-project`
