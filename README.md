# Udacity's Data Scientist Nanodegree Capstone

## Convolutional Neural Networks for a Dog Identification App

## Project Description
The Capstone Project for Dog Identification is one of the most popular Udacity projects across Machine Learning and Artificial Intellegence Nanodegree Programs. Powerful ideas and approaches for addressing task classification are provided by the burgeoning and amazing fields of Deep Learning and Artificial Neural Networks. In this project, we have used deep learning principles to create concepts for a dog recognition app. The software is designed to accept any image input from the user. If a dog is found in the picture, an estimation of the breed will be given. If a human is detected, it will estimate the dog breed that most closely resembles a human.

## Project Motivation
Dog breed identification is crucial when rescuing dogs, providing them loving families, rehabilitating them, and several other  circumstances because dogs are frequently difficult to categorise merely by looking. The classification of a dog's breeds reveals more about the personality of the breed. Our fondness and passion for dogs inspired us to choose this initiative.

https://huggingface.co/spaces/mehzhats/dogbreedidentifier

The steps involved in the project are as follows:

## Project Components
The project is divided in three major parts

### 1. ETL Pipeline
Data Processing Pipeline, to load data from source, merge and clean the data and save it in a SQLite database.

### 2. ML Pipeline
Machine Learning Pipeline, to load preprocessed data from the SQLite database, split the dataset into training and test sets, build and train the model using GridSearchCV and export the final model as pkl file.

### 3. Flask Web App
Web App to show data visualisation of the trained data using Plotly. An interactive portal is available to use model in real time.

## Getting Started

### Dependencies

1. Python 3.10.4
2. Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
3. Natural Language Process Libraries: NLTK
4. SQLlite Database Libraqries: SQLalchemy
5. Web App and Data Visualization: Flask, Plotly

A requirements.txt file is created to install all the dependencies needed for this project. To install them run following command.

```pip install -r requirements.txt```

## Authors
* [MehzHats](https://github.com/MehzHats)

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## Screenshots

1. This is an example of a message you can type to test Machine Learning model performance.

![message_result](images/message_result.png)

2. Data visualisation for the least 10 categories.

![least_categories](images/least_categories.png)

3. Data visualisation for the message category distribution.

![least_categories](images/message_cat_dist.png)

4. Data visualisation for the words count distribution.

![least_categories](images/word_count_dist.png)

5. The f1 score, precision and recall for the test set output is shown for each category.

model_metrics_1             |  model_metrics_2
:-------------------------:|:-------------------------:
![](images/model_metrics/model_metrics_1.png )  |  ![](images/model_metrics/model_metrics_2.png)


model_metrics_3             |  model_metrics_4
:-------------------------:|:-------------------------:
![](images/model_metrics/model_metrics_3.png )  |  ![](images/model_metrics/model_metrics_4.png)


model_metrics_4             |  model_metrics_6
:-------------------------:|:-------------------------:
![](images/model_metrics/model_metrics_5.png )  |  ![](images/model_metrics/model_metrics_6.png)


model_metrics_7             |  model_metrics_8
:-------------------------:|:-------------------------:
![](images/model_metrics/model_metrics_3.png )  |  ![](images/model_metrics/model_metrics_4.png)

## Acknowledgements

* [Udacity](https://www.udacity.com/) Data Science Nanodegree Program
* [Appen](https://appen.com/) Data from Appen (formally Figure 8) to build a model for an API that classifies disaster messages.