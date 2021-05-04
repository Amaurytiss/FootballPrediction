# FootballPrediction
We are trying to predict the results of football games by using machine learning algorithms

This repository is kinda messy but you can try to run the notebook in RenduProjetFootball

Made by Amaury Tisseau and Gaspard Canevet

## Intro

During my Data Science Master at IMT Atlantique, we had to do a Data Science / Machine Learning project. We decided to work on sports bet, especially football betting. We asked ourselves if it was possible to predict the outcomes of football matches with enough accuracy to win money on betting sites.

## Starting Point

As we are french students, the league we are the most aware of is the French ligue 1. So we decided to work only on this one at first.
We found some really good datasets on Football-data.uk, with a lot of features which would be perfect to start with.
Predicting the outcome of a football game is in fact a classification problem, we only had to tell if a game is a win, a draw or a loose.

## Data processed

To train our models, we first need to have our data ready. For a football game, we cannot know in advance how many goals or shoot every team is going to score. So we chose to represent each game by the average stats over the season of both teams playing. The processed dataset would look like that (only the first columns here)

[[https://github.com/Amaurytiss/FootballPrediction/blob/main/photos%20wiki/Capture.JPG|alt=dataset]]

## Models

We used many different Machine learning models, but the more efficient one on this task seemed to be the Random Forest Classifier. After a few tries, we figured out that training on two seasons, to predict the third one was the most efficient way.

Here is a confusion matrix for one of our Random Forest model. 2 is for HomeTeam win, 1 is for Draw and 0 is for AwayTeam win
The score is the sklearn.score for classifier, which is the harsh accuracy.

[[https://github.com/Amaurytiss/FootballPrediction/blob/main/photos%20wiki/forest.jpg|alt=forest]]

## Adding Features

To improve our results, we thought about adding some more features to our datasets, to see if we could find one which allows our model to be more accurate. We tried the Win / loose streak of each team, the year budget of the team, the FIFA videogames score of each teams, and the average public of teams. We tried every possible combination added to the original dataset.

[[https://github.com/Amaurytiss/FootballPrediction/blob/main/photos%20wiki/add_feat.JPG]]

So best combination was the one with the budget and the public. So we kept using this combination.

Here are some predictions to show how we could use them

[[https://github.com/Amaurytiss/FootballPrediction/blob/main/photos%20wiki/predic.JPG]]

If we wanted to follow the predictions as some advices, we would need more than only a simple result. That's why we also used the RandomForestClassfier.proba_predict to have the certainty of the prediction.

[[https://github.com/Amaurytiss/FootballPrediction/blob/main/photos%20wiki/proba.JPG]]

## Betting simulation

In order to test our results, we tryed to simulate a year of betting on our predictions. We juste used the odds from Bet356 which were already in our datasets. Here are 10 simulations where we bet 10€ on every game.

[[https://github.com/Amaurytiss/FootballPrediction/blob/main/photos%20wiki/simu.JPG]]

We can see that 9 out of 10 times we are winning money, but there is once where we loose. This shows that despite the results we obtain, it is still too uncertain to really earn money with that.

## Odd scrapping

If we wanted to use these algorithms in real life, we would need to see which is the best betting site for each game. So we decided to do some web scrapping algorithms that can get the odds from different websites directly. We did that using selenium and chromedriver

[[https://github.com/Amaurytiss/FootballPrediction/blob/main/photos%20wiki/odd.JPG]]

