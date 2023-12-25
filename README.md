# MUSIC RECOMMENDATION SYSTEM BASED ON LÜSCHER TEST

The following research is to present a Personalized Recommendation System, the output of which will be based upon the results of the Lüscher Color Test - the exact emotions of the taker are assigned to a specific choice of colors, presented to them to choose the ones they may find attractive - taken by a given user. 
A brief description of the psychological state is given afterwards. The aim of the project is to bring these descriptive results simpler range of emotions, where the latter are to be assigned to specific music recommendations, e.g., songs, sorted by types of emotions.

# Methods
The initial dataset has songs, which are each evaluated in 6 criteria by sound analysis. 
A LLM - in this case GPT-4.0 is used to create the training data - text prompts. These are than processed by BERT and a MLP to train the recommender, which learns to assosiate textual with the original criteria.

<div align="center">
  <img src="https://github.com/tuskenmax/music-recsys/blob/main/resources/luscher.png" alt="Lüscher Color Test" width="500" />
</div>

<p align="center"><em>Lüscher Color Test</em></p>
User is asked to choose the colors they like. Each time a user selects their favourite, it gets removed and the operation continues until only one - least likeable is left.
The colors are then shuffled and the test is carried out once again.

</p>
</p>
<p align="center">
  Training the Recommender Model
</p>

<p align="center">
  <img src="https://github.com/tuskenmax/music-recsys/blob/main/resources/trainl.png" alt="Training Loss" width="250" />
  <img src="https://github.com/tuskenmax/music-recsys/blob/main/resources/vall.png" alt="Validation Loss" width="250" />
</p>

<p align="center">
  <em>Training Loss (left)</em>
  <span style="margin-left: 100px;"></span>
  <span style="margin-left: 100px;"></span>
  <em>Validation Loss (right)</em>
</p>

# Results
The created model for classification the textual inputs into emotional criteria has achieved the following results:
- Mean Absolute Error (MAE): 0.0659
- Accuracy: 0.98
- Precision: 0.98
These metrics define that the suggested model as a combination of BERT and MLP shows good results in classifying a user input.

# Conclusion
This research links the emotions defined by the result from the Lüscher Color Test with music preferences.

Using GPT-4.0 for creating training data, BERT, and MLP to train the model, it connects color choices to song attributes, generating personalized music recommendations.

The code can be accessed [here](https://github.com/tuskenmax/music-recsys/blob/main/resources/music%20recommender%20based%20on%20prompt.ipynb)
