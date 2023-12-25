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

# Usage example:
**A user takes a color test, than get's a result:**

Tries to escape from his problems, difficulties and tensions by abrupt, headstrong and ill-considered decisions or changes of direction.
Defensive. Feels his position is threatened or inadequately established. Determined to pursue his objectives despite the anxiety induced by opposition.
Physiological interpretation: Suppressed agitation resulting from unsatisfactory or discordant personal relationships. Can lead to irritability, angry outbursts or sexual neuroses. There is a possibility of cardiac complaints.
Psychological interpretation: Considerable distress is arising from some unsatisfactory relationship. He feels helpless to restore affinity and any semblance of mutual trust, so the situation is regarded as a depressing and unhappy state which he must continue to tolerate. Beset to the point of nervous prostration.
In brief: Helpless and irritable disharmony.
Physiological interpretation: Stress resulting from frustration in his attempts to achieve security and understanding.
Psychological interpretation: Is responsive to outside stimuli and wants to experience everything intensely, but is finding the existing situation extremely frustrating. Needs sympathetic understanding and a sense of security. Distressed by his apparent powerlessness to achieve his goals.

In brief: Frustrated empathy.

**The system processes the output and provides 10 songs, based on the user's current emotional state:**

1. https://open.spotify.com/track/48xNw8YZxvrc3qB5IKaJ8w
2. https://open.spotify.com/track/5BOMJwJhP84KsMLCqmTUYD
3. https://open.spotify.com/track/4uWjXFthU2ts7QfKZDWuwR
4. https://open.spotify.com/track/1ZqHjApl3pfzwjweTfMi0g
5. https://open.spotify.com/track/356dWMVMXLJI6OEoq0ITf1
6. https://open.spotify.com/track/5XDUR6VWpCtgv2hYH5GdMB
7. https://open.spotify.com/track/1IhnbJZ7FywEL7YCyLKDgv
8. https://open.spotify.com/track/0gEyKnHvgkrkBM6fbeHdwK
9. https://open.spotify.com/track/1y3r6RXiJZNBV1EI0NggpS
10. https://open.spotify.com/track/4lKWSv9JLE8B1YINRFv42O
