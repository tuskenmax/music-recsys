from src.commons.data_utils import load_latest_model
from src.config_files.config import DEVICE
from src.commons.data_utils import encode_sentence, load_data_songs, load_data_queries
import torch

model = load_latest_model()
# model = torch.load('data\models\model.pkl')
sentence = """sad bith that is done im dead"""
emb = encode_sentence(sentence)
input_tensor = emb.to(DEVICE)
input_tensor = input_tensor.clone().detach().unsqueeze(0).float()
model.eval()

with torch.no_grad():
    output = model(input_tensor)

output = output.squeeze().tolist()
top_indices = sorted(range(len(output)), key=lambda i: output[i], reverse=True)[:10]
print(top_indices)

df_songs = load_data_songs()
for index in top_indices:
    print(df_songs.loc[index, 'url'])

print(df_songs)