# The Podcast Copilot will automatically create and post a LinkedIn promotional post for a new episode of the Behind the Tech podcast.  
# Given the audio recording of the episode, the copilot will use a locally-hosted Whisper model to transcribe the audio recording.
# The copilot uses the Dolly 2 model to extract the guest's name from the transcript.
# The copilot uses the Bing Search Grounding API to retrieve a bio for the guest.
# The copilot uses the GPT-4 model in the Azure OpenAI Service to generate a social media blurb for the episode, given the transcript and the guest's bio.
# The copilot uses the DALL-E 2 model to generate an image for the post.
# The copilot calls a LinkedIn plugin to post.


import os
from dotenv import load_dotenv
load_dotenv()
import openai


# Inputs about the podcast
podcast_url = "https://go.ivoox.com/sq/277993"
podcast_audio_file = "c:\\src\\labs\\PodcastCopilot\\NTN160.mp3" 

# open textfile NTN160.txt and read into variable transcript
with open('NTN160.txt', 'r') as file:
    transcript = file.read().replace('\n', '')

# print the length of the transcript
print("Length of transcript: " + str(len(transcript)) + "\n")

# make a string variable named ppp a multiline string
bio = "My name is Bruno Capuano! Sr Cloud Advocate at Microsoft"

# Step 4 - Put bio in the prompt with the transcript
system_template="You are a helpful large language model that can create a LinkedIn promo blurb for episodes of the podcast ""No Tiene Nombre"", when given transcripts of the podcasts.  The ""No Tiene Nombre"" podcast is hosted by Bruno Capuano].\n"

question="Create a short summary of this podcast episode that would be appropriate to post on LinkedIn to promote the podcast episode.  The post should be from the first-person perspective of Bruno Capuano, who hosts the podcast.\n Here is the transcript of the podcast episode: {transcript} \n Here is the bio of the guest: {bio} \n"

openai.api_key = os.getenv("OPENAI_API_KEY")

completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": system_template, "content": question}
  ]
)

resp = completion.choices[0].message
print(resp)