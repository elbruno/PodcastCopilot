# The Podcast Copilot will automatically create and post a LinkedIn promotional post for a new episode of the Behind the Tech podcast.  
# Given the audio recording of the episode, the copilot will use a locally-hosted Whisper model to transcribe the audio recording.
# The copilot uses the Dolly 2 model to extract the guest's name from the transcript.
# The copilot uses the Bing Search Grounding API to retrieve a bio for the guest.
# The copilot uses the GPT-4 model in the Azure OpenAI Service to generate a social media blurb for the episode, given the transcript and the guest's bio.
# The copilot uses the DALL-E 2 model to generate an image for the post.
# The copilot calls a LinkedIn plugin to post.

from pydub import AudioSegment
from pydub.silence import split_on_silence
import whisper
import torch
from langchain.chains import TransformChain, LLMChain, SequentialChain
from langchain.chat_models import AzureChatOpenAI
from langchain.llms import HuggingFacePipeline
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import requests
import time
from PIL import Image
from io import BytesIO
import datetime
import json
from dalle_helper import ImageClient

from dotenv import load_dotenv
load_dotenv()


print("Imports are complete")


# Endpoint Settings
openai_api_type = os.getenv('openai_api_type')
openai_api_base = os.getenv(openai_api_base)
openai_api_key = os.getenv(openai_api_key)
gpt4_deployment_name = os.getenv(gpt4_deployment_name)

# Inputs about the podcast
podcast_url = "https://go.ivoox.com/sq/277993"
podcast_audio_file = "c:\\src\\labs\\PodcastCopilot\\NTN160.mp3" 

# Step 1 - Call Whisper to transcribe audio
print("Calling Whisper to transcribe audio...\n")

# Chunk up the audio file 
print("Splitting audio file into chunks...")
print(podcast_audio_file)
sound_file = AudioSegment.from_mp3(podcast_audio_file)
audio_chunks = split_on_silence(sound_file, min_silence_len=1000, silence_thresh=-40 )
count = len(audio_chunks)
print("Audio split into " + str(count) + " audio chunks")

# Call Whisper to transcribe audio
model = whisper.load_model("base")
transcript = ""
for i, chunk in enumerate(audio_chunks):
    # If you have a long audio file, you can enable this to only run for a subset of chunks
    if i < 10 or i > count - 10:
        out_file = "chunk{0}.wav".format(i)
        print("Exporting", out_file)
        chunk.export(out_file, format="wav")
        result = model.transcribe(out_file)
        transcriptChunk = result["text"]
        print(transcriptChunk)
        
        # Append transcript in memory if you have sufficient memory
        transcript += " " + transcriptChunk

        # Alternatively, here's how to write the transcript to disk if you have memory constraints
        textfile = open("chunk{0}.txt".format(i), "w")
        textfile.write(transcript)
        textfile.close()
        print("Exported chunk{0}.txt".format(i))

print("Transcript: \n")
print(transcript)
print("\n")


# Step 4 - Put bio in the prompt with the transcript
system_template="You are a helpful large language model that can create a LinkedIn promo blurb for episodes of the podcast No Tiene Nombre, when given transcripts of the podcasts.  The No Tiene Nombre podcast is hosted by Bruno Capuano].\n"
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

user_prompt=PromptTemplate(
    template="Create a short summary of this podcast episode that would be appropriate to post on LinkedIn to promote the podcast episode.  The post should be from the first-person perspective of Bruno Capuano, who hosts the podcast.\n" +
            "Here is the transcript of the podcast episode: {transcript} \n" +
            "Here is the bio of the guest: {bio} \n",
    input_variables=["transcript", "bio"],
)
human_message_prompt = HumanMessagePromptTemplate(prompt=user_prompt)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# Get formatted messages for the chat completion
blurb_messages = chat_prompt.format_prompt(transcript={transcript}, bio={bio}).to_messages()
print("Chat Prompt - blurb_messages: \n")
print(blurb_messages)

# # Step 5 - Make a call to Azure OpenAI Service to get a social media blurb, 
# print("Calling GPT-4 model on Azure OpenAI Service to get a social media blurb...\n")
# gpt4 = AzureChatOpenAI(
#     openai_api_base=gpt4_endpoint,
#     openai_api_version="2023-03-15-preview",
#     deployment_name=gpt4_deployment_name,
#     openai_api_key=gpt4_api_key,
#     openai_api_type = openai_api_type,
# )
# #print(gpt4)   #shows parameters

# output = gpt4(blurb_messages)
# social_media_copy = output.content

# gpt4_chain = LLMChain(llm=gpt4, prompt=chat_prompt, output_key="social_media_copy")

# print("Social Media Copy:\n")
# print(social_media_copy)
# print("\n")


