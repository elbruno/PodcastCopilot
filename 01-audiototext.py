from pydub import AudioSegment
from pydub.silence import split_on_silence
import whisper

print("Imports are complete")

# Inputs about the podcast
podcast_name = "No Tiene Nombre"
podcast_author = "Bruno Capuano"
podcast_url = "https://go.ivoox.com/sq/277993"
podcast_audio_file = "c:\\src\\labs\\PodcastCopilot\\NTN160.mp3" 

# Call Whisper to transcribe audio
print("Calling Whisper to transcribe audio...\n")

# Chunk up the audio file 
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