#    Copyright (c) 2023
#    Author      : Bruno Capuano
#    Change Log  :
#
#    The MIT License (MIT)
#
#    Permission is hereby granted, free of charge, to any person obtaining a copy
#    of this software and associated documentation files (the "Software"), to deal
#    in the Software without restriction, including without limitation the rights
#    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#    copies of the Software, and to permit persons to whom the Software is
#    furnished to do so, subject to the following conditions:
#
#    The above copyright notice and this permission notice shall be included in
#    all copies or substantial portions of the Software.
#
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#    THE SOFTWARE.

from pydub import AudioSegment
from pydub.silence import split_on_silence
import whisper

import time

# Call Whisper to transcribe audio
print("Calling Whisper to transcribe audio...\n")

# add a start time flag
start_time = time.time()
print(f"Start time: {start_time} seconds\n")

# Inputs about the podcast
podcast_name = "No Tiene Nombre"
podcast_episode_name = "NTN160"
podcast_author = "Bruno Capuano"
podcast_url = "https://go.ivoox.com/sq/277993"
podcast_audio_file = ".\\NTN160.mp3" 

# Chunk up the audio file 
sound_file = AudioSegment.from_mp3(podcast_audio_file)
audio_chunks = split_on_silence(sound_file, min_silence_len=1000, silence_thresh=-40 )
count = len(audio_chunks)
print("Audio split into " + str(count) + " audio chunks \n")

# Call Whisper to transcribe audio
model = whisper.load_model("base")
transcript = ""
for i, chunk in enumerate(audio_chunks):
    # If you have a long audio file, you can enable this to only run for a subset of chunks
    if i < 10 or i > count - 10:
        out_file = "chunk{0}.wav".format(i)
        print("\r\nExporting >>", out_file, " - ", i, "/", count)
        chunk.export(out_file, format="wav")
        result = model.transcribe(out_file)
        transcriptChunk = result["text"]
        print(transcriptChunk)
        
        # Append transcript in memory if you have sufficient memory
        transcript += " " + transcriptChunk

# Print the transcript
print("Transcript: \n")
print(transcript)
print("\n")

# let's write the transcript to disk, for future exercises
transcript_filename = f"{podcast_episode_name}.txt"
textfile = open(transcript_filename, "w" , encoding='utf-8')
transcript_to_txt = transcript.encode("utf-8")
textfile.write(transcript)
textfile.close()
print(f"Transcript saved to {textfile.name} \n")


# calculate the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time

# print the elapsed time
print(f"Elapsed time: {elapsed_time} seconds")