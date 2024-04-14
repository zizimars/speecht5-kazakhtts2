import os

base_directory_path = "/home/zhuldyz/Downloads/ISSAI_KazakhTTS2"
idSpeaker = os.listdir(base_directory_path)
idSpeaker = idSpeaker[:-1]
list1 = []
list2 = []
# print(idSpeaker)
for id in idSpeaker:
    if id == "F2":
        directory_speaker = os.path.join(base_directory_path, id)
        audio_dir = os.path.join(directory_speaker, "Audio")
        audiofiles = os.listdir(audio_dir)
        transcripts_dir = os.path.join(directory_speaker, "Transcripts")
        transcripts = os.listdir(transcripts_dir)

        for i in range(len(audiofiles)):
            audio = os.path.join(audio_dir, audiofiles[i])
            # transcript = os.path.join(transcripts_dir, transcripts[i])
            aud = audiofiles[i].split('.')
            # Path
            list1.append(aud[0])
            # audio_id
            try:
                tra = transcripts[i].split('.')
                list2.append(tra[0])
            except IndexError:
                print(i)
audio_to_delete = set(list1) - set(list2)
for i in audio_to_delete:
    os.remove(os.path.join(audio_dir, i + '.wav')) 
    
print(len(os.listdir(audio_dir)))
