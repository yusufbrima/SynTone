import numpy as np
import torch
from TTS.api import TTS
from pathlib import Path
# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
# print(TTS().list_models())

sentences = [
            "The road to fulfillment starts with a single step out of your comfort zone.",
            "Rome wasn't built in a day; success comes from perseverance through small daily improvements.",
            "Your mindset shapes your reality more than you know; stay positive and open to grow.",
            "Progress requires an openness to feedback; be ready to adapt and evolve your path.", 
            "Limiting beliefs can be shed through courage and an open heart; you contain unlimited potential.",
            "There is no elevator to success; you have to take the stairs one step at a time.",
            "Stormy seas make sturdy sailors; challenges help you unlock your inner strength.",
            "A river cuts through a rock not because of its power but its persistence; keep going.",
            "Progress is measured by how far you've come, not how far is left; celebrate small wins.",
            "You miss 100% of the shots you don't take; believe in yourself and take risks.",
            "Time flies like an arrow; fruit flies like a banana; seize the day and live fully.",
            "A smooth sea never made a skilled sailor; grow through life's ups and downs with grace.",
            "You can't control the wind but you can adjust the sails; adapt to circumstances.", 
            "When you change the way you look at things, the things you look at change; choose positivity.",
            "You have two choices in life: accept conditions as they exist, or accept responsibility for changing them.",
            "Be the architect of your life; envision your dream and work daily to build it.",
            "Skill comes after hours and days and weeks and years of practice; mastery arrives in time.",
            "Fire tests gold; hardship tests brave hearts; you will emerge wiser and stronger.",
            "Storms make oaks take deeper roots; hard times can strengthen character and resolve.", 
            "Enduring setbacks while maintaining the effort defines grit; persist through challenges.",
            "Progress lies not in enhancing what is, but advancing toward what will be.",
            "You can waste your life drawing lines or you can live crossing them; take action today.",
            "We awaken by asking the right questions; let your curiosity fuel your personal growth.",
            "You've been criticizing yourself for years and it hasn't worked; try encouraging yourself.",
            "Your outlook, not your income, determines your outcome; stay positive, better days are ahead.",
            "Don't let the noise drown out your inner voice; listen to your intuition.", 
            "It's better to walk alone, than with a crowd going in the wrong direction.",
            "There are no failures in life, only lessons; growth comes from progress, not perfection. ",
            "Every champion was once a contender that refused to give up.",
            "Be brave enough to start a conversation that matters.", 
            "Victory comes from finding opportunities in problems, not avoiding them.",
            "The phoenix must burn to rise from the ashes; you will overcome this challenge.",
            "Focus on the journey, not the destination; living in the moment brings joy.",
            "You cannot change what you refuse to confront.",
            "Worrying does not eliminate problems, it eliminates peace; breathe through challenges one day at a time.",
            "The temptation to quit will be greatest just before you are about to succeed; patience and persistence pay off.",
            "You cannot discover new oceans unless you have the courage to lose sight of the shore.", 
            "Don't be pushed around by the fears in your mind; be led by the dreams in your heart.",
            "Know that the darkest night comes just before the dawn; have hope, morning is on the way.",
            "Let adversity make you better, not bitter; grow through difficult times.",
            "Progress begins when you stop making excuses and start making changes.",
            "Do what you can with what you have from where you are; small steps add up over time.",
            "The secret to getting ahead is getting started; take action today, not tomorrow.",
            "Write the vision and make it plain so you can run with perseverance.",
            "Be less concerned about having a long life and more about having a meaningful life.",
            "We suffer more in our imagination than in reality; manage your thoughts skillfully.",
            "Be patient and tough; someday this pain will be useful to you.",
            "Storms make trees take deeper roots; challenging times can strengthen character.",
            "It's not who you are that holds you back, it's who you think you're not; believe in yourself.",
            "Character cannot be developed in ease and quiet; growth comes through experience and suffering.",
            "We delight in the beauty of the butterfly but rarely consider the changes it has gone through to achieve that beauty.",
            "Don't let your struggle become your identity; you are more than your hardships.",
            "Rainbows exist to prove that beautiful things can happen after the storm passes.",
            "Every problem has within it the seeds of its own solution; have faith you will prevail.",
            "Vision without action is a daydream; action without vision is a nightmare; live wisely.",
            "Diamonds form under pressure; believe you will emerge stronger from this challenge.",
            "In an arrow's flight you can only control the aim, not the winds; stay steady through turbulence.",
            "Courage doesn't mean you don't get afraid; courage means you don't let fear stop you.",
            "Tranquility comes when you stop reacting to every little thing life throws your way.",
            "The wound is where the light enters you; growth comes from healing.",
            "At the end of the storm there's a golden sky; there is light after darkness.",
            "Stay strong, make them wonder how you're still smiling.",
            "Tough times don't last but tough people do; believe in yourself.", 
            "Observe, adapt, and overcome; learn from setbacks to step forward wiser.",
            "When written in Chinese, the word crisis is composed of two characters -- one represents danger and the other represents opportunity.",
            "Stay the course; the darkest hour comes just before the dawn.",
            "What defines us is how well we rise after falling; never stop getting back up.",
            "Hardships often prepare people for an extraordinary destiny; embrace the journey.",
            "Do what scares you; great rewards await outside your comfort zone.",
            "Fall seven times, stand up eight; be relentless in achieving your dreams.", 
            "You will face many defeats in life but never let yourself be defeated.",
            "Though no one can go back and make a new beginning, anyone can start today and make a new ending.",
            "When you get to your wits end, you'll find God lives there; have faith.",
            "Turbulence is life telling you to buckle up; prepare yourself for the ride ahead.", 
            "If opportunity doesn't knock, build a door; stay proactive in creating your future.",
            "Diamonds take immense heat and pressure to form; you will overcome this challenge shining bright.",
            "It's not the load that breaks you down, it's the way you carry it; shift your perspective.",
            "The winds and waves are always on the side of the ablest navigators; chart your course wisely.",
            "You can't stop the waves but you can learn to swim; adapt and keep moving forward.",
            "Embrace patience in the present moment; great things take time to grow.",
            "Every problem is a gift without packaging; cultivate gratitude for life's challenges.",
            "The lotus flower blooms in the mud; find beauty in the difficult times.",
            "Rain cleanses, nourishes, and renews; believe that better days are ahead.",
            "From struggle blooms strength and wisdom; plant seeds of growth in dark times."
          ]
# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

speaker_list = Path("Speakers").glob("*.wav")
speaker_list = [str(s) for s in speaker_list]

# Run TTS
# ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
# Text to speech list of amplitude values as output
idx = np.random.randint(len(sentences))
s_idx = np.random.randint(len(speaker_list))
text = "Love looks not with the eyes, but with the mind, And therefore is winged Cupid painted blind" #sentences[idx]
print(text, speaker_list[s_idx])
wav = tts.tts(text=f"{text}", speaker_wav="Speakers/Yusuf_M.wav", language="en")
# Text to speech to a file
tts.tts_to_file(text=f"{text}", speaker_wav=f"{speaker_list[s_idx]}", language="en", file_path="output.wav")