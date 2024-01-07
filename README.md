# Learning to Generate Context-Sensitive Backchannel Smiles for Embodied AI Agents with Applications in Mental Health Dialogues
This repository contains data and code for the Backchannel Smile Generation paper from AAAI-24 Cognitive and Mental Health workshop.
[Paper](https://github.com/bmaneesh/Generating-Context-Sensitive-Backchannel-Smiles/)

## Backchannel dataset: The `data/data.zip` contains the dataset for backchannel smile generation.
Unzip `data/data.zip` to `data/` to run the code. `clean_data_8_context_length_60s_turns.pkl` contains the data required for the generative model in `src/LSTM_decoder.py`
It contains Backchannel (BC) smile (and non-smile) annotations together with the speaker and listener turns that correspond to it. Contents of the file are described below:

### Smile-related attributes:
```
id: video name from the RealTalk dataset
person: if listener is sitting to the left or right in the video
smile_idx: unique indicator of smile; not relevant to the analysis
start_timestamp: start time of the BC smile in milliseconds
intensity_max:  intensity of the BC smile (valid when IS_SMILE=TRUE)
duration: duration of the BC smile (valid when IS_SMILE=TRUE)
smile_landmarks: (context_length X landmark dimension) array containing listener's facial landmarks to be predicted from the audio-text feature of the speaker and listener
```

### Speaker attributes:
```
speaker_turn_end_timestamp: speaker's turn end timestamp
speaker_turn_duration: duration of speaker turn
speaker_turn_audio_filename: filename of the speaker's last turn. These are loaded dynamically
speaker_gender: annotated speaker gender (1=female)
speaker_pcm_RMSenergy: loudness of the speaker's turn
speaker_F0_sma: mean pitch of listener's turn
speaker_negate: proportion of negation words used in speaker's turn
speaker_turn_audio_embeddings: VGGish audio embeddings for speaker turn
speaker_turn_text: what speaker said in their last turn
```

### Listener attributes:
```
listener_turn_start_timestamp: listener's last turn start timestamp
listener_turn_duration: duration of the listener's last turn
listener_turn_audio_filename: filename of the listener's last turn. These are loaded dynamically
listener_compare: proportion of comparison words from listener's turn
listener_WC: number of words in the listener's turn
listener_F0_sma: mean pitch of listener's turn
IS_SMILE: TRUE if the listener-speaker turn exchange resulted in a BC smile.
listener_pcm_RMSenergy: loudness of the listener's turn
listener_preceeding_landmarks: preceeding frame facial landmarks of the listener
listener_turn_audio_embeddings: VGGish audio embeddings for listener turn
listener_turn_text: what listener said in their last turn
```

The 49 2-dimensional facial landmarks  are derived from [AFARtoolbox](https://github.com/AffectAnalysisGroup/AFARtoolbox). All durations and timestamps are in seconds but `start_timestamp` is in milliseconds. Language attributes were derived from the Linguistic-Inquiry Word Count ([LIWC](https://www.liwc.app/) 2015) framework and prosody attributes were derived from [OpenSMILE (v3)](https://audeering.github.io/opensmile/).

Raw turn-level audio for the dataset can be downloaded from [here]() and placed in `turn_level_audio` folder. This is step is optional.

To run the generative model, use `python src/LSTM_decoder.py`.

Visualizations and some code are built upon existing an [work](https://github.com/BenSaunders27/ProgressiveTransformersSLP) on Sign-Language Generation.

Citation:
```

```
