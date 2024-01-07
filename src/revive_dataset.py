"""
On Friday, 11/10/2023, the realtalk data and annotations on server was found to be corrupt.
This file is using the pkl/csv files that contain that were created as an intermediate step to recreate the audio files for model training.
Wish me luck!
"""

import sys
sys.path.append('./')
import os
import glob
import os.path
from typing import Optional
import io

import torch

import pandas as pd
import numpy as np
from numpy import random
import pickle
# from pydub import AudioSegment, effects
from sklearn.model_selection import StratifiedShuffleSplit, GroupKFold#, StratifiedGroupKFold
# from ffmpeg import extract_audio
# from generic_tools import map_parallel
from tqdm import tqdm
# from facial_expression_generation.data.data_generation import create_landmarks_dataset_for_generation
# from transcriptions.forced_alignment import read_text_grid


import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

RANDOM_SEED = 123
np.random.seed(RANDOM_SEED) # https://stackoverflow.com/questions/70149388/why-seed-method-isnt-part-of-randint-function for randint function
TARGET_AUDIO_SAMPLING_RATE = 16000


def initialize_data_file():
    SMILE_DATA_FILE = './liwc_opensmile_results_realtalk_manual_smiles_dataset_nan_version.csv'
    NEGATIVE_DATA_FILE = './liwc_opensmile_results_realtalk_manual_smiles_negative_dataset_nan_version.csv'
    SMILE_DATA = pd.read_csv(SMILE_DATA_FILE)
    # use smiles that are non-zero intensity from AFAR
    SMILE_DATA = SMILE_DATA.loc[SMILE_DATA['intensity_max'] > 0, ]
    # use smiles that have turn info for both speaker and listener; unavailable turns have np.nan in their turn entries
    unavailable_rows = SMILE_DATA.apply(lambda x: isinstance(x['speaker_turn_text'], float) or isinstance(x['listener_turn_text'], float), axis=1)
    # SMILE_DATA.loc[unavailable_rows][['video', 'person', 'start_timestamp', 'end_timestamp', 'speaker_turn_text', 'listener_turn_text']]
    SMILE_DATA = SMILE_DATA.loc[~unavailable_rows]
    SMILE_DATA['IS_SMILE'] = True
    NON_SMILE_DATA = pd.read_csv(NEGATIVE_DATA_FILE)
    NON_SMILE_DATA['IS_SMILE'] = False
    delete_indices = []
    NON_SMILE_DATA = pd.read_csv(NEGATIVE_DATA_FILE)
    NON_SMILE_DATA['IS_SMILE'] = False
    NON_SMILE_DATA['id'] = NON_SMILE_DATA['video']
    return SMILE_DATA, NON_SMILE_DATA



class DatasetCreator:
    def __init__(self, context_length=8) -> None:
        self.initialize_file_paths()
        self.turn_df = pd.read_pickle(self.turn_file)
        self.conditioning_features = ['speaker_gender', 'speaker_negate', 'listener_compare', 'listener_WC', 'speaker_pcm_RMSenergy', 'listener_F0_sma']
        self.context_length = context_length
        self.frame_downsample_rate = 3
        self.validate_folder(self.AUDIO_DIR)
        self.validate_folder(self.AUDIO_FOR_OPENSMILE_DIR)
        self.extract_realtalk_audio()
        SMILE_DATA, NON_SMILE_DATA  = initialize_data_file()
        video_smile_dist = self.video_distribution_of_smiles(SMILE_DATA)
        balanced_non_smile_data = self.balance_negative_instance(NON_SMILE_DATA, video_smile_dist)
        self.mean_smile_duration = SMILE_DATA['duration'].mean()
        self.data = pd.concat((SMILE_DATA, balanced_non_smile_data), axis=0)
        self.data.drop(columns=['right-turns', 'left-turns'], inplace=True)
        self.assign_duration_to_non_smiles() # mean smile duration is assigned to all non-smiles
        smiles_with_turn_timestamps = self.assign_turn_timestamps()
        self.data = smiles_with_turn_timestamps
        # self.prepare_for_forced_alignment()
        # self.add_smile_landmarks(smiles_with_turn_timestamps)
        self.manipulate_data_for_generation(smiles_with_turn_timestamps, frame_downsample_rate=self.frame_downsample_rate)
        self.forced_aligned_turn_text_extraction() # output_dir=self.forced_alignment_output
        # self.extract_opensmile_features()
        self.infer_conditioning_features()
        # self.organize_audio_for_opensmile(list(video_smile_dist.keys()))
        with open(os.path.join(self.dataset_folder, f"realtalk_dataset_context_{self.context_length}_downsample_{self.frame_downsample_rate}_speaker_turn_60s_speaker_gender_smile_idx_added.pkl"), 'wb') as f:
            pickle.dump(self.frame_wise_data, f)

    def validate_folder(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        return
    

    def manipulate_data_for_generation(self, dataset_df, frame_downsample_rate):

        print('Manipulating data for generation--landmarks are broken in context window lengths')
        # dataset_with_landmarks_df = create_landmarks_dataset_for_generation(output_dir=None, dataset = dataset_df)
        # with open('./data/landmark_generation_data_reformatted_with_negatives.pkl', 'wb') as f:
        #     pickle.dump(dataset_with_landmarks_df, f)
        with open('./data/landmark_generation_data_reformatted_with_negatives.pkl', 'rb') as f:
            dataset_with_landmarks_df = pickle.load(f)
        
        if frame_downsample_rate > 1:
            for sid, smile in enumerate(dataset_with_landmarks_df):
                for person in ['listener']:
                    number_of_frames = len(dataset_with_landmarks_df[sid]['smile_landmarks'][person])
                    downsampled_indices = list(range(0, number_of_frames, frame_downsample_rate))
                    dataset_with_landmarks_df[sid]['smile_landmarks'][person] = [landmarks for frame_id, landmarks in enumerate(smile['smile_landmarks'][person]) if frame_id in downsampled_indices]

        # negative samples have no gender assignments, so infer from positive samples
        # this is not the "person" gender
        speaker_gender_mapping = dict()
        for i in range(len(dataset_with_landmarks_df)):
            if isinstance(dataset_with_landmarks_df[i]['speaker_gender'], str):
                speaker_gender_mapping[dataset_with_landmarks_df[i]['id']+'_'+dataset_with_landmarks_df[i]['person']] = dataset_with_landmarks_df[i]['speaker_gender']
        
        for i in range(len(dataset_with_landmarks_df)):
            person = dataset_with_landmarks_df[i]['id']+'_'+dataset_with_landmarks_df[i]['person']
            dataset_with_landmarks_df[i]['speaker_gender'] = [1 if speaker_gender_mapping[person]=='F' else 0][0]

        self.frame_wise_data, number_of_frames_for_prediction = self.frame_wise_prediction_data(dataset_with_landmarks_df, context_length=self.context_length, downsampling_rate=frame_downsample_rate)
        # import matplotlib.pyplot as plt
        # speaker_turn_durations = [i['speaker_turn_duration'] for i in self.frame_wise_data]
        # plt.hist(speaker_turn_durations, bins=149)
        # plt.savefig('/mnt/Volume1/realtalk/revived_data/speaker_turn_duration_context_8_downsample_3.png')
        # plt.close('all')
        # print(f'{number_of_frames_for_prediction} prediction examples available')
        return
    
    def frame_wise_prediction_data(self, dataset_df, context_length, downsampling_rate):
        frame_wise_data = list()
        number_of_frames_for_prediction = 0
        print('Curating frame level data for generation')
        smile_idx = 0
        for smile in tqdm(dataset_df):
            for frame_idx in range(context_length, len(smile['smile_landmarks']['listener'])):
                frame_wise_data.append(dict())
                frame_wise_data[-1]['start_timestamp'] = smile['start_timestamp']
                # smile start timestamp is not the same as the speaker turn end timestamp--as smiles longer than context are broken into multiple smiles
                # downsampling_rate is to account for downsampling for model training; fps is 25; 1e3 is to convert to ms
                frame_wise_data[-1]['speaker_turn_end_timestamp'] = (smile['start_timestamp'] + ((frame_idx-1-context_length) * downsampling_rate / 25) * 1e3) / 1e3
                frame_wise_data[-1]['speaker_turn_start_timestamp'] = smile['speaker_turn_start']
                frame_wise_data[-1]['speaker_turn_duration'] = frame_wise_data[-1]['speaker_turn_end_timestamp'] - frame_wise_data[-1]['speaker_turn_start_timestamp']
                # listener turns are not affected by smile start timestamp as a speaker always holds the floor when smiling
                frame_wise_data[-1]['listener_turn_start_timestamp'] = smile['listener_turn_start']
                # frame_wise_data[-1]['listener_turn_end_timestamp'] = smile['listener_turn_stop']
                frame_wise_data[-1]['listener_turn_duration'] = smile['listener_turn_stop'] - smile['listener_turn_start']
                frame_wise_data[-1]['listener_turn_audio_filename'] = self.find_turn_based_on_start_time(smile['video'], smile['person'], smile['listener_turn_start'], is_speaker=False)
                frame_wise_data[-1]['speaker_turn_audio_filename'] = self.find_turn_based_on_start_time(smile['video'], smile['person'], smile['speaker_turn_start'], is_speaker=True)
                frame_wise_data[-1]['person'] = smile['person']
                frame_wise_data[-1]['id'] = smile['id'] + '_frame_idx_' + str(frame_idx)
                frame_wise_data[-1]['smile_idx'] = smile_idx
                # extract running audio features here
                for i in self.conditioning_features:
                    frame_wise_data[-1][i] = smile[i]
                frame_wise_data[-1]['IS_SMILE'] = smile['IS_SMILE']
                frame_wise_data[-1]['intensity_max'] = smile['intensity_max']
                frame_wise_data[-1]['duration'] = smile['duration'] / 1e3 # in ms
                frame_wise_data[-1]['smile_landmarks'] = np.array(smile['smile_landmarks']['listener'][frame_idx-context_length: frame_idx]).reshape((context_length, -1))
                frame_wise_data[-1]['listener_preceeding_landmarks'] = smile['smile_landmarks']['listener_preceeding_frame'][0].reshape((1, -1))
                # listener embeddings correspond to the whole listener turn
                # speaker embeddings correspond to the part of the speaker turn just preceeding the landmarks to predict
                if ('speaker_turn_audio_embeddings' not in smile) and ('listener_turn_audio_embeddings' not in smile):
                    frame_wise_data[-1]['speaker_turn_audio_embeddings'] = self.extract_vggish_embeddings(frame_wise_data[-1]['speaker_turn_audio_filename'], turn_duration=frame_wise_data[-1]['speaker_turn_duration'])
                    frame_wise_data[-1]['listener_turn_audio_embeddings'] = self.extract_vggish_embeddings(frame_wise_data[-1]['listener_turn_audio_filename'], turn_duration=frame_wise_data[-1]['listener_turn_duration'])
                # find_speaker_listener_turns()
                number_of_frames_for_prediction += 1
            smile_idx += 1
        
        return frame_wise_data, number_of_frames_for_prediction
    
    def find_turn_based_on_start_time(self, video, listener_left_or_right, turn_start, is_speaker=False):
        if is_speaker:
            if listener_left_or_right == 'left':
                person = 'right'
            else:
                person = 'left'
        else:
            person = listener_left_or_right
        # person_turn_audio_files = [i for i in os.listdir(os.path.join(self.forced_aligned_folder, video+'_'+person)) if (i.startswith(video+'_'+person+'_'+str(turn_start)) and i.endswith('.wav'))]
        # change the self.forced_aligned_input to self.forced_aligned_output
        person_turn_audio_files = [i for i in os.listdir(os.path.join(self.forced_aligned_input, video+'_'+person)) if (i.startswith(video+'_'+person+'_'+str(turn_start)) and i.endswith('.wav'))]
        assert len(person_turn_audio_files) == 1, print('More than one turn or no turns were found for {0} {1} {2}'.format(video, person, turn_start))
        # return os.path.join(self.forced_aligned_folder, video+'_'+person, person_turn_audio_files[0])    
        return os.path.join(self.forced_aligned_input, video+'_'+person, person_turn_audio_files[0])    
    
    def extract_vggish_embeddings(self, audio_file, turn_duration):
        if vars(self).get('vggish', None) is None:
            self.vggish = torch.hub.load('harritaylor/torchvggish', 'vggish')
            self.vggish.eval()
        temp_file = os.path.join(self.cache_folder, 'cache_audio_embeddings.wav')
        
        import torchaudio.transforms as T
        resampling_rate = 16000
        original_sampling_rate = 48000
        resampler = T.Resample(original_sampling_rate, resampling_rate)
        MAX_AUDIO_DURATION = 60 # seconds
        TARGET_LENGTH = MAX_AUDIO_DURATION * resampling_rate
        _waveform, sampling_rate = torchaudio.load(audio_file)
        _waveform = _waveform[0, :int(turn_duration*original_sampling_rate)]
        resampled_waveform = resampler(_waveform)
        # take the last segment of audio of MAX_AUDIO_DURATION seconds
        if len(resampled_waveform) > TARGET_LENGTH:
            trimmed_resampled_waveform = resampled_waveform[-1*int(TARGET_LENGTH):]
        else:
            pad_seq = torch.zeros((int(TARGET_LENGTH - len(resampled_waveform))))
            trimmed_resampled_waveform = torch.cat((pad_seq, resampled_waveform), dim=0)
        trimmed_resampled_waveform = trimmed_resampled_waveform.unsqueeze(0)
        torchaudio.save(temp_file, trimmed_resampled_waveform, resampling_rate)
        try:
            embeddings = self.vggish.forward(temp_file)
        except:
            # some turns are too short for vggish to process
            embeddings = torch.zeros((1, 128))
        embeddings = embeddings.detach().to(device='cpu').numpy()
        return embeddings

    def prepare_for_forced_alignment(self):
        self.validate_folder(self.forced_aligned_input)
        self.validate_folder(self.forced_aligned_folder)

        # create separate audio file for left and right individual using the turn information
        for person in ['left', 'right']:
            for audio_file in glob.glob(os.path.join(self.AUDIO_FOR_OPENSMILE_DIR, '*.wav')):
                folder = os.path.basename(audio_file).replace('.wav', '')+'_'+person
                _waveform, sampling_rate = torchaudio.load(audio_file)
                _waveform = _waveform[0, :]
                person_turns = self.turn_df[self.turn_df['id'] == os.path.basename(audio_file).replace('.wav', '')][person+'-turns'][0]
                for _, turn in person_turns.iterrows():
                    is_smile_turn = self.check_turn_related_to_smile(turn, person, audio_file) # the turn could be a speaker or listener turn
                    if is_smile_turn:
                        self.validate_folder(os.path.join(self.forced_aligned_input, folder))
                        person_spoken_text_to_write = person_turns.at[_, person+'-turns']
                        if len(person_spoken_text_to_write) > 0:
                            filename = os.path.basename(audio_file).replace('.wav', '')+'_'+person+'_'+str(turn['start_time'])+'_'+str(turn['stop_time'])
                            with open(os.path.join(self.forced_aligned_input, folder, filename+'.txt'), 'w') as f:
                                f.write(person_spoken_text_to_write)
                            # preserve only the person audio
                            preserving_vector = np.zeros(len(_waveform))
                            # for _, turn in person_turns.iterrows():
                            start = int(turn['start_time'] * sampling_rate)
                            stop = int(turn['stop_time'] * sampling_rate)
                            preserving_vector[start:stop] = 1
                            person_waveform = _waveform * preserving_vector
                            person_waveform = person_waveform[start:stop]
                            assert person_waveform.shape[0] >0, print('empty waveform as input to forced aligner')
                            torchaudio.save(os.path.join(self.forced_aligned_input, folder, '{0}.wav'.format(filename)), person_waveform, sampling_rate)
                        else:
                            print('Empty turn as input to forced aligner??')
        return
    

    def check_turn_related_to_smile(self, turn, person, audio_file):
        videoname = os.path.basename(audio_file).replace('.wav', '')
        # we ignore person as using person condition would only have turns where the person is listening
        smiles_in_video = self.data[(self.data['id'] == videoname)] #  & (self.data['person'] == person)

        # if the given turn is associated with a smile, return True
        if len(smiles_in_video) > 0:
            for _, smile in smiles_in_video.iterrows():
                # even if the turn belongs to one of the smile, we want to use it for forced alignment
                if (smile['listener_turn_start'] == turn['start_time']) or (smile['speaker_turn_start']==turn['start_time']):
                    return True
        return False
    

    def forced_aligned_turn_text_extraction(self):
        # audio signal boundaries extraction for turns is already done in self.frame_wise_prediction_data(). Text features within turn boundary are extracted here
        # use the MFA output to create the CSV for LIWC.

        invalid_smiles = list()
        words_list = list()
        for i in range(len(self.frame_wise_data)):
            video = self.frame_wise_data[i]['id'].split('_')[0]
            listener = self.frame_wise_data[i]['person']
            try:
                for person in ['speaker', 'listener']:
                    turn_duration = self.frame_wise_data[i][f'{person}_turn_duration']
                    turn_text_filename = self.frame_wise_data[i][f'{person}_turn_audio_filename'].replace('forced_alignment_input', 'forced_alignment_output').replace('.wav', '.TextGrid')
                    # turn_start is with the timestamp wrt the beginning of the video
                    turn_start, turn_stop = os.path.basename(turn_text_filename).replace('.TextGrid', '').split('_')[-2:]
                    turn_start, turn_stop = float(turn_start), float(turn_stop)
                    # forced aligned gives timestamps wrt the beginning of the turn
                    forced_aligned_content = read_text_grid(turn_text_filename)
                    # we only need the duration of the turn to retrieve elapsed text from the beginning of the turn
                    speaker_turn_idx_before_smile_onset = pd.Index(forced_aligned_content['words']['start(msec)']).get_loc(turn_duration*1e3, 'ffill')
                    self.frame_wise_data[i][f'{person}_turn_text'] = ' '.join(forced_aligned_content['words'].iloc[:speaker_turn_idx_before_smile_onset]['words'].tolist())
                    if len(self.frame_wise_data[i][f'{person}_turn_text']) == 0:
                        print('Empty turn text for {0} {1} {2}'.format(video, person, turn_start))
                    # use this for LIWC CSV
                    words_list.extend(forced_aligned_content['words']['words'].tolist())
            except:
                invalid_smiles.append(i)
        # remove invalid smiles
        self.frame_wise_data = [self.frame_wise_data[i] for i in range(len(self.frame_wise_data)) if i not in invalid_smiles]
        print(f"{len(invalid_smiles)} have no turn text after forced alignment")
        words_list = list(set(words_list)) # 2231 unique words used in the dataset
        words = {i: words_list[i] for i in range(len(words_list))}
        # pd.DataFrame.from_dict(words, orient='index').to_csv(os.path.join(self.liwc_input_path, 'LIWC_word_input.csv'))
        return

    def assign_turn_timestamps(self, last_nturns=1):
        print('Assigning turn timestamps')
        num_smiles = len(self.data)
        dataset_df = pd.merge(self.data, self.turn_df, left_on=['id'], right_on=['id'], suffixes=('', ''), how='left')
        dataset_df['speaker_turn_text'] = np.nan
        dataset_df['speaker_turn_start'] = np.nan
        dataset_df['speaker_turn_stop'] = np.nan
        dataset_df['listener_turn_text'] = np.nan
        dataset_df['listener_turn_start'] = np.nan
        dataset_df['listener_turn_stop'] = np.nan

        for i in range(num_smiles):
            speaker, listener = self.infer_speaker_listener(dataset_df, row_idx=i)
            smile_start_time = dataset_df['start_timestamp'].iloc[i]/1e3

            speaker_turn_idx_before_smile_onset = pd.Index(dataset_df[speaker + '-turns'].iloc[i]['start_time']).get_loc(smile_start_time, 'ffill')
            # how many of speaker turns before smile onset to fetch
            speaker_turn_before_smile_onset = dataset_df[speaker + '-turns'].iloc[i].iloc[speaker_turn_idx_before_smile_onset-last_nturns+1:speaker_turn_idx_before_smile_onset+1]
            speaker_turn_text_before_smile_onset = speaker_turn_before_smile_onset[speaker + '-turns'].tolist()
            speaker_turn_text_before_smile_onset_start = speaker_turn_before_smile_onset['start_time'].min()
            speaker_turn_text_before_smile_onset_stop = speaker_turn_before_smile_onset['stop_time'].max()


            listener_turn_idx_before_smile_onset = pd.Index(dataset_df[listener + '-turns'].iloc[i]['stop_time']).get_loc(smile_start_time, 'ffill')
            # how many of listener turns before smile onset to fetch
            listener_turn_before_smile_onset = dataset_df[listener + '-turns'].iloc[i].iloc[max(listener_turn_idx_before_smile_onset-last_nturns+1, 0):listener_turn_idx_before_smile_onset+1]
            listener_turn_text_before_smile_onset = listener_turn_before_smile_onset[listener + '-turns'].tolist()
            listener_turn_text_before_smile_onset_start = listener_turn_before_smile_onset['start_time'].min()
            listener_turn_text_before_smile_onset_stop = listener_turn_before_smile_onset['stop_time'].max()

            dataset_df['speaker_turn_text'].iloc[i] = speaker_turn_text_before_smile_onset
            # turn timestamps (start, stop) would go beyond the smile onset (maybe offset also) here
            dataset_df['speaker_turn_start'].iloc[i] = speaker_turn_text_before_smile_onset_start
            dataset_df['speaker_turn_stop'].iloc[i] = speaker_turn_text_before_smile_onset_stop
            dataset_df['listener_turn_text'].iloc[i] = listener_turn_text_before_smile_onset
            # turn timestamps (start, stop) would go beyond the smile onset (maybe offset also) here
            dataset_df['listener_turn_start'].iloc[i] = listener_turn_text_before_smile_onset_start
            dataset_df['listener_turn_stop'].iloc[i] = listener_turn_text_before_smile_onset_stop

        # find turns where we could not find listener or speaker turn corresponding to the smile
        unavailable_smile_turns = (np.isnan(dataset_df[['speaker_turn_start', 'listener_turn_start']]).sum(axis=1) > 0).sum()
        print('{0} smile do not have either speaker or listener or both turn unavailable in realtalk'.format(unavailable_smile_turns))

        return dataset_df
    

    # def extract_opensmile_features(self):
    #     print('Extracting opensmile features for speaker/listener turns based on the turn timestamps')
    #     for i in tqdm(range(len(self.frame_wise_data))):
    #         opensmile_features = pd.read_csv(os.path.join(self.opensmile_features, self.frame_wise_data['id'].iloc[i]+'.csv'))

    #     return
    
    def infer_conditioning_features(self):
        # read main file to gender speaker gender
        liwc_output_df = pd.read_csv(os.path.join(self.liwc_input_path, 'LIWC2015 Results (LIWC_word_input).csv'))
        liwc_output_df.set_index('word', inplace=True)

        # add speaker gender to each row as well
        # liwc_output.rename(columns={'gender': 'listener_gender'}, inplace=True)
        # assert len(liwc_output) == len(dataset_df), print('length of liwc output and dataset_df should be the same to infer speaker gender')
        # liwc_output['speaker_gender'] = dataset_df['speaker_gender']


        # convert opensmile outputs that are specific to speaker and listener into one file to use in R
        # we use these features because of how they correlate with different speaker traits (Memon et al. 2020)

        # self.frame_wise_data = self.frame_wise_data[:30]
        # map_parallel(self.sub_function, list(range(len(self.frame_wise_data))), num_workers=4)

        for i in range(len(self.frame_wise_data)):
            for person in ['speaker', 'listener']:
                opensmile_path = os.path.join(self.opensmile_features, f"{self.frame_wise_data[i]['id'].split('_frame')[0]}.csv")
                opensmile_output = pd.read_csv(opensmile_path)
                opensmile_output.set_index('time', inplace=True)
                start = self.frame_wise_data[i][f'{person}_turn_start_timestamp']
                end = start+self.frame_wise_data[i][f'{person}_turn_duration']
                opensmile_output_trimmed = opensmile_output.loc[start:end].reset_index()
                opensmile_acoustic_correlates = self.compute_acoustic_correlates_of_behavior(opensmile_output_trimmed)# .replace({np.nan: 0})
                for feature in self.conditioning_features:
                    feature = feature.replace('speaker_', '').replace('listener_', '')
                    # if feature.replace('speaker_', '').replace('listener_', '') in opensmile_acoustic_correlates.index:
                    if feature in opensmile_acoustic_correlates.index:
                        self.frame_wise_data[i][person+'_'+feature] = opensmile_acoustic_correlates[feature]

                if len(self.frame_wise_data[i][f"{person}_turn_text"]) > 0:
                    liwc_features = self.compute_liwc_features(self.frame_wise_data[i][f"{person}_turn_text"], liwc_output_df)
                    for feature in liwc_features.keys():
                        if feature.replace('speaker_', '').replace('listener_', '') in self.conditioning_features:
                            self.frame_wise_data[i][person+'_'+feature] = liwc_features[feature]
        return
    

    def sub_function(self, i):
        for person in ['speaker', 'listener']:
            opensmile_path = os.path.join(self.opensmile_features, f"{self.frame_wise_data[i]['id'].split('_frame')[0]}.csv")
            opensmile_output = pd.read_csv(opensmile_path)
            opensmile_output.set_index('time', inplace=True)
            start = self.frame_wise_data[i][f'{person}_turn_start_timestamp']
            end = start+self.frame_wise_data[i][f'{person}_turn_duration']
            opensmile_output_trimmed = opensmile_output.loc[start:end].reset_index()
            opensmile_acoustic_correlates = self.compute_acoustic_correlates_of_behavior(opensmile_output_trimmed)# .replace({np.nan: 0})
            for feature in self.conditioning_features:
                feature = feature.replace('speaker_', '').replace('listener_', '')
                # if feature.replace('speaker_', '').replace('listener_', '') in opensmile_acoustic_correlates.index:
                if feature in opensmile_acoustic_correlates.index:
                    self.frame_wise_data[i][person+'_'+feature] = opensmile_acoustic_correlates[feature]

            if len(self.frame_wise_data[i][f"{person}_turn_text"]) > 0:
                liwc_features = self.compute_liwc_features(self.frame_wise_data[i][f"{person}_turn_text"], liwc_output_df)
                for feature in liwc_features.keys():
                    if feature.replace('speaker_', '').replace('listener_', '') in self.conditioning_features:
                        self.frame_wise_data[i][person+'_'+feature] = liwc_features[feature]
        return
    
    def compute_liwc_features(self, turn_text, liwc_df):
        out_features = dict()
        words = turn_text.split(' ')
        # to handle excel reading true as boolean instead of string
        if 'true' in words:
            words.remove('true')
            words.append('true--')
        out_features['WC'] = liwc_df.loc[words]['WC'].sum()
        for feat in set(liwc_df.columns.tolist()).difference(set(['WC'])):
            out_features[feat] = liwc_df.loc[words][feat].mean()
        return out_features

    def organize_audio_for_opensmile(self, videos):
        import shutil
        for video in videos:
            video = video.split('__')[0]
            # copy audio files to a new folder
            audio_file = os.path.join(self.AUDIO_DIR, os.path.basename(video)+'.wav')
            if not os.path.exists(audio_file):
                print(f"Audio file {audio_file} not found")
                continue
            # os.system(f'cp {audio_file} {self.AUDIO_FOR_OPENSMILE_DIR}')
            shutil.copy(audio_file, self.AUDIO_FOR_OPENSMILE_DIR)
        return
    
    @staticmethod
    def video_distribution_of_smiles(smiles_df):
        dist = dict()
        for smile in smiles_df.iterrows():
            name = smile[1]['video'] + '__'+ smile[1]['person']
            if name not in dist:
                dist[name] = 0
            dist[name] += 1
        return dist

    def extract_realtalk_audio(self):
        print('Extracting audio from video files')
        self.number_of_videos = len(glob.glob(os.path.join(self.VIDEO_DIR, '*.avi')))
        self.number_of_audio_files = len(glob.glob(os.path.join(self.AUDIO_DIR, '*.wav')))
        if self.number_of_videos == self.number_of_audio_files:
            return
        for file in tqdm(glob.glob(os.path.join(self.VIDEO_DIR, '*.avi'))):
            extract_audio(in_file=file, out_file=os.path.join(self.AUDIO_DIR, os.path.basename(file).replace('.avi', '.wav')))
        return
    
    def assign_duration_to_non_smiles(self):
        # set non-smile duration to mean smile duration
        self.data.loc[self.data['end_timestamp']==-1, 'duration'] = int(self.mean_smile_duration)
        self.data.loc[self.data['end_timestamp']==-1, 'end_timestamp'] = self.data.loc[self.data['end_timestamp']==-1, 'start_timestamp'] + int(self.mean_smile_duration)
        return    

    def balance_negative_instance(self, data, smile_dist):
        print('here')
        balanced_instances = []
        total_smiles = sum(smile_dist.values())
        for name in smile_dist:
            video, person = name.split('__')
            if smile_dist[name] > 0:
                negative_examples = data[(data['video'] == video) & (data['person'] == person)]
                num_negative_examples = len(negative_examples)
                # negative_sample_indices = np.random.randint(0, high=num_negative_examples, size=smile_dist[name])
                negative_sample_indices = np.arange(num_negative_examples)
                np.random.shuffle(negative_sample_indices)
                negative_sample_indices = negative_sample_indices[:smile_dist[name]].tolist()
                assert len(negative_sample_indices) == len(set(negative_sample_indices)), print("negative samples are redundant")
                # combine smile annotations and non-smile samples
                balanced_instances.append(negative_examples.iloc[negative_sample_indices])
        balanced_instances = pd.concat(balanced_instances, axis=0)
        assert len(balanced_instances) == total_smiles, print("Balanced instances don't match total smiles")
        return balanced_instances

    @staticmethod
    def infer_speaker_listener(dataset_df, row_idx):
        if dataset_df['person'].iloc[row_idx] == 'left':
            speaker = 'right'
            listener = 'left'
        else:
            speaker = 'left'
            listener = 'right'
        return speaker, listener
    
    @staticmethod
    def compute_acoustic_correlates_of_behavior(opensmile_output):
        """
        # we use the acoustic correlates of confidence, enthusiasm, doubtfulness, warmth and maturity (Memon. 2020)
        # confidence-pcm_RMSenergy, pcm_zcr, F0
        # enthusiasm-pcm_RMSenergy, pcm_zcr, F0
        # doubtfulness-pcm_RMSenergy, pcm_zcr, F0
        # warmth (male) -F0_sma_range, pcm_fftMag_spectrulSlope, slopeV0500_sma3nz_amean, slopeVU0500_sma3nz_amean, F1frequency_sma3nz_stddevNorm, F2frequency_sma3nz_stddevNorm
        # warmth (female) -F0_sma_range, F1frequency_sma3nz_amean, F2bandwidth_sma3nz_stddevNorm, pcm_fftMag_spectrulFlux, F1frequency_sma3nz_stddevNorm, F0semitoneFrom27.5Hz
        # maturity (female)-F0semitoneFrom27.5Hz, mfcc4_sma3_mean, F1frequency_sma3nz_amean, mfcc2_sma3_mean
        # maturity (male)-F0semitoneFrom27.5Hz, F3frequency_sma3nz_stddevNorm, F3bandwidth_sma3nz_stddevNorm
        # Likeability (male)-F0semitoneFrom27.5Hz, audSpec_Rfilt_sma_centroid
        # Likeability (female)-spectralFlux_sma3_amean
        # Dominance (male) - F0semitoneFrom27.5Hz

        # check if these match, remaining are already given by opensmile
        # pcm_fftMag_spectralFlux='spectralFlux_sma3_stddevNorm' (from the description in Memon 2020)
        # F0semitoneFrom27.5Hz='F0semitoneFrom27.5Hz_sma3nz_percentile50.0'
        # slopeV0500_sma3nz_amean='slopeV0-500_sma3nz_amean'
        # F2frequency_sma3nz_stddevF0_sma_range='F2frequency_sma3nz_stddevNorm'
        # F0_sma_range='F0_sma_range' to calculate
        # pcm_fftMag_spectralSlope (cant find--already covered as slopeV0-500_sma3nz_amean and slopeUV0-500_sma3nz_amean)???
        # pcm_RMSenergy to calculate
        # pcm_zcr to calculate, dropped

        """

        existing_opensmile_columns_of_correlates = ['F0_sma', 'F3frequency_sma3nz_stddevNorm', 'F1frequency_sma3nz_stddevNorm',
        'F3bandwidth_sma3nz_stddevNorm', 'F1frequency_sma3nz_amean', 'F2bandwidth_sma3nz_stddevNorm',
        'spectralFlux_sma3_amean', 'spectralFlux_sma3_stddevNorm', 'F0semitoneFrom27.5Hz_sma3nz_percentile50.0',
        'slopeV0-500_sma3nz_amean', 'slopeUV0-500_sma3nz_amean', 'F2frequency_sma3nz_stddevNorm', 'mfcc4_sma3_amean', 'mfcc2_sma3_amean']

        opensmile_mean_correlates = opensmile_output[existing_opensmile_columns_of_correlates].mean()
        # features_to_calculate like ['pcm_RMSenergy', 'F0_sma_range']
        opensmile_mean_correlates['F0_sma_range'] = opensmile_output['F0_sma'].max() - opensmile_output['F0_sma'].min()
        # calculate pcm_RMSenergy from pcm_loudness_sma
        opensmile_output['pcm_RMSenergy'] = pow(10, (opensmile_output['pcm_loudness_sma'] / 10)) # convert loudness from db to linear scale
        try:
            opensmile_mean_correlates['pcm_RMSenergy'] = pow(opensmile_output['pcm_RMSenergy'].pow(2).sum() / len(opensmile_output), 1/2) # calculate RMS energy
        except ZeroDivisionError:
            opensmile_mean_correlates['pcm_RMSenergy'] = np.nan
        # turn duration is different from word-count; this considers intra-individual pauses as well
        opensmile_mean_correlates['turn_duration'] = opensmile_output['time'].max() - opensmile_output['time'].min()

        return opensmile_mean_correlates
    

    def initialize_file_paths(self):
        self.VIDEO_DIR = './realtalk/videos'
        self.AUDIO_DIR = './realtalk/revived_data/audio'
        self.AUDIO_FOR_OPENSMILE_DIR = './realtalk/revived_data/audio_for_opensmile'
        self.opensmile_features = './realtalk/opensmile_features_for_audio'
        self.turn_file = './realtalk/realtalk_turns_mode_speaker.pkl'
        self.speaker_liwc_file = './realtalk/revived_data/speaker_LIWC2015 Results (realtalk_manual_smiles_negative_dataset).csv'
        self.listener_liwc_file = './realtalk/revived_data/listener_LIWC2015 Results (realtalk_manual_smiles_negative_dataset).csv'
        self.landmarks_file = './data/landmark_generation_data_reformatted_with_negatives.pkl'
        self.forced_aligned_folder = './realtalk/revived_data/forced_alignment_output'
        self.forced_aligned_input = './realtalk/revived_data/forced_alignment_input'
        self.cache_folder = './realtalk/cache'
        self.liwc_input_path = './realtalk/revived_data/'
        self.dataset_folder = './realtalk/revived_data/'
        self.validate_folder(self.cache_folder)
        return



class TurnDataset(Dataset):
    def __init__(self, fold, context_length, downsample_rate, partition='test', use_conditioning_vectors=True) -> None:
        
        with open(f'./data/clean_data_{context_length}_context_length_60s_turns.pkl', 'rb') as f:
            self.data = pickle.load(f)
        if use_conditioning_vectors:
            self.conditioning_features = ['speaker_gender', 'speaker_negate', 'listener_compare', 'listener_WC', 'speaker_pcm_RMSenergy', 'listener_F0_sma']
        else:
            self.conditioning_features = []
        self.normalize_landmarks = True

        if use_conditioning_vectors:
            self.mean, self.std = self.normalize_conditioning_features()
        self.folds = split_annotations(self.data, landmark_generation=self.data)
        if partition == 'training':
            self.idx = self.folds[fold]['train']
        elif partition == 'validation':
            self.idx = self.folds[fold]['valid']
        else:
            self.idx = self.folds[fold]['test']
        self.data = [i for i in self.data if i['id'].split('_frame')[0] in self.idx]
        self.persons_in_dataset = set([i['id'].split('_frame')[0]+'_'+i['person'] for i in self.data])
                
        # normalize landmarks
        if self.normalize_landmarks:
            self.znormalize_landmarks_list()
    
    def normalize_conditioning_features(self):
        data = {feature: [] for feature in self.conditioning_features}
        for feature in self.conditioning_features:
            for i in self.data:
                data[feature].append(i[feature])
        data = pd.DataFrame(data)
        mean = data.mean()
        std = data.std()

        for i in range(len(self.data)):
            for feature in self.conditioning_features:
                self.data[i][feature] = (self.data[i][feature] - mean[feature]) / std[feature]
            # for person in ['speaker', 'listener']:
            #     if self.data[i][person+'_gender'] == 'F':
            #         self.data[i][person+'_gender'] = self.gender_mapping['F']
            #     else:
            #         self.data[i][person+'_gender'] = self.gender_mapping['M']

        return mean, std
    
    def __len__(self):
        return len(self.data)
        # return int(100)


    def znormalize_landmarks_list(self):
        # if normalize_landmarks:
            # z-score normalize landmarks
            # all_listener_landmarks = np.vstack([i['smile_landmarks']['listener'] for i in self.data])
            # self.listener_landmarks_mean = all_listener_landmarks.mean(axis=0)
            # self.listener_landmarks_std = all_listener_landmarks.std(axis=0)

            # for i in range(len(self.data)):
            #     # z-score normalize speaker and listener landmarks
            #     self.data[i]['smile_landmarks']['listener'] -= self.listener_landmarks_mean
            #     self.data[i]['smile_landmarks']['listener'] /= self.listener_landmarks_std

        
        # normalization: Sign Language Production using Neural Machine Translation and Generative Adversarial Networks. Stoll et al. BMVC (2018)
        all_listener_landmarks = np.vstack([i['smile_landmarks'] for i in self.data])
        mean_face = all_listener_landmarks.mean(axis=0).reshape((-1, 2))
        mean_face_centroid = mean_face.mean(axis=0)
        # define translated landmarks
        translated_landmarks = np.zeros_like(all_listener_landmarks)
        for i in range(len(self.data)):
            self.data[i]['smile_displacements'] = list()
            # normalize the preceeding frame as well
            frame = self.data[i]['listener_preceeding_landmarks'][0].reshape((-1, 2))
            frame_centroid = frame.mean(axis=0)
            frame = frame + (mean_face_centroid - frame_centroid) # did not do this for 0.03 MSE model
            scale = np.abs(mean_face[1, :] - mean_face[10, :]) / np.abs(frame[1, :] - frame[10, :])
            self.data[i]['listener_preceeding_landmarks'][0] = (mean_face_centroid + (frame - mean_face_centroid) * scale).reshape((-1))
            for fid, frame in enumerate(self.data[i]['smile_landmarks']):
                frame = frame.reshape((-1, 2))
                frame_centroid = frame.mean(axis=0)
                frame = frame + (mean_face_centroid - frame_centroid)
                # define scaling factor (or width)--use extreme landmarks of both eyebrows [1, 10] correspond to face's left and right
                scale = np.abs(mean_face[1, :] - mean_face[10, :]) / np.abs(frame[1, :] - frame[10, :])

                # replace with normalized face
                self.data[i]['smile_landmarks'][fid] = (mean_face_centroid + (frame - mean_face_centroid) * scale).reshape((-1))

                if fid == 0:
                    displacement = self.data[i]['smile_landmarks'][0] - self.data[i]['listener_preceeding_landmarks'][0]
                else:
                    displacement = self.data[i]['smile_landmarks'][fid] -  self.data[i]['smile_landmarks'][fid-1]
                self.data[i]['smile_displacements'].append(displacement)

        # match min-max of lip corner spread across people in the dataset
        self.person_level_min_max = dict()
        for i in self.persons_in_dataset:
            self.person_level_min_max[i] = dict()
            person_displacements = np.vstack([j['smile_displacements'] for j in self.data if j['id'].split('_frame')[0]+'_'+j['person'] == i])
            # person_displacements = self.get_person_data(video=i.replace('_left', '').replace('_right', ''), person=['left' if i.endswith('left') else 'right'][0], key='smile_displacements')
            self.person_level_min_max[i]['min'] = np.min(person_displacements, axis=0)
            self.person_level_min_max[i]['max'] = np.max(person_displacements, axis=0)
            self.person_level_min_max[i]['mean'] = np.mean(person_displacements, axis=0)
            self.person_level_min_max[i]['std'] = np.std(person_displacements, axis=0)

        for i in range(len(self.data)):
            person = self.data[i]['id'].split('_frame')[0] + '_' + self.data[i]['person']
            # do min-max normalization on person level
            self.data[i]['smile_displacements'] = ((self.data[i]['smile_displacements'] - self.person_level_min_max[person]['min']) / (self.person_level_min_max[person]['max'] - self.person_level_min_max[person]['min'])).tolist()
            # do z-score norm
            # self.data[i]['smile_displacements'] = ((self.data[i]['smile_displacements'] - self.person_level_min_max[person]['mean']) / (self.person_level_min_max[person]['std'])).tolist()
            self.data[i]['smile_displacements_norm'] = {'min': self.person_level_min_max[person]['min'], 'max': self.person_level_min_max[person]['max'], 'mean': self.person_level_min_max[person]['mean'], 'std': self.person_level_min_max[person]['std']}
        return
    

    def plot_normalization_effect(self):
        # pre-normalize landmarks
        pre_normalize_landmarks = dict()
        for i in self.persons_in_dataset:
            pre_normalize_landmarks[i] = dict()
            pre_normalize_landmarks[i]['pre_norm'] = self.get_person_data(video=i.replace('_left', '').replace('_right', ''), person=['left' if i.endswith('left') else 'right'], key='smile_displacements')

        for i in self.persons_in_dataset:
            pre_normalize_landmarks[i] = dict()
            pre_normalize_landmarks[i]['pre_norm'] = self.get_person_data(video=i.replace('_left', '').replace('_right', ''), person=['left' if i.endswith('left') else 'right'], key='smile_displacements')


        return
    
    # def get_person_data(self, video, person, key):
    #     id = video + '_' + person
    #     return np.vstack([j[key] for j in self.data if j['id'].split('_frame')[0]+'_'+j['person'] == id])
    

    def z_normalize(self, data, normalize_landmarks):
        # z-score normalize conditioning features
        data[self.conditioning_features] -= self.mean
        data[self.conditioning_features] /= self.std

        if normalize_landmarks:

            data['smile_landmarks']['listener'] -= self.mean['smile_landmarks']['listener']
            data['smile_landmarks']['listener'] /= self.std['smile_landmarks']['listener']
            data['smile_landmarks']['speaker'] -= self.mean['smile_landmarks']['speaker']
            data['smile_landmarks']['speaker'] /= self.std['smile_landmarks']['speaker']

        return data


    @staticmethod
    def validate_files(data):
        # missing_listener_audio = [i for i in self.data['listener_turn_audio_filename'] if not os.path.exists(i)]
        # missing_speaker_audio = [i for i in self.data['speaker_turn_audio_filename'] if not os.path.exists(i)]
        valid_listener_audio = pd.Series([isinstance(i, str) and os.path.isfile(i) for i in data['listener_turn_audio_filename']])
        valid_speaker_audio = pd.Series([isinstance(i, str) and os.path.isfile(i) for i in data['speaker_turn_audio_filename']])
        valid_listener_text = pd.Series([isinstance(i, str) for i in data['listener_turn_text']])
        valid_speaker_text = pd.Series([isinstance(i, str) for i in data['speaker_turn_text']])
        valid_indices = valid_listener_audio & valid_speaker_audio & valid_listener_text & valid_speaker_text
        return valid_indices.tolist(), valid_listener_audio, valid_speaker_audio, valid_listener_text, valid_speaker_text    
    
    def __getitem__(self, index):
        row_data = self.data[index]
        label = row_data['IS_SMILE']
        amplitude = row_data['intensity_max']
        duration = row_data['duration'] # ms to sec conversion
        speaker_turn_audio_path = row_data['speaker_turn_audio_filename']
        listener_turn_audio_path = row_data['listener_turn_audio_filename']
        speaker_turn_audio = row_data['speaker_turn_audio_embeddings']
        listener_turn_audio = row_data['listener_turn_audio_embeddings']
        preceeding_landmarks = row_data['listener_preceeding_landmarks']
        
        speaker_turn_text = row_data['speaker_turn_text']
        listener_turn_text = row_data['listener_turn_text']
        smile_start_time = row_data['start_timestamp']
        normalization_params = row_data['smile_displacements_norm']

        video_name = row_data['id'].split('_frame')[0]+'_'+row_data['person']+'_frame' +'_'.join(row_data['id'].split('_frame')[1:])
        
        # # do augmentation
        # noise = torch.rand(row_data['smile_displacements'].shape)
        # row_data['smile_displacements'] = row_data['smile_displacements'] + noise

        conditioning_vector = [row_data[i] for i in self.conditioning_features]

        return {'audio': {'speaker': speaker_turn_audio, 'listener': listener_turn_audio, 'speaker_file': speaker_turn_audio_path, 'listener_file': listener_turn_audio_path}, 
                'text': {'speaker': speaker_turn_text, 'listener': listener_turn_text}, 
                'label': {'label': label, 'amplitude': amplitude, 'duration': duration, 'smile_landmarks': row_data['smile_landmarks'], 
                          'smile_preceeding_landmarks': preceeding_landmarks, 'smile_displacements': row_data['smile_displacements']}, 
                'conditioning_vector': conditioning_vector, 'smile_start_time': smile_start_time, 'listener': row_data['person'], 
                'video': video_name, 'normalization_params': normalization_params, 'smile_idx': row_data['smile_idx']}

def collate_fn(examples):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    speaker_audio = torch.stack([torch.Tensor(example['audio']['speaker']) for example in examples]).squeeze()
    listener_audio = torch.stack([torch.Tensor(example['audio']['listener']) for example in examples]).squeeze()
    speaker_audio_file = [os.path.basename(example['audio']['speaker_file']).replace('.mp4', '') for example in examples]
    listener_audio_file = [os.path.basename(example['audio']['listener_file']).replace('.mp4', '') for example in examples]
    # speaker_text_tokens = torch.stack([example['text_tokens']['speaker']["input_ids"] for example in examples])
    # speaker_text_attn_mask = torch.stack([example['text_tokens']['speaker']["attention_mask"] for example in examples])
    # listener_text_tokens = torch.stack([example['text_tokens']['listener']["input_ids"] for example in examples])
    # listener_text_attn_mask = torch.stack([example['text_tokens']['listener']["attention_mask"] for example in examples])
    speaker_text_tokens = None
    speaker_text_attn_mask = None
    listener_text_tokens = None
    listener_text_attn_mask = None
    
    video_name = [example['video'] for example in examples]
    smile_idx = [example['smile_idx'] for example in examples]
    normalization_params = [example['normalization_params'] for example in examples]
    conditioning_vector = torch.stack([torch.Tensor(example['conditioning_vector']) for example in examples])
    context_length = examples[0]['label']['smile_landmarks'].shape[0]
    counter = np.linspace(0, 1, context_length)
    
    # each frame landmark is now concatenated with label, amplitude, and duration; (B, context_length, (49*2+3))
    # context_length dim is required to add positional encoding when input to the decoder
    _smile_landmarks = list()
    _smile_displacements = list()
    batch_size = 0
    for eid, example in enumerate(examples):
        landmarks = torch.Tensor(example['label']['smile_landmarks'])
        displacements = torch.Tensor(example['label']['smile_displacements'])
        label = torch.Tensor([example['label']['label']]).long().repeat(context_length, 1)
        amplitude = torch.Tensor([example['label']['amplitude']]).long().repeat(context_length, 1)
        duration = torch.Tensor([example['label']['duration']]).float().repeat(context_length, 1)
        landmarks = torch.cat((landmarks, label, amplitude, duration), dim=1)
        displacements = torch.cat((displacements, label, amplitude, duration), dim=1)
        # counter dimension is added to the landmarks here--calculate_dtw in helpers.py requires count to be the last variable
        # landmarks = torch.cat((landmarks, torch.Tensor(counter).unsqueeze(1)), dim=1)
        _smile_landmarks.append(torch.cat((landmarks, torch.Tensor(counter).unsqueeze(1)), dim=1))
        _smile_displacements.append(torch.cat((displacements, torch.Tensor(counter).unsqueeze(1)), dim=1))
        # _smile_landmarks.append(landmarks)
        batch_size += 1
    smile_landmarks = torch.stack(_smile_landmarks)
    smile_displacements = torch.stack(_smile_displacements)

    label = torch.stack([torch.Tensor([example['label']['label']]).long() for example in examples])
    duration = torch.stack([torch.Tensor([example['label']['duration']]) for example in examples])
    amplitude = torch.stack([torch.Tensor([example['label']['amplitude']]) for example in examples])
    preceeding_landmarks = torch.stack([torch.Tensor(example['label']['smile_preceeding_landmarks']) for example in examples])

    return {
        "audio": {'speaker': speaker_audio.squeeze(dim=1).to(device), 'listener': listener_audio.squeeze(dim=1).to(device)},
        # "text_tokens": {'speaker': speaker_text_tokens.to(device), 'listener': listener_text_tokens.to(device)},
        # "text_masks": {'speaker': speaker_text_attn_mask.to(device), 'listener': listener_text_attn_mask.to(device)},
        "label": {'label': label, "duration": duration, "amplitude": amplitude, 
        "smile_landmarks": {'decoder_input': smile_landmarks.to(device), 'decoder_output': smile_landmarks[:, -1, :].to(device), 'preceeding_landmarks': preceeding_landmarks.to(device)},
        "smile_displacements": smile_displacements.to(device)},
                #   predicted frame is not padded as input
        # "smile_landmarks": {'decoder_input': smile_landmarks[:, :-1, :].to(device), 'decoder_output': smile_landmarks[:, -1, :].to(device)}}, 
        "conditioning_vector": conditioning_vector.to(device), 'batch_size': batch_size, "return_loss": False, 'normalization_params': normalization_params,
        "speaker_audio_file": speaker_audio_file, "listener_audio_file": listener_audio_file, "video_name": video_name, 'smile_idx': smile_idx
    }


def split_annotations(smile_data, landmark_generation, training_proportion=0.70, testing_proportion=0.15):
    if landmark_generation:
        folds_file = './data/landmark_generation_folds.pkl'
    else:
        folds_file = './folds1.pkl'
    if os.path.exists(folds_file):
        with open(folds_file, 'rb') as f:
            folds = pickle.load(f)
            return folds
    
    if landmark_generation:
        train_test_sgkf = GroupKFold(n_splits=10)
        train_valid_sgkf = GroupKFold(n_splits=9)
        groups = [i['video'] for i in smile_data]
        folds = list()
        for i, (train_valid_index, test_index) in enumerate(train_test_sgkf.split(smile_data, groups=groups)):
            train_valid_data = [smile_data[i] for i in train_valid_index]
            train_valid_groups = [i['video'] for i in train_valid_data]
            for (train_index, valid_index) in train_valid_sgkf.split(train_valid_data, groups=train_valid_groups):
                train_samples = [train_valid_groups[i] for i in train_index]
                valid_samples = [train_valid_groups[i] for i in valid_index]
                break
            test_samples = [groups[i] for i in test_index]
            unique_train_videos = set(train_samples)
            unique_valid_videos = set(valid_samples)
            unique_test_videos = set(test_samples)
            train_test_common_videos = unique_train_videos.intersection(unique_test_videos)
            valid_test_common_videos = unique_valid_videos.intersection(unique_test_videos)
            print(f"Fold {i+1}: {len(train_test_common_videos)} common videos between {len(unique_train_videos)} train and {len(unique_test_videos)} test")
            print(f"Fold {i+1}: {len(valid_test_common_videos)} common videos between {len(unique_valid_videos)} validation and {len(unique_test_videos)} test")
            folds.append({'train': train_samples, 'valid': valid_samples, 'test': test_samples})            
    else:
        raise NotImplementedError
    with open(folds_file, 'wb') as f:
        pickle.dump(folds, f)
    return folds


if __name__ == '__main__':
    dataset_creator = DatasetCreator()
    # test = TurnDataset(context_length=8, downsample_rate=3, fold=0, partition='test')
    # out = test[0]
