from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random
import os
from tqdm import tqdm
import pickle

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import pearsonr

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
# from data import SmilesDataset, collate_fn, resample_trim_audio
from revive_dataset import TurnDataset, collate_fn
from plot_videos import plot_video
from metrics import ape, pck

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

seed = 51
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)



# resample_trim_audio()

import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


MAX_LENGTH = 8


class EncoderRNN(nn.Module):
    def __init__(self, num_gru_layers=1, gru_dropout=0):
        super(EncoderRNN, self).__init__()
        self.num_gru_layers = num_gru_layers

        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_gru_layers, dropout=gru_dropout, batch_first=True)
        self.hidden_size_matcher = nn.Linear(hidden_size+NUM_CONDITIONING_FACTORS, hidden_size)
        nn.init.xavier_uniform_(self.hidden_size_matcher.weight)

    def forward(self, input, conditioning_vector):
        speaker_embedding = input['speaker']
        listener_embedding = input['listener']
        # ignore listener behavior
        if not use_listener_turn:
            listener_embedding = torch.zeros(input['listener'].shape).to(device)

        output, hidden = self.gru(speaker_embedding, listener_embedding.mean(dim=1).unsqueeze(0))
        hidden = F.relu(self.hidden_size_matcher(torch.cat((hidden, conditioning_vector.unsqueeze(dim=0)), dim=2))) #.unsqueeze(0)
        return output, hidden
    

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_gru_layers, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.teacher_forcing_likelihood = 0.8
        self.num_gru_layers = num_gru_layers
        self.embedding = nn.Linear(output_size, hidden_size)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, num_layers=num_gru_layers, dropout=dropout_p, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        nn.init.xavier_uniform_(self.out.weight)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_preceeding_output, target_tensor=None, teacher_forcing_likelihood=0.8):
        batch_size = encoder_outputs.size(0)

        encoder_hidden = encoder_hidden.mean(dim=0).unsqueeze(0) # do this if the encoder has multiple layers but decoder has only 1 layer
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            # uniform sampling to decide teacher forcing
            use_target_likelihood = np.random.sample(1)[0]

            if use_target_likelihood < self.teacher_forcing_likelihood:
                decoder_input = target_preceeding_output.detach()
            else:
                decoder_input = torch.zeros(batch_size, 1, landmark_features, dtype=torch.float, device=device)

            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            # if target_tensor is not None:
            if use_target_likelihood < self.teacher_forcing_likelihood:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                decoder_input = decoder_output.detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        # decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))
        # embedded = input

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights


def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, teacher_forcing_likelihood=0.8):

    total_loss = 0
    for data in dataloader:
        # input_tensor, target_tensor = data
        audio_tensor = data['audio']
        conditioning_vector = data['conditioning_vector']
        attributes = data['label']
        # initial_landmark = data['label']['smile_landmarks']['preceeding_landmarks']
        initial_landmark = torch.zeros(data['label']['smile_landmarks']['decoder_input'][:, 0, :-4].unsqueeze(1).shape, device=device)
        # use this if you are predicting absolute landmarks
        # target_tensor = data['label']['smile_landmarks']['decoder_input'][:, :, :-4]
        # use this if you are predicting displacement in landmarks
        target_tensor = data['label']['smile_displacements'][:, :, :-4]

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        encoder_outputs, encoder_hidden = encoder(audio_tensor, conditioning_vector)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, initial_landmark, target_tensor, teacher_forcing_likelihood)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1, decoder_outputs.size(-1))
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001, patience=5,
               print_every=100, plot_every=100, teacher_forcing_likelihood=0.8, teacher_forcing_reduce_every=50):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    print_train_loss_total = 0
    best_loss = np.inf

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate, weight_decay=1e-2, momentum=0.99)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate, weight_decay=1e-2, momentum=0.99)

    criterion = nn.MSELoss()
    encoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, 'min', patience=patience, factor=0.5, min_lr=1e-8, verbose=True)
    decoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, 'min', patience=patience, factor=0.5, min_lr=1e-8, verbose=True)

    for epoch in tqdm(range(1, n_epochs + 1)):
        if teacher_forcing_likelihood > 0:
            if epoch % teacher_forcing_reduce_every == 0:
                teacher_forcing_likelihood -= 0.05
                # print('Teacher forcing likelihood: ', teacher_forcing_likelihood)
        else:
            teacher_forcing_likelihood = 0
            # print('Teacher forcing likelihood reset: ', teacher_forcing_likelihood)
        train_loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_likelihood)
        if epoch % print_every == 0:
            loss, targets, preds, performance_metrics, _ = evaluate(encoder, decoder, validation_dataloader, criterion, epoch=epoch, generate_video=True)
        else:
            loss, _, _, _, _ = evaluate(encoder, decoder, validation_dataloader, criterion, epoch=epoch, generate_video=False)
        print_loss_total += loss
        plot_loss_total += loss
        print_train_loss_total += train_loss

        encoder_scheduler.step(loss)
        decoder_scheduler.step(loss)

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_train_loss_avg = print_train_loss_total / print_every
            print_loss_total = 0
            print_train_loss_total = 0
            _pck = sum(list(performance_metrics['landmark_accuracy'].values())) / 2
            _ape = sum(list(performance_metrics['landmark_position_error'].values())) / 2
            print(f"{timeSince(start, epoch / n_epochs)} ({epoch} {epoch / n_epochs * 100}%) LOSS: {print_loss_avg} PCK: {_pck} APE: {_ape}")
            writer.add_scalar('Loss/validation', print_loss_avg, epoch)
            writer.add_scalar('Loss/training', print_train_loss_avg, epoch)
            writer.add_scalar('LR', encoder_optimizer.param_groups[0]['lr'], epoch)
            writer.add_histogram('outputs/gt_displacements', targets, epoch)
            writer.add_histogram('outputs/pred_displacements', preds, epoch)
            # report avg across smiles and non-smiles
            writer.add_histogram('performance/PCK', _pck, epoch)
            writer.add_histogram('performance/APE', _ape, epoch)

            if print_loss_avg < best_loss:
                best_loss = print_loss_avg
                best_epoch = epoch
                torch.save({'encoder_state_dict': encoder.state_dict(), 'epoch': epoch, 'optimizer_state_dict': encoder_optimizer.state_dict()}, os.path.join(logdir, f'encoder_{epoch}.pth'))
                torch.save({'decoder_state_dict': decoder.state_dict(), 'epoch': epoch, 'optimizer_state_dict': decoder_optimizer.state_dict()}, os.path.join(logdir, f'decoder_{epoch}.pth'))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)
    print(f'Using epoch-{best_epoch} for testing')
    test_performance = test(encoder, decoder, best_epoch)
    return test_performance

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def evaluate(encoder, decoder, validation_dataloader, criterion, epoch=0, generate_video=False, num_instances_to_visualize=20):
    total_loss = 0
    decoder_outputs_viz = list()
    target_tensor_viz = list()
    preceeding_frames = list()
    attributes_viz = list()
    normalization_params = list()
    meta_data = dict()
    meta_data['ids'] = list()
    meta_data['duration'] = list()
    meta_data['IS_SMILE'] = list()
    meta_data['intensity'] = list()
    meta_data['smile_idx'] = list()
    with torch.no_grad():
        for data in validation_dataloader:
            audio_tensor = data['audio']
            conditioning_vector = data['conditioning_vector']
            attributes = data['label']
            meta_data['ids'].extend(data['video_name'])
            meta_data['duration'].extend(data['label']['duration'].squeeze().tolist())
            meta_data['intensity'].extend(data['label']['amplitude'].squeeze().tolist())
            meta_data['IS_SMILE'].extend(data['label']['label'].squeeze().tolist())
            meta_data['smile_idx'].extend(data['smile_idx'])
            # for plotting
            preceeding_frame = data['label']['smile_landmarks']['preceeding_landmarks']
            # for initial frame
            initial_landmark = torch.zeros(data['label']['smile_landmarks']['decoder_input'][:, 0, :-4].unsqueeze(1).shape, device=device)
            # use this if you are predicting absolute landmarks
            # target_tensor = data['label']['smile_landmarks']['decoder_input'][:, :, :-4]
            # use this if you are predicting displacement of landmarks
            target_tensor = data['label']['smile_displacements'][:, :, :-4]

            encoder_outputs, encoder_hidden = encoder(audio_tensor, conditioning_vector)
            decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, initial_landmark, target_tensor)

            if criterion is not None:
                loss = criterion(
                    decoder_outputs.view(-1, decoder_outputs.size(-1)),
                    target_tensor.view(-1, decoder_outputs.size(-1))
                )
                total_loss += loss.item()
            else:
                loss = 0
                total_loss = 0

            decoder_outputs = decoder_outputs.cpu().numpy()
            target_tensor = target_tensor.cpu().numpy()
            for i in range(len(decoder_outputs)):
                for j in range(len(decoder_outputs[i])):
                    decoder_outputs[i, j, :] = (decoder_outputs[i, j, :] * (data['normalization_params'][i]['max'] - data['normalization_params'][i]['min'])) + data['normalization_params'][i]['min']
                    target_tensor[i, j, :] = (target_tensor[i, j, :] * (data['normalization_params'][i]['max'] - data['normalization_params'][i]['min'])) + data['normalization_params'][i]['min']
                    # z-norm
                    # decoder_outputs[i, j, :] = (decoder_outputs[i, j, :] * data['normalization_params'][i]['std']) + data['normalization_params'][i]['mean']
                    # target_tensor[i, j, :] = (target_tensor[i, j, :] * data['normalization_params'][i]['std']) + data['normalization_params'][i]['mean']

            # decoder_outputs_viz.append(decoder_outputs.cpu().numpy())
            # target_tensor_viz.append(target_tensor.cpu().numpy())
            decoder_outputs_viz.append(decoder_outputs)
            target_tensor_viz.append(target_tensor)
            attributes_viz.append(attributes['label'])
            preceeding_frames.append(preceeding_frame.cpu().numpy())
            normalization_params.extend(data['normalization_params'])
            
    mean_loss = total_loss / len(validation_dataloader)

    decoder_outputs_viz = np.vstack(decoder_outputs_viz)
    target_tensor_viz = np.vstack(target_tensor_viz)
    attributes_viz = np.vstack(attributes_viz)
    preceeding_frames = np.vstack(preceeding_frames)
    meta_data['initial_landmarks'] = preceeding_frames
    
    max_examples_to_viz = min(num_instances_to_visualize, len(attributes_viz))
    smile_indices = np.where(attributes_viz == 1)[0]
    np.random.shuffle(smile_indices)
    smile_series = smile_indices[:max_examples_to_viz] # 10 samples for smiles and 10 for non-smiles
    meta_data['smile_ids'] = [i for i in meta_data['ids'] if i in smile_series]

    non_smile_indices = np.where(attributes_viz == 0)[0]
    np.random.shuffle(non_smile_indices)
    non_smile_series = non_smile_indices[:max_examples_to_viz] # 10 samples for smiles and 10 for non-smiles
    meta_data['non_smile_ids'] = [i for i in meta_data['ids'] if i in smile_series]

    if generate_video:
        landmark_accuracy = dict()
        landmark_mean_absolute_error = dict()
        for series, series_to_plot in {'smile': smile_series, 'non-smile': non_smile_series}.items():
            
            preds = list()
            gt = list()
            # batch sample
            for i in series_to_plot:
                gt_preceeding_frame = preceeding_frames[i, :].reshape(1, 1, -1).copy()
                pred_preceeding_frame = preceeding_frames[i, :].reshape(1, 1, -1).copy()
                # time series sample
                for j in range(len(decoder_outputs_viz[i])):
                    if predict_dynamics:
                        # decoder_outputs_viz[i, j, :] = (decoder_outputs_viz[i, j, :] * (_normalization_params['max'] - _normalization_params['min'])) + _normalization_params['min']
                        # target_tensor_viz[i, j, :] = (target_tensor_viz[i, j, :] * (_normalization_params['max'] - _normalization_params['min'])) + _normalization_params['min']
                        # use last known frame and predicted displacement for next frame prediction
                        current_predicted_frame = pred_preceeding_frame + decoder_outputs_viz[i, j, :] 
                        # use last known frame and gt. displacement for next frame prediction
                        current_gt_frame = gt_preceeding_frame + target_tensor_viz[i, j, :]
                        preds.append(current_predicted_frame)
                        gt.append(current_gt_frame)
                        # gt_preceeding_frame = current_gt_frame
                        # pred_preceeding_frame = current_predicted_frame
                    else:
                        preds.append(decoder_outputs_viz[i, j, :])
                        gt.append(target_tensor_viz[i, j, :])
            video_name = f"test_output_{series}_ep_{epoch}"
            plot_video(preds,
                        logdir,
                        video_name,
                        references=gt,
                        sequence_ID='validation_seq', FPS=8)
            landmark_accuracy[series] = pck(hypotheses=np.vstack(preds), references=np.vstack(gt))
            landmark_mean_absolute_error[series] = ape(hypotheses=np.vstack(preds), references=np.vstack(gt))
        performance_metrics = {'landmark_accuracy': landmark_accuracy, 'landmark_position_error': landmark_mean_absolute_error}
    else:
        performance_metrics = None
    return mean_loss, decoder_outputs_viz, target_tensor_viz, performance_metrics, meta_data


def test(encoder, decoder, best_epoch):
    if not os.path.isfile(os.path.join(logdir, f'encoder_best_model.pth')):
        # save model if it does not exist
        torch.save({'encoder_state_dict': encoder.state_dict(), 'epoch': best_epoch, 'optimizer_state_dict': None}, os.path.join(logdir, f'encoder_best_model.pth'))
        torch.save({'decoder_state_dict': decoder.state_dict(), 'epoch': best_epoch, 'optimizer_state_dict': None}, os.path.join(logdir, f'decoder_best_model.pth'))
    else:
        encoder.load_state_dict(torch.load(os.path.join(logdir, f'encoder_best_model.pth'))['encoder_state_dict'])
        decoder.load_state_dict(torch.load(os.path.join(logdir, f'decoder_best_model.pth'))['decoder_state_dict'])
    print('Loaded pretrained model')

    _, predictions, gts, test_performance, meta_data = evaluate(encoder, decoder, test_dataloader, criterion=None, epoch=best_epoch, generate_video=False, num_instances_to_visualize=1e8)
    landmark_accuracy = list()
    landmark_mean_absolute_error = list()
    landmark_correlation = list()
    persons_in_dataset = test_dataloader.dataset.persons_in_dataset
    performance = dict()
    performance['durations'] = list()
    performance['ids'] = list()
    performance['intensities'] = list()
    performance['smile_wise_performance'] = dict()
    _hypotheses = list()
    _gt = list()
    _durations = list()
    _intensities = list()
    _person = list()
    for person in persons_in_dataset:
        # why do you need to use set here? why is there duplication in the input data?
        person_smile_idx = dict()
        for i, name in enumerate(meta_data['ids']):
            if name.startswith(person):
                if meta_data['smile_idx'][i] not in person_smile_idx:
                    person_smile_idx[meta_data['smile_idx'][i]] = [i]
                else:
                    person_smile_idx[meta_data['smile_idx'][i]].append(i)
        for (smile, idx_in_dataset) in person_smile_idx.items():
            gt = list()
            preds = list()
            is_smile = meta_data['IS_SMILE'][idx_in_dataset[0]]
            duration = meta_data['duration'][idx_in_dataset[0]]
            intensity = meta_data['intensity'][idx_in_dataset[0]]
            # sort sequences of the "smile"
            person_ids_in_dataset = sorted([meta_data['ids'][i] for i in idx_in_dataset], key=lambda x: int(x.split('_frame_idx_')[-1]))
            # get the indices of the sorted sequences of the "smile"
            person_ids_in_dataset = [meta_data['ids'].index(i) for i in person_ids_in_dataset]
            # person_frames_sequence = sorted([int(meta_data['ids'][i].split('_frame_idx_')[-1]) for i in idx_in_dataset])
            # person_ids_in_dataset = list(set(person_frames_sequence).intersection(set(person_smile_idx[smile])))
            # person_ids_in_dataset = [i for i, name in enumerate(meta_data['ids']) for j in person_frames_sequence if (name.startswith(person+'_frame_idx_'+str(j))) and (meta_data['smile_idx'][i]==smile)]
            assert len(person_ids_in_dataset) == len(idx_in_dataset), print('number of smile sequences in dataset and retrieved number do not match')

            # person_frames_sequence = sorted(set([int(i.split('_')[-1]) for i in meta_data['ids'] if i.startswith(person)]))
            # person_ids_in_dataset = [i for i, name in enumerate(meta_data['ids']) for j in person_frames_sequence if name.startswith(person+'_frame_idx_'+str(j))]
            target_tensor_viz = gts[person_ids_in_dataset, :, :]
            decoder_outputs_viz = predictions[person_ids_in_dataset, :, :]
            preceeding_frame = meta_data['initial_landmarks'][person_ids_in_dataset, :, :][0] # we only need the first frame of the first sequence to reconstruct the entire sequence
            gt_preceeding_frame = preceeding_frame.reshape(1, 1, -1).copy()
            pred_preceeding_frame = preceeding_frame.reshape(1, 1, -1).copy()
            # gt_preceeding_frame = np.zeros(preceeding_frame.reshape(1, 1, -1).shape)
            # pred_preceeding_frame = np.zeros(preceeding_frame.reshape(1, 1, -1).shape)
            for pred_frame_id in range(len(decoder_outputs_viz)):
                # use last known frame and predicted displacement for next frame prediction
                current_predicted_frame = pred_preceeding_frame + decoder_outputs_viz[pred_frame_id, :, :]
                # use last known frame and gt. displacement for next frame prediction
                current_gt_frame = gt_preceeding_frame + target_tensor_viz[pred_frame_id, :, :]
                current_predicted_frame = current_predicted_frame.squeeze()
                current_gt_frame = current_gt_frame.squeeze()
                if pred_frame_id == 0:
                    # if the predicted frame is the first frame in the sequence, use entire sequence
                    preds.extend([current_predicted_frame[i, :].squeeze() for i in range(len(current_predicted_frame))])
                    gt.extend([current_gt_frame[i, :].squeeze() for i in range(len(current_gt_frame))])
                else:
                    # if the predicted frame is any other frame in the sequence, use only the last frame; rest of the frames are already covered in previous pred_frame_ids
                    preds.append(current_predicted_frame[-1, :])
                    gt.append(current_gt_frame[-1, :])
                # gt_preceeding_frame = current_gt_frame
                # pred_preceeding_frame = current_predicted_frame
            if is_smile:
                video_name = f"debug_test_output_{person}_smile_{smile}"
            else:
                video_name = f"debug_test_output_{person}_non-smile_{smile}"
            # plot_video(preds,
            #             logdir,
            #             video_name,
            #             references=gt,
            #             sequence_ID=person, FPS=8)
            _hypotheses.append(np.vstack(preds))
            _gt.append(np.vstack(gt))
            _durations.append(duration)
            _intensities.append(intensity)
            performance['smile_wise_performance'][smile] = dict()
            # use the last added element in the _hypotheses and _gt for current smile
            performance['smile_wise_performance'][smile]['smile_landmark_accuracy'] = pck(hypotheses=_hypotheses[-1], references=_gt[-1])
            performance['smile_wise_performance'][smile]['smile_landmark_mean_absolute_error'] = ape(hypotheses=_hypotheses[-1], references=_gt[-1])
            performance['smile_wise_performance'][smile]['smile_landmark_correlation'] = sum([pearsonr(_gt[-1][:, i], _hypotheses[-1][:, i])[0] for i in range(len(_gt[-1][0]))])/len(_gt[-1][0])
            performance['smile_wise_performance'][smile]['duration'] = duration
            performance['smile_wise_performance'][smile]['intensity'] = intensity
            performance['smile_wise_performance'][smile]['ids'] = (person, smile)
            _person.append((person, smile))
    hypotheses = np.vstack(_hypotheses)
    references = np.vstack(_gt)

    with open(os.path.join(logdir, 'test_predictions.pkl'), 'wb') as f:
        pickle.dump({'hypotheses': _hypotheses, 'gt': _gt, 'durations': _durations, 'intensities': _intensities, 'ids': _person, 'performance': performance['smile_wise_performance']}, f)
    landmark_accuracy.append(pck(hypotheses=hypotheses, references=references))
    landmark_mean_absolute_error.append(ape(hypotheses=hypotheses, references=references))
    # mean correlation across all landmarks
    landmark_correlation.append(sum([pearsonr(references[:, i], hypotheses[:, i])[0] for i in range(len(references[0]))])/len(references[0]))
    performance['durations'].extend(_durations)
    performance['intensities'].extend(_intensities)
    performance['ids'].append(_person)
    print(f"TEST PCK: {sum(landmark_accuracy)/len(landmark_accuracy)} APE: {sum(landmark_mean_absolute_error)/len(landmark_mean_absolute_error)} PEARSONR: {sum(landmark_correlation)/len(landmark_correlation)}")
    performance['landmark_accuracy'] = sum(landmark_accuracy)/len(landmark_accuracy)
    performance['landmark_position_error'] = sum(landmark_mean_absolute_error)/len(landmark_mean_absolute_error) 
    performance['landmark_correlation'] = sum(landmark_correlation)/len(landmark_correlation)
    
    for metric in ['duration', 'smile_landmark_accuracy', 'smile_landmark_mean_absolute_error', 'smile_landmark_correlation']:
        temp = list()
        for smile in performance['smile_wise_performance']:
            temp.append(performance['smile_wise_performance'][smile][metric])
        performance['mean_'+metric] = sum(temp)/len(temp)
    print('-'*20)
    print(f"MEAN TEST PCK: {performance['mean_smile_landmark_accuracy']} APE: {performance['mean_smile_landmark_mean_absolute_error']} PEARSONR: {performance['mean_smile_landmark_correlation']}")
    print('-'*20)
    return performance


hidden_size = 128
batch_size = 256

fold_id = 0
text_tokenizer = None
audio_normalizer = None
context_length = 8
frame_downsample = 3
visualizations_path = 'lstm_outputs/60s_speaker_turn_normalized'
predict_dynamics = True
use_conditioning = True # False returns empty list and model is slightly changed to accomodate this
use_listener_turn = True # False sets listener speech to zero in the encoder
os.makedirs(visualizations_path, exist_ok=True)

test_performances = list()
for fold_id in range(10):
    train_dataloader = DataLoader(dataset=TurnDataset(context_length=context_length, downsample_rate=frame_downsample, fold=fold_id, partition='training', use_conditioning_vectors=use_conditioning), batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    validation_dataloader = DataLoader(dataset=TurnDataset(context_length=context_length, downsample_rate=frame_downsample, fold=fold_id, partition='validation', use_conditioning_vectors=use_conditioning), batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    test_dataloader = DataLoader(dataset=TurnDataset(context_length=context_length, downsample_rate=frame_downsample, fold=fold_id, partition='test', use_conditioning_vectors=use_conditioning), batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    print(f"Using {len(train_dataloader.dataset)} for training, {len(validation_dataloader.dataset)} for validation and {len(test_dataloader.dataset)} for test")

    landmark_features = 98
    dropout = 0.3
    NUM_CONDITIONING_FACTORS = len(train_dataloader.dataset.conditioning_features)

    encoder = EncoderRNN(num_gru_layers=1, gru_dropout=dropout).to(device)
    decoder = AttnDecoderRNN(hidden_size, landmark_features, num_gru_layers=1, dropout_p=dropout).to(device)
    logdir = os.path.join(visualizations_path, f'runs/redo_test_with_listener_with_conditioning_vector/cv_{fold_id}') #
    writer = SummaryWriter(log_dir=logdir)

    test_perf = train(train_dataloader, encoder, decoder, 25, print_every=5, plot_every=5, learning_rate=1e-4, patience=20, 
        teacher_forcing_likelihood=0.9, teacher_forcing_reduce_every=20)
    test_performances.append(test_perf)
print(logdir.replace(f'cv_{fold_id}', 'test_performances.pkl'))
with open(logdir.replace(f'cv_{fold_id}', 'test_performances.pkl'), 'wb') as fp:
    pickle.dump(test_performances, fp)

perf_across_folds = dict()
for metric in ['duration', 'smile_landmark_accuracy', 'smile_landmark_mean_absolute_error', 'smile_landmark_correlation']:
    temp = list()
    for ifold in test_performances:
        temp.append(ifold['mean_'+metric])
    perf_across_folds['mean_across_folds_'+metric] = sum(temp)/len(temp)
print('-'*20)
print(f"MEAN TEST PCK: {perf_across_folds['mean_across_folds_smile_landmark_accuracy']} APE: {perf_across_folds['mean_across_folds_smile_landmark_mean_absolute_error']} \
      PEARSONR: {perf_across_folds['mean_across_folds_smile_landmark_correlation']}")
print('-'*20)
