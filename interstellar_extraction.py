import subprocess
import os
import shutil
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import correlate
import tempfile

# --- Configuration ---
# VVVVVV  YOU MUST EDIT THESE TWO LINES BELOW  VVVVVV
MKV_FILE = "interstellar_clip_1min_5audio.mkv"
AUDIO_TRACK_SPECIFIERS = ['0:1', '0:2', '0:3', '0:4', '0:5']
# ^^^^^^  YOU MUST EDIT THESE TWO LINES ABOVE  ^^^^^^

OUTPUT_DIR = "interstellar_music_extraction"
OUTPUT_FILENAME = "interstellar_extracted_music.wav"

TARGET_SAMPLE_RATE = 48000  # Standard for Blu-ray audio
ALIGNMENT_SEGMENT_DURATION_S = 300 # Use 5 minutes for alignment calculation (can be reduced for faster tests)
STFT_N_FFT = 2048
STFT_HOP_LENGTH = 512

# --- Helper Functions ---

def create_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    return os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

def extract_audio_tracks(mkv_path, track_specifiers, temp_dir):
    extracted_files = []
    print("Extracting audio tracks...")
    for i, specifier in enumerate(track_specifiers):
        # Construct a unique filename for each track based on its specifier
        safe_specifier_name = specifier.replace(':', '_').replace('/', '_') # Make specifier filename-safe
        output_wav = os.path.join(temp_dir, f"track_{i}_{safe_specifier_name}.wav")
        try:
            command = [
                'ffmpeg', '-y', '-i', mkv_path,
                '-map', specifier,
                '-acodec', 'pcm_s24le', # Signed 24-bit PCM for quality
                '-ar', str(TARGET_SAMPLE_RATE),
                output_wav
            ]
            print(f"Executing: {' '.join(command)}")
            process = subprocess.run(command, check=True, capture_output=True, text=True)
            extracted_files.append(output_wav)
            print(f"Extracted {specifier} to {output_wav}")
        except subprocess.CalledProcessError as e:
            print(f"Error extracting track {specifier}: {e}")
            print(f"FFmpeg stdout: {e.stdout}")
            print(f"FFmpeg stderr: {e.stderr}")
            raise # Stop if any extraction fails
    return extracted_files

def load_and_standardize_audio(wav_files):
    print("Loading and standardizing audio...")
    audio_data_list_raw = []
    min_channels_found = float('inf')

    for f_idx, f_path in enumerate(wav_files):
        try:
            data, sr = librosa.load(f_path, sr=TARGET_SAMPLE_RATE, mono=False)
            if sr != TARGET_SAMPLE_RATE:
                 print(f"Warning: Track {f_idx} has sr {sr}, resampling to {TARGET_SAMPLE_RATE}")
                 data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)

            # Ensure data is 2D (channels, samples)
            if data.ndim == 1:
                data = data.reshape(1, -1) # Make mono 2D

            audio_data_list_raw.append(data)
            current_channels = data.shape[0]
            if current_channels < min_channels_found:
                min_channels_found = current_channels
        except Exception as e:
            print(f"Error loading or processing {f_path}: {e}")
            raise

    if not audio_data_list_raw:
        print("No audio data loaded.")
        return [], 0

    # Standardize channel count to the minimum found across all tracks
    standardized_audio_data = []
    for data in audio_data_list_raw:
        if data.shape[0] > min_channels_found:
            print(f"Warning: Truncating channels from {data.shape[0]} to {min_channels_found} for one track.")
            standardized_audio_data.append(data[:min_channels_found, :])
        else:
            standardized_audio_data.append(data)

    # Standardize length to the shortest track
    if not standardized_audio_data: # Should not happen if audio_data_list_raw was populated
        return [], 0
        
    min_length = min(track.shape[1] for track in standardized_audio_data)
    final_audio_data = [track[:, :min_length] for track in standardized_audio_data]
    
    print(f"All tracks standardized to {min_channels_found} channels and {min_length} samples.")
    return final_audio_data, min_channels_found


def align_tracks_channel_wise(audio_tracks_data, reference_track_idx=0, sample_rate=48000, segment_duration_s=300):
    print("Aligning tracks (channel-wise)...")
    if not audio_tracks_data or len(audio_tracks_data) <= 1:
        print("Not enough tracks to align or no tracks provided.")
        return audio_tracks_data

    num_tracks = len(audio_tracks_data)
    num_channels = audio_tracks_data[0].shape[0]
    reference_track = audio_tracks_data[reference_track_idx]
    
    aligned_tracks = [None] * num_tracks
    aligned_tracks[reference_track_idx] = reference_track.copy()

    segment_samples = min(reference_track.shape[1], int(segment_duration_s * sample_rate))
    if segment_samples == 0:
        print("Warning: Alignment segment is 0 samples. Skipping alignment.")
        # Return tracks as is, but ensure all are copies
        return [track.copy() for track in audio_tracks_data]


    for i in range(num_tracks):
        if i == reference_track_idx:
            continue

        current_track_to_align = audio_tracks_data[i]
        aligned_channels_for_current_track = []
        print(f"  Aligning track {i+1}/{num_tracks} to reference track {reference_track_idx+1}...")

        for ch_idx in range(num_channels):
            print(f"    Aligning channel {ch_idx+1}/{num_channels}...")
            ref_channel_segment = reference_track[ch_idx, :segment_samples]
            
            # For robust alignment, correlate against a slightly larger portion of the current track
            # to allow the segment to be found even if it's shifted.
            correlation_window_current_track = current_track_to_align[ch_idx, :] # Use full channel for now
            
            # Pad the signal being searched to avoid edge effects with 'valid' mode
            # Pad by len(ref_channel_segment) - 1 on both sides
            pad_len = len(ref_channel_segment) - 1
            padded_corr_window = np.pad(correlation_window_current_track, (pad_len, pad_len), 'constant')

            try:
                # Correlate the reference segment against the padded current track channel
                correlation = correlate(padded_corr_window, ref_channel_segment, mode='valid', method='fft')
                
                if correlation.size == 0:
                    print(f"      Warning: Correlation array empty for track {i+1}, ch {ch_idx+1}. Using zero shift.")
                    delay_samples = 0
                else:
                    # The lag is relative to the start of the 'valid' correlation output.
                    # A peak at index `k` in `correlation` means the ref_segment starts at `k` in `correlation_window_current_track`
                    # So, `padded_corr_window` started `pad_len` samples earlier.
                    # The delay of `correlation_window_current_track` relative to `ref_channel_segment`.
                    # lag_index = np.argmax(np.abs(correlation)) # More robust to phase inversions
                    lag_index = np.argmax(correlation) 
                    
                    # delay_samples: how many samples to shift current_track_to_align[ch_idx]
                    # to make it align with reference_track[ch_idx]
                    # If peak is at lag_index, it means ref_segment best matches starting at `lag_index` of the *unpadded* current_track
                    # So, if lag_index is positive, current track is "behind" ref, needs to be shifted left.
                    # delay = lag_index (if current track segment starts at lag_index)
                    # We want to shift current_track so its segment aligns with ref_segment
                    # The `correlate` output's 0-th index corresponds to ref_segment aligning with the start of correlation_window_current_track.
                    # Let's verify the lag calculation carefully.
                    # `delay_arr` in previous example was `np.arange(len(correlation)) - (len(segment_being_searched) - 1)`
                    # Here `segment_being_searched` is `ref_channel_segment`.
                    # And `correlation_window_current_track` is the signal it's searched in (after padding).
                    # The 'valid' mode of correlate means the output length is M-N+1, where M is len(padded_corr_window) and N is len(ref_channel_segment)
                    # A peak at an index `k` in the correlation means the `ref_channel_segment` aligns best when its start
                    # is placed at index `k` of `padded_corr_window`.
                    # Since `padded_corr_window` has `pad_len` padding at the start, the actual start in `correlation_window_current_track` is `k - pad_len`.
                    # This `k - pad_len` is the offset. If positive, `correlation_window_current_track` is delayed.
                    # We want to shift `correlation_window_current_track` by `-(k - pad_len)`
                    delay_samples = -(lag_index - pad_len)


                print(f"      Track {i+1}, Ch {ch_idx+1}: Detected shift of {delay_samples} samples.")

                original_channel_data = current_track_to_align[ch_idx, :]
                shifted_channel = np.roll(original_channel_data, delay_samples)

                # Zero out samples that were rolled in from the other end
                if delay_samples > 0: # Shifted right (delayed), zero out the start
                    shifted_channel[:delay_samples] = 0
                elif delay_samples < 0: # Shifted left (advanced), zero out the end
                    shifted_channel[delay_samples:] = 0
                
                aligned_channels_for_current_track.append(shifted_channel)

            except Exception as e:
                print(f"      Error during alignment of track {i+1}, ch {ch_idx+1}: {e}. Using original channel.")
                aligned_channels_for_current_track.append(current_track_to_align[ch_idx, :].copy())
        
        aligned_tracks[i] = np.array(aligned_channels_for_current_track)
        
    # Final length check and truncation/padding to match reference track precisely
    ref_len = reference_track.shape[1]
    final_aligned_tracks = []
    for track_idx, track_data in enumerate(aligned_tracks):
        if track_data.shape[1] > ref_len:
            final_aligned_tracks.append(track_data[:, :ref_len])
        elif track_data.shape[1] < ref_len:
            padding_needed = ref_len - track_data.shape[1]
            padded_track = np.pad(track_data, ((0,0), (0, padding_needed)), 'constant')
            final_aligned_tracks.append(padded_track)
        else:
            final_aligned_tracks.append(track_data)

    print("Alignment finished.")
    return final_aligned_tracks


def process_audio(aligned_audio_tracks, num_output_channels):
    print("Processing audio (STFT -> Median -> ISTFT) per channel...")
    if not aligned_audio_tracks:
        print("No aligned audio tracks to process.")
        return np.array([])

    num_tracks_in_list = len(aligned_audio_tracks)
    if num_tracks_in_list == 0:
        print("Aligned audio tracks list is empty.")
        return np.array([])
        
    track_length = aligned_audio_tracks[0].shape[1]
    
    final_processed_channels = []

    for ch_idx in range(num_output_channels):
        print(f"  Processing channel {ch_idx + 1}/{num_output_channels}...")
        
        stfts_for_channel = []
        for track_idx in range(num_tracks_in_list):
            # Ensure the track has enough channels
            if ch_idx < aligned_audio_tracks[track_idx].shape[0]:
                channel_data = aligned_audio_tracks[track_idx][ch_idx, :]
                stft_result = librosa.stft(channel_data, n_fft=STFT_N_FFT, hop_length=STFT_HOP_LENGTH)
                stfts_for_channel.append(stft_result)
            else:
                print(f"    Warning: Track {track_idx+1} does not have channel {ch_idx+1}. Skipping for this track.")
        
        if not stfts_for_channel:
            print(f"    Error: No STFTs generated for channel {ch_idx+1}. Appending zeros.")
            # Append a silent channel of the correct length if no STFTs could be generated
            final_processed_channels.append(np.zeros(track_length))
            continue

        # Stack STFTs: (num_valid_tracks_for_channel, num_frequency_bins, num_time_frames)
        stft_stack = np.array(stfts_for_channel)
        
        print(f"    Calculating median STFT for channel {ch_idx + 1} (from {stft_stack.shape[0]} tracks)...")
        # Median of complex numbers: numpy.median computes it element-wise for real and imaginary parts.
        median_stft = np.median(stft_stack, axis=0)
        
        print(f"    Performing ISTFT for channel {ch_idx + 1}...")
        # Ensure 'length' matches the original track_length for ISTFT to avoid truncation/padding issues
        processed_channel_audio = librosa.istft(median_stft, hop_length=STFT_HOP_LENGTH, length=track_length)
        final_processed_channels.append(processed_channel_audio)

    if not final_processed_channels:
        print("No channels were processed.")
        return np.array([])
        
    output_audio = np.array(final_processed_channels)
    print("Audio processing finished.")
    return output_audio

# --- Main Execution ---
if __name__ == "__main__":
    if not os.path.exists(MKV_FILE):
        print(f"Error: MKV file not found at {MKV_FILE}")
        exit(1)

    if not AUDIO_TRACK_SPECIFIERS or len(AUDIO_TRACK_SPECIFIERS) < 2: # Need at least 2 for comparison
        print("Error: Please specify at least two AUDIO_TRACK_SPECIFIERS for comparison.")
        exit(1)

    output_file_full_path = create_output_dir()
    # Create a temporary directory that will be automatically cleaned up
    with tempfile.TemporaryDirectory(prefix="interstellar_audio_") as temp_dir:
        print(f"Created temporary directory: {temp_dir}")
        extracted_wav_files = []
        try:
            extracted_wav_files = extract_audio_tracks(MKV_FILE, AUDIO_TRACK_SPECIFIERS, temp_dir)
            if not extracted_wav_files or len(extracted_wav_files) < len(AUDIO_TRACK_SPECIFIERS):
                 raise Exception("Not all audio tracks were extracted successfully or fewer than specified.")

            audio_data_list, num_common_channels = load_and_standardize_audio(extracted_wav_files)
            if not audio_data_list: # Check if list is empty
                raise Exception("Failed to load or standardize audio tracks: No data returned.")
            if num_common_channels == 0:
                 raise Exception("No common audio channels found after standardization, or all tracks were empty.")
            
            # Use the refined alignment function
            aligned_audio = align_tracks_channel_wise(audio_data_list, sample_rate=TARGET_SAMPLE_RATE, segment_duration_s=ALIGNMENT_SEGMENT_DURATION_S)
            if not aligned_audio: # Check if list is empty
                raise Exception("Failed to align audio tracks: No data returned.")

            final_output_audio = process_audio(aligned_audio, num_common_channels)
            
            if final_output_audio.size == 0:
                raise Exception("Processing resulted in empty audio.")

            print(f"Saving processed audio to {output_file_full_path}...")
            # librosa.output.write_wav is deprecated, use soundfile
            sf.write(output_file_full_path, final_output_audio.T, TARGET_SAMPLE_RATE) # Transpose for (samples, channels)
            print("Done!")

        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
        # temp_dir is automatically removed when exiting the 'with' block
        print(f"Temporary directory {temp_dir} will be automatically cleaned up.")