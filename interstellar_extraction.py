# Interstellar Music Isolation - Channel-Specific Processing Script
# Version: 0.2.1 (Reverted merge function to simpler form for testing)

import subprocess
import os
import shutil
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import correlate
import tempfile
import traceback

# --- Configuration ---
# IMPORTANT: REVIEW AND CONFIRM/EDIT THESE SETTINGS BEFORE RUNNING!

# Path to your source MKV file (can be the full movie or a test clip)
MKV_FILE = "interstellar_clip_1min_5audio.mkv" # From your last successful clip creation
# For running on the full movie (AFTER successful testing with the clip):
# MKV_FILE = "/Volumes/Swiths 4TB SSD/Projects/Interstellar Project/Disc/UHD/Interstellar_t00.mkv"

# Original audio track specifiers FROM THE FULL MKV file that you want to use.
# These are the streams ffmpeg will extract before splitting them into channels.
# This should match the tracks you mapped into "interstellar_clip_1min_5audio.mkv"
# (e.g., if those were 0:1, 0:4, 0:6, 0:8, 0:9 from the original)
ORIGINAL_AUDIO_TRACK_SPECIFIERS = ['0:1', '0:2', '0:3', '0:4', '0:5'] # THESE ARE THE RE-MAPPED NUMBERS FOR THE CLIP
# If running on full MKV, use original specifiers like:
# ORIGINAL_AUDIO_TRACK_SPECIFIERS = ['0:1', '0:4', '0:6', '0:8', '0:9']


# Corresponding short language codes for naming files. MUST match the order and number of specifiers above.
LANGUAGE_CODES = ['eng', 'fra', 'deu', 'ita', 'spa'] # Matches 5 tracks

# Channel layout of your source audio tracks (e.g., "5.1", "stereo", "7.1")
CHANNEL_LAYOUT_STRING = "5.1"

# Standard channel names for your CHANNEL_LAYOUT_STRING.
STANDARD_CHANNEL_NAMES = ['FL', 'FR', 'FC', 'LFE', 'SL', 'SR']

# Output directory for all processed files
OUTPUT_DIR_BASE = "interstellar_channel_processed_output_reverted_merge" # Changed output dir to avoid overwrite

# --- Script Parameters ---
TARGET_SAMPLE_RATE = 48000
ALIGNMENT_SEGMENT_DURATION_S = 300
STFT_N_FFT = 2048
STFT_HOP_LENGTH = 512

# --- Helper Functions --- (Identical to before, repeated for completeness)

def ensure_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")

def extract_and_split_one_language_track(mkv_path, original_track_specifier, lang_code,
                                         temp_storage_dir, channel_names, channel_layout_str):
    print(f"\nProcessing language: {lang_code} (Track Specifier in MKV: {original_track_specifier})")
    language_channel_paths = {}
    temp_multichannel_wav = os.path.join(temp_storage_dir, f"{lang_code}_temp_multichannel.wav")

    try:
        print(f"  Extracting multi-channel track {original_track_specifier} for {lang_code}...")
        extract_command = [
            'ffmpeg', '-y', '-i', mkv_path,
            '-map', original_track_specifier, # This specifier is for the mkv_path
            '-acodec', 'pcm_s24le', '-ar', str(TARGET_SAMPLE_RATE),
            temp_multichannel_wav
        ]
        print(f"    Executing FFmpeg: {' '.join(extract_command)}")
        subprocess.run(extract_command, check=True, capture_output=True, text=True)
        print(f"    Successfully extracted multi-channel for {lang_code} to: {temp_multichannel_wav}")

        print(f"  Splitting multi-channel track for {lang_code} into mono channels...")
        channel_output_tags = "".join([f"[{name}]" for name in channel_names])
        filter_complex_arg = f"[0:a]channelsplit=channel_layout={channel_layout_str}{channel_output_tags}"
        
        split_command_base = ['ffmpeg', '-y', '-i', temp_multichannel_wav, '-filter_complex', filter_complex_arg]
        current_split_command = list(split_command_base)
        for ch_name in channel_names:
            mono_channel_output_path = os.path.join(temp_storage_dir, f"{lang_code}_{ch_name}.wav")
            current_split_command.extend(['-map', f"[{ch_name}]", mono_channel_output_path])
            language_channel_paths[ch_name] = mono_channel_output_path
        
        print(f"    Executing FFmpeg channelsplit: {' '.join(current_split_command)}")
        subprocess.run(current_split_command, check=True, capture_output=True, text=True)
        
        for ch_name, path in language_channel_paths.items():
            print(f"      Split {lang_code} channel {ch_name} to: {path}")
        
        if os.path.exists(temp_multichannel_wav):
            os.remove(temp_multichannel_wav)
            print(f"    Removed temporary multi-channel file: {temp_multichannel_wav}")

    except subprocess.CalledProcessError as e:
        print(f"  ERROR processing language {lang_code} with FFmpeg.")
        print(f"    Command: {' '.join(e.cmd)}\n    FFmpeg stdout: {e.stdout}\n    FFmpeg stderr: {e.stderr}")
        return {}
    except Exception as e_gen:
        print(f"  An unexpected error occurred processing language {lang_code}: {e_gen}")
        traceback.print_exc()
        return {}
    return language_channel_paths

def group_mono_files_by_channel_role(all_languages_channel_data, target_channel_names):
    print("\nGrouping mono channel files by their role...")
    grouped_by_role = {ch_name: [] for ch_name in target_channel_names}
    for language_data in all_languages_channel_data:
        for ch_name, file_path in language_data.items():
            if ch_name in grouped_by_role:
                grouped_by_role[ch_name].append(file_path)
    for ch_name, files in grouped_by_role.items():
        print(f"  Channel Role '{ch_name}': Found {len(files)} language versions.")
    return grouped_by_role

def load_standardize_and_align_mono_list(mono_file_paths, sample_rate, alignment_segment_s, ref_idx=0):
    if not mono_file_paths or len(mono_file_paths) < 2:
        print("  Not enough mono files to process (need at least 2).")
        if len(mono_file_paths) == 1:
            try:
                data, _ = librosa.load(mono_file_paths[0], sr=sample_rate, mono=True)
                return [data]
            except Exception as e:
                print(f"    Error loading single mono file {mono_file_paths[0]}: {e}")
                return None
        return None

    print(f"  Loading {len(mono_file_paths)} mono files for this channel role...")
    loaded_mono_tracks_raw = []
    for f_path in mono_file_paths:
        try:
            data, _ = librosa.load(f_path, sr=sample_rate, mono=True)
            loaded_mono_tracks_raw.append(data)
        except Exception as e:
            print(f"    Error loading mono file {f_path}: {e}")
            return None

    min_len = min(len(track) for track in loaded_mono_tracks_raw)
    standardized_mono_tracks = [track[:min_len] for track in loaded_mono_tracks_raw]
    print(f"    Standardized all mono tracks for this role to {min_len} samples.")

    print(f"  Aligning {len(standardized_mono_tracks)} mono tracks for this role...")
    reference_mono_track = standardized_mono_tracks[ref_idx]
    aligned_mono_tracks = [None] * len(standardized_mono_tracks)
    aligned_mono_tracks[ref_idx] = reference_mono_track.copy()

    segment_samples = min(len(reference_mono_track), int(alignment_segment_s * sample_rate))
    if segment_samples == 0 and len(standardized_mono_tracks) > 1:
        print("    Warning: Alignment segment is 0 samples. Using unaligned tracks for this role.")
        return standardized_mono_tracks

    for i in range(len(standardized_mono_tracks)):
        if i == ref_idx: continue
        current_mono_track = standardized_mono_tracks[i]
        ref_segment = reference_mono_track[:segment_samples]
        pad_len_corr = len(ref_segment) - 1
        padded_current_track_corr = np.pad(current_mono_track, (pad_len_corr, pad_len_corr), 'constant')
        try:
            correlation = correlate(padded_current_track_corr, ref_segment, mode='valid', method='fft')
            delay_samples = -(np.argmax(correlation) - pad_len_corr) if correlation.size > 0 else 0
            print(f"      Track {i}: Detected shift of {delay_samples} samples.")
            shifted_track = np.roll(current_mono_track, delay_samples)
            if delay_samples > 0: shifted_track[:delay_samples] = 0
            elif delay_samples < 0: shifted_track[delay_samples:] = 0
            aligned_mono_tracks[i] = shifted_track
        except Exception as e:
            print(f"      Error during alignment of track {i}: {e}. Using original standardized track.")
            aligned_mono_tracks[i] = current_mono_track.copy()
    print(f"    Alignment finished for this channel role.")
    return aligned_mono_tracks

def process_mono_group_with_stft_median(aligned_mono_audio_list, n_fft, hop_length, original_length):
    if not aligned_mono_audio_list: return None
    print(f"  Performing STFT on {len(aligned_mono_audio_list)} aligned mono tracks for this role...")
    stfts_for_role = [librosa.stft(track_data, n_fft=n_fft, hop_length=hop_length) for track_data in aligned_mono_audio_list]
    if not stfts_for_role: return None
    stft_stack = np.array(stfts_for_role)
    print(f"    Calculating median STFT (from stack shape: {stft_stack.shape})...")
    median_stft_for_role = np.median(stft_stack, axis=0)
    print("    Performing ISTFT...")
    processed_mono_audio = librosa.istft(median_stft_for_role, hop_length=hop_length, length=original_length)
    print("    STFT/Median/ISTFT processing finished for this channel role.")
    return processed_mono_audio

# --- REVERTED MERGE FUNCTION ---
def merge_processed_mono_to_multichannel(processed_mono_channel_paths_dict, output_wav_path,
                                         ordered_channel_names_for_merge, channel_layout_str):
    """
    Merges processed mono WAV files into a single multi-channel WAV file using FFmpeg.
    (Reverted to simpler join filter without explicit :map=... option for testing)
    """
    print(f"\nMerging processed mono channels into: {output_wav_path} (using simpler join filter)")
    
    merge_command = ['ffmpeg', '-y']
    input_streams_for_join = []
    
    for i, ch_name in enumerate(ordered_channel_names_for_merge):
        if ch_name not in processed_mono_channel_paths_dict:
            print(f"  ERROR: Missing processed file for channel '{ch_name}'. Cannot merge.")
            return False
        mono_file_path = processed_mono_channel_paths_dict[ch_name]
        if not os.path.exists(mono_file_path):
            print(f"  ERROR: Processed file for channel '{ch_name}' not found at '{mono_file_path}'. Cannot merge.")
            return False
        merge_command.extend(['-i', mono_file_path])
        input_streams_for_join.append(f"[{i}:a]") 

    if len(input_streams_for_join) != len(ordered_channel_names_for_merge):
        print("  ERROR: Number of input streams does not match number of expected channels. Cannot merge.")
        return False

    join_filter_inputs_str = "".join(input_streams_for_join)
    # THIS IS THE REVERTED, SIMPLER JOIN FILTER (NO EXPLICIT :map=... part)
    filter_complex_arg = f"{join_filter_inputs_str}join=inputs={len(ordered_channel_names_for_merge)}:channel_layout={channel_layout_str}[a]"
    
    merge_command.extend(['-filter_complex', filter_complex_arg, '-map', '[a]', output_wav_path])
    
    try:
        print(f"  Executing FFmpeg merge: {' '.join(merge_command)}")
        subprocess.run(merge_command, check=True, capture_output=True, text=True)
        print(f"  Successfully merged channels to: {output_wav_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ERROR during FFmpeg merge.")
        print(f"    Command: {' '.join(e.cmd)}\n    FFmpeg stdout: {e.stdout}\n    FFmpeg stderr: {e.stderr}")
        return False
    except Exception as e_gen:
        print(f"  An unexpected error occurred during merge: {e_gen}")
        traceback.print_exc()
        return False

# --- Main Execution Logic --- (Identical to before, repeated for completeness)
if __name__ == "__main__":
    print("Starting Interstellar Music Isolation - Channel-Specific Processing...")
    print(f"Source MKV: {MKV_FILE}")
    # IMPORTANT: If MKV_FILE is a clip, ORIGINAL_AUDIO_TRACK_SPECIFIERS must use the
    # re-mapped stream numbers from *within that clip*.
    # If MKV_FILE is the full movie, these are the original stream numbers from the full movie.
    print(f"Using audio track specifiers (relative to MKV_FILE): {ORIGINAL_AUDIO_TRACK_SPECIFIERS}")
    print(f"With language codes: {LANGUAGE_CODES}")
    print(f"Assuming channel layout: {CHANNEL_LAYOUT_STRING} with channels: {STANDARD_CHANNEL_NAMES}")

    if not os.path.exists(MKV_FILE):
        print(f"FATAL ERROR: Source MKV file not found at '{MKV_FILE}'. Please check the path.")
        exit(1)
    if len(ORIGINAL_AUDIO_TRACK_SPECIFIERS) != len(LANGUAGE_CODES):
        print("FATAL ERROR: The number of 'ORIGINAL_AUDIO_TRACK_SPECIFIERS' must match 'LANGUAGE_CODES'.")
        exit(1)
    if len(ORIGINAL_AUDIO_TRACK_SPECIFIERS) < 2:
        print("FATAL ERROR: Need at least 2 language tracks for comparison.")
        exit(1)

    ensure_dir_exists(OUTPUT_DIR_BASE)
    processed_mono_output_dir = os.path.join(OUTPUT_DIR_BASE, "processed_mono_channels")
    ensure_dir_exists(processed_mono_output_dir)

    with tempfile.TemporaryDirectory(prefix="interstellar_split_mono_") as temp_dir_for_splits:
        print(f"Using temporary directory for split mono files: {temp_dir_for_splits}")
        all_languages_mono_channel_data = []
        for i, source_track_spec in enumerate(ORIGINAL_AUDIO_TRACK_SPECIFIERS):
            lang_code = LANGUAGE_CODES[i]
            lang_specific_channel_paths = extract_and_split_one_language_track(
                mkv_path=MKV_FILE,
                original_track_specifier=source_track_spec, # This is the specifier for the MKV_FILE
                lang_code=lang_code,
                temp_storage_dir=temp_dir_for_splits,
                channel_names=STANDARD_CHANNEL_NAMES,
                channel_layout_str=CHANNEL_LAYOUT_STRING
            )
            if lang_specific_channel_paths:
                all_languages_mono_channel_data.append(lang_specific_channel_paths)
            else:
                print(f"  Skipping language {lang_code} due to errors in extraction/splitting.")
        
        if len(all_languages_mono_channel_data) < 2:
            print("FATAL ERROR: Less than 2 languages successfully extracted/split. Cannot proceed.")
            exit(1)

        grouped_mono_files = group_mono_files_by_channel_role(
            all_languages_mono_channel_data,
            STANDARD_CHANNEL_NAMES
        )
        paths_of_final_processed_mono_channels = {}
        for channel_role in STANDARD_CHANNEL_NAMES:
            print(f"\n--- Processing Channel Role: {channel_role} ---")
            mono_files_for_this_role = grouped_mono_files.get(channel_role, [])
            if len(mono_files_for_this_role) < 2:
                print(f"  Not enough versions for '{channel_role}' (found {len(mono_files_for_this_role)}). Skipping.")
                continue

            aligned_mono_tracks = load_standardize_and_align_mono_list(
                mono_file_paths=mono_files_for_this_role,
                sample_rate=TARGET_SAMPLE_RATE,
                alignment_segment_s=ALIGNMENT_SEGMENT_DURATION_S
            )
            if not aligned_mono_tracks:
                print(f"  Failed to load/standardize/align for '{channel_role}'. Skipping.")
                continue
            
            length_for_istft = len(aligned_mono_tracks[0])
            processed_mono_audio_for_role = process_mono_group_with_stft_median(
                aligned_mono_audio_list=aligned_mono_tracks,
                n_fft=STFT_N_FFT,
                hop_length=STFT_HOP_LENGTH,
                original_length=length_for_istft
            )
            if processed_mono_audio_for_role is None:
                print(f"  Failed to process audio for '{channel_role}'. Skipping.")
                continue

            output_mono_filename = f"processed_{channel_role}.wav"
            output_mono_filepath = os.path.join(processed_mono_output_dir, output_mono_filename)
            try:
                sf.write(output_mono_filepath, processed_mono_audio_for_role, TARGET_SAMPLE_RATE)
                print(f"  Successfully saved processed '{channel_role}' channel to: {output_mono_filepath}")
                paths_of_final_processed_mono_channels[channel_role] = output_mono_filepath
            except Exception as e:
                print(f"  Error saving processed '{channel_role}' channel: {e}")
                traceback.print_exc()
        
        if len(paths_of_final_processed_mono_channels) == len(STANDARD_CHANNEL_NAMES):
            print("\nAll channel roles processed. Attempting to merge into a final multi-channel file...")
            final_merged_output_path = os.path.join(OUTPUT_DIR_BASE, f"final_merged_{CHANNEL_LAYOUT_STRING.replace('.', '_')}_output.wav")
            merge_success = merge_processed_mono_to_multichannel(
                processed_mono_channel_paths_dict=paths_of_final_processed_mono_channels,
                output_wav_path=final_merged_output_path,
                ordered_channel_names_for_merge=STANDARD_CHANNEL_NAMES,
                channel_layout_str=CHANNEL_LAYOUT_STRING
            )
            if merge_success: print(f"Final merged multi-channel audio saved to: {final_merged_output_path}")
            else: print("Failed to merge processed mono channels.")
        else:
            print("\nNot all channel roles successfully processed. Skipping final multi-channel merge.")
            print(f"Processed mono channels are available in: {processed_mono_output_dir}")
    print("\n--- Interstellar Music Isolation Script Finished ---")