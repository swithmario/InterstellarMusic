# Interstellar Music Isolation - Channel-Specific Processing Script
# Version: 0.2.3 (Restored EXPLICIT merge map, configured for 1-min 5-audio clip test)

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
MKV_FILE = "interstellar_clip_1min_5audio.mkv"
ORIGINAL_AUDIO_TRACK_SPECIFIERS = ['0:1', '0:2', '0:3', '0:4', '0:5'] 
LANGUAGE_CODES = ['eng', 'fra', 'deu', 'ita', 'spa']
CHAPTER_DEFINITIONS = [("TestClipFull", 0.0, 60.0)] 
CHANNEL_LAYOUT_STRING = "5.1"
STANDARD_CHANNEL_NAMES = ['FL', 'FR', 'FC', 'LFE', 'SL', 'SR']
OUTPUT_DIR_BASE = "interstellar_clip_5audio_EXPLICIT_MERGE_test" # New output dir
TARGET_SAMPLE_RATE = 48000
ALIGNMENT_SEGMENT_DURATION_S = 58 
STFT_N_FFT = 2048
STFT_HOP_LENGTH = 512

# --- Helper Functions --- (Identical to Version 0.2.2)
def ensure_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")

def extract_and_split_one_language_track(mkv_path, track_specifier_for_mkv, lang_code,
                                         temp_storage_dir, channel_names, channel_layout_str,
                                         segment_start_time, segment_duration):
    print(f"\nProcessing language: {lang_code} (Track Specifier in '{os.path.basename(mkv_path)}': {track_specifier_for_mkv}) for segment {segment_start_time}-{segment_start_time+segment_duration}s")
    language_channel_paths = {}
    temp_multichannel_wav = os.path.join(temp_storage_dir, f"{lang_code}_temp_multichannel_segment.wav")
    try:
        print(f"  Extracting segment from multi-channel track {track_specifier_for_mkv} for {lang_code}...")
        extract_command = ['ffmpeg', '-y', '-ss', str(segment_start_time), '-i', mkv_path, '-t', str(segment_duration), '-map', track_specifier_for_mkv, '-acodec', 'pcm_s24le', '-ar', str(TARGET_SAMPLE_RATE), temp_multichannel_wav]
        print(f"    Executing FFmpeg: {' '.join(extract_command)}")
        subprocess.run(extract_command, check=True, capture_output=True, text=True)
        print(f"    Successfully extracted segment for {lang_code} to: {temp_multichannel_wav}")
        print(f"  Splitting multi-channel segment for {lang_code} into mono channels...")
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
        for ch_name, path in language_channel_paths.items(): print(f"      Split {lang_code} channel {ch_name} to: {path}")
        if os.path.exists(temp_multichannel_wav):
            os.remove(temp_multichannel_wav)
            print(f"    Removed temporary multi-channel file: {temp_multichannel_wav}")
    except subprocess.CalledProcessError as e:
        print(f"  ERROR processing language {lang_code} with FFmpeg.\n    Command: {' '.join(e.cmd)}\n    FFmpeg stdout: {e.stdout}\n    FFmpeg stderr: {e.stderr}")
        return {}
    except Exception as e_gen:
        print(f"  An unexpected error occurred processing language {lang_code}: {e_gen}"); traceback.print_exc()
        return {}
    return language_channel_paths

def group_mono_files_by_channel_role(all_languages_channel_data, target_channel_names):
    print("\nGrouping mono channel files by their role...")
    grouped_by_role = {ch_name: [] for ch_name in target_channel_names}
    for language_data in all_languages_channel_data:
        for ch_name, file_path in language_data.items():
            if ch_name in grouped_by_role: grouped_by_role[ch_name].append(file_path)
    for ch_name, files in grouped_by_role.items(): print(f"  Channel Role '{ch_name}': Found {len(files)} language versions.")
    return grouped_by_role

def load_standardize_and_align_mono_list(mono_file_paths, sample_rate, alignment_segment_s, ref_idx=0):
    if not mono_file_paths or len(mono_file_paths) < 2:
        print("  Not enough mono files to process (need at least 2 for alignment).")
        if len(mono_file_paths) == 1:
            try:
                data, _ = librosa.load(mono_file_paths[0], sr=sample_rate, mono=True)
                print(f"    Loaded single mono file, length: {len(data)} samples. No alignment performed.")
                return [data]
            except Exception as e: print(f"    Error loading single mono file {mono_file_paths[0]}: {e}"); return None
        return None
    print(f"  Loading {len(mono_file_paths)} mono files for this channel role...")
    loaded_mono_tracks_raw = []
    for f_path in mono_file_paths:
        try:
            data, _ = librosa.load(f_path, sr=sample_rate, mono=True)
            if len(data) == 0: print(f"    WARNING: Loaded empty audio from {f_path}. Skipping."); continue
            loaded_mono_tracks_raw.append(data)
        except Exception as e: print(f"    Error loading mono file {f_path}: {e}")
    if len(loaded_mono_tracks_raw) < 2: print(f"  Less than 2 valid mono tracks for role. Found {len(loaded_mono_tracks_raw)}."); return loaded_mono_tracks_raw if loaded_mono_tracks_raw else None
    min_len = min(len(track) for track in loaded_mono_tracks_raw)
    if min_len == 0: print("   ERROR: Min length 0. Cannot STFT."); return None
    standardized_mono_tracks = [track[:min_len] for track in loaded_mono_tracks_raw]
    print(f"    Standardized {len(standardized_mono_tracks)} mono tracks to {min_len} samples.")
    if len(standardized_mono_tracks) < 2: print("  Only one valid track after standardization. No alignment."); return standardized_mono_tracks
    print(f"  Aligning {len(standardized_mono_tracks)} mono tracks..."); reference_mono_track = standardized_mono_tracks[ref_idx]
    aligned_mono_tracks = [None] * len(standardized_mono_tracks); aligned_mono_tracks[ref_idx] = reference_mono_track.copy()
    segment_samples = min(min_len, int(alignment_segment_s * sample_rate))
    if segment_samples == 0: print("    Warning: Alignment segment 0. Using unaligned."); return standardized_mono_tracks
    for i in range(len(standardized_mono_tracks)):
        if i == ref_idx: continue
        current_mono_track = standardized_mono_tracks[i]; actual_segment_samples_for_corr = min(segment_samples, len(current_mono_track))
        if actual_segment_samples_for_corr == 0: print(f"      Track {i}: Corr segment 0. Zero shift."); delay_samples = 0
        else:
            ref_segment = reference_mono_track[:actual_segment_samples_for_corr]; pad_len_corr = len(ref_segment) - 1
            padded_current_track_corr = np.pad(current_mono_track, (pad_len_corr, pad_len_corr), 'constant')
            try:
                correlation = correlate(padded_current_track_corr, ref_segment, mode='valid', method='fft')
                delay_samples = -(np.argmax(correlation) - pad_len_corr) if correlation.size > 0 else 0
                print(f"      Track {i}: Shift {delay_samples} samples.")
            except Exception as e: print(f"      Error in correlation for track {i}: {e}. Zero shift."); delay_samples = 0
        shifted_track = np.roll(current_mono_track, delay_samples)
        if delay_samples > 0: shifted_track[:delay_samples] = 0
        elif delay_samples < 0: shifted_track[delay_samples:] = 0
        aligned_mono_tracks[i] = shifted_track
    print(f"    Alignment finished."); return aligned_mono_tracks

def process_mono_group_with_stft_median(aligned_mono_audio_list, n_fft, hop_length, original_length):
    if not aligned_mono_audio_list or len(aligned_mono_audio_list) == 0 : print("    No aligned mono tracks."); return None
    if len(aligned_mono_audio_list) == 1: print("    One track in group, no median."); return aligned_mono_audio_list[0]
    print(f"  Performing STFT on {len(aligned_mono_audio_list)} tracks..."); stfts_for_role = []
    for track_data in aligned_mono_audio_list:
        if len(track_data) == 0: print("    Skipping STFT for empty track."); continue
        try: stfts_for_role.append(librosa.stft(track_data, n_fft=n_fft, hop_length=hop_length))
        except Exception as e: print(f"    Error STFT for track: {e}")
    if not stfts_for_role or len(stfts_for_role) < 2:
        print(f"    Not enough STFTs for median (found {len(stfts_for_role)}).")
        if stfts_for_role: print("    Processing single STFT."); return librosa.istft(stfts_for_role[0], hop_length=hop_length, length=original_length)
        return None
    stft_stack = np.array(stfts_for_role); print(f"    Median STFT (stack: {stft_stack.shape})..."); median_stft_for_role = np.median(stft_stack, axis=0)
    print("    Performing ISTFT..."); processed_mono_audio = librosa.istft(median_stft_for_role, hop_length=hop_length, length=original_length)
    print("    STFT/Median/ISTFT finished."); return processed_mono_audio



# --- MERGE FUNCTION WITH POTENTIAL FIX FOR SL/SR MAPPING ---
def merge_processed_mono_to_multichannel(processed_mono_channel_paths_dict, output_wav_path,
                                         ordered_channel_names_for_merge, channel_layout_str):
    print(f"\nMerging processed mono channels into: {output_wav_path} (using EXPLICIT map in join filter - Attempt 4: BL/BR for SL/SR)")
    merge_command = ['ffmpeg', '-y']
    input_map_str_parts = []
    for i, ch_name in enumerate(ordered_channel_names_for_merge): # This uses FL,FR,FC,LFE,SL,SR to find files
        if ch_name not in processed_mono_channel_paths_dict: print(f"  ERROR: Missing processed file for '{ch_name}'. Cannot merge."); return False
        mono_file_path = processed_mono_channel_paths_dict[ch_name]
        if not os.path.exists(mono_file_path) or os.path.getsize(mono_file_path) < 100: print(f"  ERROR: File for '{ch_name}' ('{mono_file_path}') missing/empty. Cannot merge."); return False
        merge_command.extend(['-i', mono_file_path]); input_map_str_parts.append(f"[{i}:a]")
    if len(input_map_str_parts) != len(ordered_channel_names_for_merge): print("  ERROR: Not all channel files valid for merge."); return False
    
    join_filter_inputs_str = "".join(input_map_str_parts)
    
    # --- MODIFICATION HERE for explicit_map_str ---
    map_definitions = []
    # ordered_channel_names_for_merge is ['FL', 'FR', 'FC', 'LFE', 'SL', 'SR']
    # This defines the INPUT order [0:a] = FL, [1:a] = FR, etc.
    # Now we define the OUTPUT mapping for the join filter.
    for i, script_ch_name in enumerate(ordered_channel_names_for_merge):
        ffmpeg_map_name = script_ch_name # Default to script name
        if script_ch_name == 'FC':
            ffmpeg_map_name = 'FC' # Keep FC as it was accepted before SL failed. Or try 'C' if FC fails next.
        elif script_ch_name == 'SL':
            ffmpeg_map_name = 'BL' # Try Back Left for Side Left
        elif script_ch_name == 'SR':
            ffmpeg_map_name = 'BR' # Try Back Right for Side Right
        # FL, FR, LFE are usually fine as is.
        
        map_definitions.append(f"{i}.0-{ffmpeg_map_name}")
    explicit_map_str = "|".join(map_definitions)
    # This will generate: "0.0-FL|1.0-FR|2.0-FC|3.0-LFE|4.0-BL|5.0-BR"

    print(f"  Constructed explicit map string: {explicit_map_str}") # For debugging

    filter_complex_arg = (f"{join_filter_inputs_str}join=inputs={len(ordered_channel_names_for_merge)}"
                          f":channel_layout={channel_layout_str}:map={explicit_map_str}[a]")
    
    merge_command.extend(['-filter_complex', filter_complex_arg, '-map', '[a]', output_wav_path])
    try:
        print(f"  Executing FFmpeg merge: {' '.join(merge_command)}")
        subprocess.run(merge_command, check=True, capture_output=True, text=True)
        print(f"  Successfully merged channels to: {output_wav_path}"); return True
    except subprocess.CalledProcessError as e:
        print(f"  ERROR during FFmpeg merge.\n    Command: {' '.join(e.cmd)}\n    FFmpeg stdout: {e.stdout}\n    FFmpeg stderr: {e.stderr}"); return False
    except Exception as e_gen: print(f"  Unexpected error during merge: {e_gen}"); traceback.print_exc(); return False




# --- Main Execution Logic --- (Identical to Version 0.2.2)




if __name__ == "__main__":
    print("Starting Interstellar Music Isolation - Channel-Specific Processing..."); print(f"Source MKV: {MKV_FILE}")
    print(f"Using audio track specifiers (relative to MKV_FILE): {ORIGINAL_AUDIO_TRACK_SPECIFIERS}")
    print(f"With language codes: {LANGUAGE_CODES}"); print(f"Assuming channel layout: {CHANNEL_LAYOUT_STRING} with channels: {STANDARD_CHANNEL_NAMES}")
    if not os.path.exists(MKV_FILE): print(f"FATAL ERROR: Source MKV file not found at '{MKV_FILE}'."); exit(1)
    if len(ORIGINAL_AUDIO_TRACK_SPECIFIERS) != len(LANGUAGE_CODES): print("FATAL ERROR: Specifiers & lang_codes count mismatch."); exit(1)
    if len(ORIGINAL_AUDIO_TRACK_SPECIFIERS) < 2: print("FATAL ERROR: Need at least 2 tracks."); exit(1)
    run_output_dir_name = f"{OUTPUT_DIR_BASE}"; ensure_dir_exists(run_output_dir_name)
    for chapter_idx, (chapter_name, start_time, end_time) in enumerate(CHAPTER_DEFINITIONS):
        print(f"\n\n================ PROCESSING CHAPTER: {chapter_name} ({chapter_idx+1}/{len(CHAPTER_DEFINITIONS)}) ================ ")
        chapter_duration = end_time - start_time; print(f"Start: {start_time}s, End: {end_time}s, Duration: {chapter_duration:.3f}s")
        if chapter_duration <= 0: print(f"  Skipping chapter {chapter_name} (zero/negative duration)."); continue
        chapter_output_base_dir = os.path.join(run_output_dir_name, chapter_name); ensure_dir_exists(chapter_output_base_dir)
        chapter_processed_mono_output_dir = os.path.join(chapter_output_base_dir, "processed_mono_channels"); ensure_dir_exists(chapter_processed_mono_output_dir)
        with tempfile.TemporaryDirectory(prefix=f"interstellar_{chapter_name}_split_mono_") as temp_dir_for_splits:
            print(f"  Using temp dir for {chapter_name}: {temp_dir_for_splits}"); all_languages_mono_channel_data = []
            for i, source_track_spec in enumerate(ORIGINAL_AUDIO_TRACK_SPECIFIERS):
                lang_code = LANGUAGE_CODES[i]
                lang_specific_channel_paths = extract_and_split_one_language_track(
                    mkv_path=MKV_FILE, track_specifier_for_mkv=source_track_spec, lang_code=lang_code,
                    temp_storage_dir=temp_dir_for_splits, channel_names=STANDARD_CHANNEL_NAMES,
                    channel_layout_str=CHANNEL_LAYOUT_STRING, segment_start_time=start_time, segment_duration=chapter_duration)
                if lang_specific_channel_paths: all_languages_mono_channel_data.append(lang_specific_channel_paths)
                else: print(f"  Skipping lang {lang_code} for {chapter_name} (extraction/split error).")
            if len(all_languages_mono_channel_data) < 2: print(f"  FATAL for {chapter_name}: <2 langs extracted. Skipping chapter."); continue
            grouped_mono_files = group_mono_files_by_channel_role(all_languages_mono_channel_data, STANDARD_CHANNEL_NAMES)
            paths_of_final_processed_mono_channels = {}
            for channel_role in STANDARD_CHANNEL_NAMES:
                print(f"\n  --- Processing Channel Role: {channel_role} for {chapter_name} ---")
                mono_files_for_this_role = grouped_mono_files.get(channel_role, [])
                if len(mono_files_for_this_role) < 2: print(f"    <2 versions for '{channel_role}' ({len(mono_files_for_this_role)} found). Skipping role for {chapter_name}."); continue
                aligned_mono_tracks = load_standardize_and_align_mono_list(
                    mono_file_paths=mono_files_for_this_role, sample_rate=TARGET_SAMPLE_RATE, alignment_segment_s=ALIGNMENT_SEGMENT_DURATION_S)
                if not aligned_mono_tracks or not any(track is not None and track.size > 0 for track in aligned_mono_tracks): print(f"    Failed to load/align valid tracks for '{channel_role}' in {chapter_name}. Skipping."); continue
                valid_aligned_tracks = [track for track in aligned_mono_tracks if track is not None and track.size > 0]
                if not valid_aligned_tracks: print(f"    No valid aligned tracks for '{channel_role}' in {chapter_name}. Skipping."); continue
                length_for_istft = len(valid_aligned_tracks[0])
                if length_for_istft == 0: print(f"    ERROR: ISTFT length 0 for '{channel_role}' in {chapter_name}. Skipping."); continue
                processed_mono_audio_for_role = process_mono_group_with_stft_median(
                    aligned_mono_audio_list=valid_aligned_tracks, n_fft=STFT_N_FFT, hop_length=STFT_HOP_LENGTH, original_length=length_for_istft)
                if processed_mono_audio_for_role is None or processed_mono_audio_for_role.size == 0: print(f"    Failed to process audio/got empty for '{channel_role}' in {chapter_name}. Skipping."); continue
                output_mono_filename = f"processed_{chapter_name}_{channel_role}.wav"
                output_mono_filepath = os.path.join(chapter_processed_mono_output_dir, output_mono_filename)
                try:
                    sf.write(output_mono_filepath, processed_mono_audio_for_role, TARGET_SAMPLE_RATE)
                    print(f"    Saved processed '{channel_role}' for {chapter_name} to: {output_mono_filepath}"); paths_of_final_processed_mono_channels[channel_role] = output_mono_filepath
                except Exception as e: print(f"    Error saving '{channel_role}' for {chapter_name}: {e}"); traceback.print_exc()
            if len(paths_of_final_processed_mono_channels) == len(STANDARD_CHANNEL_NAMES):
                print(f"\n  All roles processed for {chapter_name}. Merging..."); final_merged_output_path = os.path.join(chapter_output_base_dir, f"final_merged_{chapter_name}_{CHANNEL_LAYOUT_STRING.replace('.', '_')}_output.wav")
                merge_success = merge_processed_mono_to_multichannel(
                    processed_mono_channel_paths_dict=paths_of_final_processed_mono_channels, output_wav_path=final_merged_output_path,
                    ordered_channel_names_for_merge=STANDARD_CHANNEL_NAMES, channel_layout_str=CHANNEL_LAYOUT_STRING)
                if merge_success: print(f"  Final merged for {chapter_name} to: {final_merged_output_path}")
                else: print(f"  Failed to merge for {chapter_name}.")
            else: print(f"\n  Not all roles processed for {chapter_name}. Skipping merge. Mono files in: {chapter_processed_mono_output_dir}")
        print(f"================ FINISHED CHAPTER: {chapter_name} ================")
    print("\n--- Interstellar Music Isolation Script Finished ---")