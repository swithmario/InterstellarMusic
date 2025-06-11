import os
import subprocess
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import tempfile
import shutil
from scipy.signal import correlate # <--- IMPORT ADDED
import traceback                 # <--- IMPORT ADDED
import matplotlib.pyplot as plt # For visualizing alignment (optional, but good for debugging)
# librosa.display will be used if you uncomment plotting
# ==============================================================================
# === CONFIGURATION (Pilot: Cornfield Chase vs FYC Chasing Drone) ===
# ==============================================================================

# --- Source File Paths ---
BLURAY_MKV_PATH = Path("/Volumes/Swiths 4TB SSD/Projects/Interstellar Project/Disc/UHD/Interstellar_t00.mkv")
FYC_AUDIO_PATH = Path('/Volumes/Swiths 4TB SSD/Projects/Interstellar Project/Audio/Music/2.mp3') # <--- !!! UPDATE THIS PATH !!!

# --- Scene Definition: "Cornfield Chase" from Blu-ray ---
# Note: Ensure FPS is exact (e.g., 23.976 or 24000/1001 if that's the case, not just 24)
BLURAY_SCENE_FPS = 23.976 # or 24000/1001
BLURAY_SCENE_START_FRAME = 8272
BLURAY_SCENE_END_FRAME = 11232

# --- Audio Track Selection from Blu-ray MKV ---
BLURAY_LANGUAGES_TO_PROCESS = {
    "eng": "0:1", # English DTS-HD MA
    "fra": "0:4", # French DTS-HD MA
    "ger": "0:6", # German DTS-HD MA
    # "ita": "0:8",
    # "spa": "0:9",
}

# --- Processing Parameters ---
TARGET_SAMPLE_RATE = 48000
OUTPUT_BIT_DEPTH_FFMPEG = "pcm_s24le" # Using 24-bit for intermediates

ALIGNMENT_SEGMENT_DURATION_S = 58 
STFT_N_FFT = 2048
STFT_HOP_LENGTH = 512

# --- Output Directory Structure ---
BASE_OUTPUT_DIR = Path("output_cornfield_pilot_v2") # Changed to avoid overwriting previous run

# --- Channel Configuration (Assuming 5.1 source) ---
CHANNEL_LAYOUT_STRING = "5.1" # THIS IS THE GLOBAL VARIABLE
CHANNEL_NAMES_5_1 = ['FL', 'FR', 'FC', 'LFE', 'SL', 'SR'] # THIS IS THE GLOBAL VARIABLE
CHANNELSPLIT_PAD_NAMES_5_1 = [f"[{name}]" for name in CHANNEL_NAMES_5_1] # THIS IS THE GLOBAL VARIABLE

# ==============================================================================
# === HELPER FUNCTIONS ===
# ==============================================================================
# (Helper functions extract_segment_and_split_channels, group_mono_files_by_channel_role,
#  load_audio_mono, process_channel_role_across_languages, etc. are assumed to be the
#  same as the last complete version I provided, so I'll omit them here for brevity
#  but they MUST be present in your actual script.)

def ensure_dir_exists(dir_path: Path):
    dir_path.mkdir(parents=True, exist_ok=True)
    # print(f"Ensured directory exists: {dir_path}") # Quieter logging

def frames_to_seconds(frame, fps):
    return frame / fps

def extract_segment_and_split_channels(
    mkv_path: Path, output_basedir_for_lang: Path, mkv_track_specifier: str,
    start_seconds: float, duration_seconds: float,
    passed_channel_layout_str: str, # Explicitly pass to avoid scope issues
    passed_channel_names_ordered: list,
    passed_ffmpeg_channel_pads: list
):
    ensure_dir_exists(output_basedir_for_lang)
    print(f"  Extracting & splitting '{mkv_track_specifier}' for {os.path.basename(output_basedir_for_lang)} ({duration_seconds:.2f}s)...")
    temp_multichannel_wav = output_basedir_for_lang / "_temp_multichannel_segment.wav"
    channel_file_paths = {}
    try:
        extract_cmd = ['ffmpeg',"-hide_banner",'-y','-ss',str(start_seconds),'-i',str(mkv_path),'-t',str(duration_seconds),'-map',mkv_track_specifier,'-acodec',OUTPUT_BIT_DEPTH_FFMPEG,'-ar',str(TARGET_SAMPLE_RATE),str(temp_multichannel_wav)]
        # print(f"    Running extract: {' '.join(extract_cmd)}")
        subprocess.run(extract_cmd, check=True, capture_output=True, text=True)
        filter_complex_arg = f"[0:a]channelsplit=channel_layout={passed_channel_layout_str}{''.join(passed_ffmpeg_channel_pads)}"
        split_cmd_base = ['ffmpeg',"-hide_banner",'-y','-i',str(temp_multichannel_wav),'-filter_complex',filter_complex_arg]
        current_split_cmd = list(split_cmd_base)
        for i, ch_name in enumerate(passed_channel_names_ordered):
            mono_output_path = output_basedir_for_lang / f"{ch_name}.wav"
            current_split_cmd.extend(["-map", passed_ffmpeg_channel_pads[i], str(mono_output_path)])
            channel_file_paths[ch_name] = mono_output_path
        # print(f"    Running split: {' '.join(current_split_cmd)}")
        subprocess.run(current_split_cmd, check=True, capture_output=True, text=True)
        # print(f"    Successfully split into {len(passed_channel_names_ordered)} mono channels in {output_basedir_for_lang}")
    except subprocess.CalledProcessError as e:
        print(f"  ERROR FFmpeg for {os.path.basename(output_basedir_for_lang)}: {e.stderr[:500]}...") # Print first 500 chars of stderr
        return None
    except Exception as exc:
        print(f"  Unexpected error for {os.path.basename(output_basedir_for_lang)}: {exc}"); traceback.print_exc(); return None
    finally:
        if temp_multichannel_wav.exists(): temp_multichannel_wav.unlink()
    return channel_file_paths

def load_audio_mono(file_path: Path, sr: int = TARGET_SAMPLE_RATE):
    try:
        audio, _ = librosa.load(file_path, sr=sr, mono=True)
        if len(audio) == 0: print(f"Warning: Loaded empty audio: {file_path}"); return None
        return audio
    except Exception as e: print(f"Error loading {file_path}: {e}"); return None

def process_channel_role_across_languages(
    list_of_mono_paths_for_role: list, output_processed_path: Path, sample_rate: int,
    alignment_segment_s: float, n_fft: int, hop_length: int
):
    # print(f"  Processing {len(list_of_mono_paths_for_role)} files for role, output to {output_processed_path.name}...")
    loaded_audios = []
    for p in list_of_mono_paths_for_role:
        audio = load_audio_mono(p, sr=sample_rate)
        if audio is not None: loaded_audios.append(audio)
    if len(loaded_audios) < 2:
        # print(f"    Need >= 2 valid audios, found {len(loaded_audios)}. Skipping role or copying single.")
        if len(loaded_audios) == 1 and loaded_audios[0] is not None:
            # print(f"    Copying single track: {list_of_mono_paths_for_role[0].name}")
            shutil.copy(list_of_mono_paths_for_role[0], output_processed_path); return True
        return False
    min_len = min(len(a) for a in loaded_audios)
    if min_len == 0: print("    Error: Min length 0. Skip role."); return False
    standardized_audios = [a[:min_len] for a in loaded_audios]
    # print(f"    Standardized to {min_len} samples.")
    aligned_audios = list(standardized_audios); reference_track = aligned_audios[0]
    actual_alignment_segment_s = min(alignment_segment_s, min_len / sample_rate)
    segment_samples = int(actual_alignment_segment_s * sample_rate)
    if segment_samples > 0 :
        for i in range(1, len(aligned_audios)):
            current_track = aligned_audios[i]; ref_segment = reference_track[:segment_samples]
            pad_len_corr = len(ref_segment) - 1 if len(ref_segment) > 0 else 0
            if pad_len_corr < 0 : continue
            padded_current_track_corr = np.pad(current_track, (pad_len_corr, pad_len_corr), 'constant')
            try:
                correlation = correlate(padded_current_track_corr, ref_segment, mode='valid', method='fft')
                delay_samples = -(np.argmax(correlation) - pad_len_corr) if correlation.size > 0 else 0
                # print(f"      Align track {i}: shift {delay_samples} samples.")
                shifted_track = np.roll(current_track, delay_samples)
                if delay_samples > 0: shifted_track[:delay_samples] = 0
                elif delay_samples < 0: shifted_track[delay_samples:] = 0
                aligned_audios[i] = shifted_track
            except Exception as e: print(f"      Error aligning track {i}: {e}")
    # else: print("    Alignment segment 0. Using unaligned.")
    stfts = []
    for audio_data in aligned_audios:
        if len(audio_data) < n_fft: print(f"    Audio too short for STFT. Skipping."); continue
        stfts.append(librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length))
    if len(stfts) < 2:
        # print(f"    Not enough STFTs for median (found {len(stfts)}).")
        if len(stfts) == 1: # print("    Using single available STFT."); sf.write(output_processed_path, aligned_audios[0], sample_rate); return True # Save the one good track from alignment
            # If only one track, it means no median filtering, so just use the aligned (or standardized) version.
            # Find which original track corresponds to stfts[0] if needed, or just use the first aligned audio.
            # The first element of aligned_audios is the reference track if others failed.
            sf.write(output_processed_path, aligned_audios[0] if aligned_audios else np.array([]), sample_rate)
            return True
        return False
    stft_stack = np.array(stfts); #print(f"    Median STFT stack: {stft_stack.shape}")
    median_stft = np.median(stft_stack, axis=0)
    processed_audio = librosa.istft(median_stft, hop_length=hop_length, length=min_len)
    try:
        sf.write(output_processed_path, processed_audio, sample_rate)
        # print(f"    Saved processed to {output_processed_path}")
        return True
    except Exception as e: print(f"    Error saving {output_processed_path}: {e}"); return False



# (Add this function alongside your other helper functions)

def align_and_warp_audio(
    target_audio_path: Path,    # Audio to be warped (e.g., processed_cornfield_FL.wav)
    reference_audio_path: Path, # Audio to align to (e.g., FYC_L.wav)
    output_warped_audio_path: Path,
    sample_rate: int = TARGET_SAMPLE_RATE,
    hop_length_chroma: int = 512 # Hop length for CQT/STFT for chroma
):
    """
    Aligns target_audio to reference_audio using DTW on chromagrams
    and saves the time-warped version of target_audio.
    """
    print(f"\n--- Aligning '{target_audio_path.name}' to '{reference_audio_path.name}' ---")

    try:
        y_target, sr_target = librosa.load(target_audio_path, sr=sample_rate)
        y_reference, sr_reference = librosa.load(reference_audio_path, sr=sample_rate)

        if sr_target != sample_rate or sr_reference != sample_rate:
            print("Warning: Sample rate mismatch after loading, this shouldn't happen if loaded with target sr.")
            # This case should ideally not occur if librosa.load is used with sr=sample_rate

        print("  Generating chromagrams...")
        # Using CQT-based chromagram as it's often good for music alignment
        chroma_target = librosa.feature.chroma_cqt(y=y_target, sr=sample_rate, hop_length=hop_length_chroma)
        chroma_reference = librosa.feature.chroma_cqt(y=y_reference, sr=sample_rate, hop_length=hop_length_chroma)

        print("  Calculating DTW path...")
        # D is the accumulated cost matrix, wp is the warping path
        D, wp = librosa.sequence.dtw(X=chroma_target, Y=chroma_reference, metric='cosine', backtrack=True)
        # wp is an array of index pairs (idx_target, idx_reference)

        print(f"    Warping path shape: {wp.shape}")
        # The warping path needs to be flipped because dtw returns Y_indices, X_indices
        wp_target_to_ref = wp[::-1, :] 

        print("  Warping target audio...")
        # Create a time-stretched version of y_target according to wp
        # We need to convert frame indices from chromagram back to audio sample indices
        # Each chroma frame corresponds to hop_length_chroma audio samples
        
        # y_target_warped = librosa.effects.time_stretch(y_target, rate=?) -> This is for uniform stretching.
        # For DTW, we need a more granular approach often called "piecewise time stretching"
        # or using the warping path to resample.
        # librosa doesn't have a direct "warp_audio_by_path" function.
        # A common way is to build an interpolated time series.

        # Let's try aligning reference to target for simplicity in this step, then apply inverse if needed,
        # or more practically, use the warping path to construct the aligned signal.
        # The goal is to make the target sound like it's playing in sync with the reference.
        # We want to "pull" samples from y_target based on the alignment to y_reference's timeline.

        # Create a mapping from reference frames to target frames
        ref_frames = np.arange(chroma_reference.shape[1])
        # For each reference frame, find the corresponding target frame from the warping path
        # wp_target_to_ref[:, 1] are reference indices, wp_target_to_ref[:, 0] are target indices
        
        # Create an interpolated mapping from reference time to target time
        # The wp gives us (target_frame_idx, reference_frame_idx)
        # We want to know, for each sample in the reference timeline, which sample to pick from the target.
        
        # Simpler approach for now: Use the warping path to generate a resampled y_target.
        # This is a complex step. A library like `pyrubberband` or manual interpolation is often needed
        # for high-quality arbitrary time warping based on a path.
        
        # For a first pass, let's create a "pseudo-aligned" version by selecting frames.
        # This is a simplification and might not be perfectly smooth.
        # A full implementation would involve phase vocoding or similar for high-quality warp.
        
        # Let's create a target audio that has the same number of chroma frames as the reference,
        # by picking/duplicating/skipping target frames based on wp.
        # This is more for visualizing alignment of features than direct audio warping for listening.

        # For now, let's just log that alignment is done and save the target.
        # The actual audio warping based on 'wp' is non-trivial to implement from scratch cleanly.
        # We will likely use the 'wp' to align features and then decide on cuts or crossfades.
        # Or, if we were to truly warp audio, we'd use a tool that can do it via phase vocoder.

        # Let's focus on getting the warping path and visually inspecting it first.
        # We can plot the alignment.
        if True: # Set to False to disable plotting if no GUI
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(D, x_axis='frames', y_axis='frames', sr=sample_rate, hop_length=hop_length_chroma)
            plt.title(f'DTW Cost Matrix: {target_audio_path.name} (y) vs {reference_audio_path.name} (x)')
            plt.plot(wp[:, 1], wp[:, 0], marker='o', color='r', markersize=3, alpha=0.3) # y_frames, x_frames
            plt.colorbar()
            plt.tight_layout()
            plot_path = output_warped_audio_path.with_name(f"dtw_path_{target_audio_path.stem}_vs_{reference_audio_path.stem}.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"    Saved DTW alignment plot to: {plot_path}")

        # For now, we are not actually creating a warped *audio* file yet,
        # as that requires more complex time-stretching algorithms based on the path.
        # The key output here is the warping path `wp` and the visual.
        # We will copy the original target audio for now as a placeholder.
        shutil.copy(target_audio_path, output_warped_audio_path)
        print(f"  Warping path calculated. Placeholder warped audio saved to: {output_warped_audio_path}")
        print(f"  Next step would be to apply this path (wp) to resample/warp '{target_audio_path.name}'.")
        
        return wp # Return the warping path for potential use

    except Exception as e:
        print(f"  Error during alignment of {target_audio_path.name} and {reference_audio_path.name}: {e}")
        traceback.print_exc()
        return None


# ==============================================================================
# === MAIN SCRIPT LOGIC ===
# ==============================================================================
def main():
    print("--- Starting Cornfield Chase Pilot Script (v0.1) ---")
    ensure_dir_exists(BASE_OUTPUT_DIR)

    bluray_start_s = frames_to_seconds(BLURAY_SCENE_START_FRAME, BLURAY_SCENE_FPS)
    bluray_end_s = frames_to_seconds(BLURAY_SCENE_END_FRAME, BLURAY_SCENE_FPS)
    bluray_duration_s = bluray_end_s - bluray_start_s
    print(f"Blu-ray segment: Start={bluray_start_s:.3f}s, End={bluray_end_s:.3f}s, Duration={bluray_duration_s:.3f}s")

    if bluray_duration_s <= 0: print("Error: Blu-ray segment duration zero/negative."); return

    bluray_mono_channels_basedir = BASE_OUTPUT_DIR / "01_bluray_mono_channels"
    ensure_dir_exists(bluray_mono_channels_basedir)
    all_languages_channel_files = {} 

    for lang_code, mkv_track_idx in BLURAY_LANGUAGES_TO_PROCESS.items():
        lang_output_dir = bluray_mono_channels_basedir / lang_code
        channel_paths = extract_segment_and_split_channels(
            mkv_path=BLURAY_MKV_PATH,
            output_basedir_for_lang=lang_output_dir,
            mkv_track_specifier=mkv_track_idx,
            start_seconds=bluray_start_s,
            duration_seconds=bluray_duration_s,
            passed_channel_layout_str=CHANNEL_LAYOUT_STRING, # Pass global config
            passed_channel_names_ordered=CHANNEL_NAMES_5_1,  # Pass global config
            passed_ffmpeg_channel_pads=CHANNELSPLIT_PAD_NAMES_5_1 # Pass global config
        )
        if channel_paths: all_languages_channel_files[lang_code] = channel_paths
        else: print(f"Skipping language {lang_code} due to error.")

    if len(all_languages_channel_files) < 2: print("Error: <2 langs processed. Cannot median filter."); return

    median_filtered_dir = BASE_OUTPUT_DIR / "02_median_filtered_channels"
    ensure_dir_exists(median_filtered_dir)
    processed_channel_role_paths = {}

    for ch_role_name in CHANNEL_NAMES_5_1:
        print(f"\n--- Processing channel role: {ch_role_name} ---")
        files_for_this_role = []
        for lang_code in BLURAY_LANGUAGES_TO_PROCESS.keys():
            if lang_code in all_languages_channel_files and ch_role_name in all_languages_channel_files[lang_code]:
                files_for_this_role.append(all_languages_channel_files[lang_code][ch_role_name])
        if len(files_for_this_role) < 2: print(f"  Not enough versions for '{ch_role_name}'. Skipping."); continue
        
        output_path_for_role = median_filtered_dir / f"processed_cornfield_{ch_role_name}.wav"
        success = process_channel_role_across_languages(
            list_of_mono_paths_for_role=files_for_this_role, output_processed_path=output_path_for_role,
            sample_rate=TARGET_SAMPLE_RATE, alignment_segment_s=ALIGNMENT_SEGMENT_DURATION_S,
            n_fft=STFT_N_FFT, hop_length=STFT_HOP_LENGTH)
        if success: processed_channel_role_paths[ch_role_name] = output_path_for_role

    print("\n--- Median filtering stage complete ---")
    if not processed_channel_role_paths: print("No channels successfully median filtered."); return
    for role, path in processed_channel_role_paths.items(): print(f"  Processed {role}: {path}")

    print(f"\n--- Preparing FYC Track: {FYC_AUDIO_PATH.name if FYC_AUDIO_PATH else 'N/A'} ---")
    fyc_prepared_dir = BASE_OUTPUT_DIR / "03_fyc_prepared"
    ensure_dir_exists(fyc_prepared_dir)
    if not FYC_AUDIO_PATH or not FYC_AUDIO_PATH.exists(): # Check if path is set and exists
        print(f"FYC audio file not found or path not set ({FYC_AUDIO_PATH}). Skipping FYC steps.")
    else:
        try:
            fyc_audio_stereo, fyc_sr = librosa.load(FYC_AUDIO_PATH, sr=TARGET_SAMPLE_RATE, mono=False)
            if fyc_audio_stereo.ndim == 1: fyc_L, fyc_R = fyc_audio_stereo, np.copy(fyc_audio_stereo)
            elif fyc_audio_stereo.shape[0] == 2: fyc_L, fyc_R = fyc_audio_stereo[0, :], fyc_audio_stereo[1, :]
            elif fyc_audio_stereo.shape[0] > 2: print("  Warning: FYC >2ch. Using first two."); fyc_L, fyc_R = fyc_audio_stereo[0, :], fyc_audio_stereo[1, :]
            else: raise ValueError("FYC audio shape unexpected.")
            sf.write(fyc_prepared_dir / f"{FYC_AUDIO_PATH.stem}_L.wav", fyc_L, TARGET_SAMPLE_RATE)
            sf.write(fyc_prepared_dir / f"{FYC_AUDIO_PATH.stem}_R.wav", fyc_R, TARGET_SAMPLE_RATE)
            print(f"  Saved prepared FYC L/R channels to: {fyc_prepared_dir}")
        except Exception as e: print(f"  Error processing FYC track: {e}"); traceback.print_exc()

    # Inside main(), after FYC preparation
    aligned_outputs_dir = BASE_OUTPUT_DIR / "04_aligned_outputs"
    ensure_dir_exists(aligned_outputs_dir)


    # --- 5. Align Processed Blu-ray Channels with FYC Channels ---
    print(f"\n--- STAGE 5: Alignment with FYC ---")

    # Define paths to the files we want to align
    # (Make sure these filenames match what your script actually creates)
    processed_bluray_fl_path = median_filtered_dir / "processed_cornfield_FL.wav"
    fyc_l_path = fyc_prepared_dir / f"{FYC_AUDIO_PATH.stem}_L.wav" # Use the stem of the original FYC file

    if processed_bluray_fl_path.exists() and fyc_l_path.exists():
        output_warped_fl_path = aligned_outputs_dir / f"warped_processed_cornfield_FL_to_FYCL.wav"
        
        # Store the warping path for FL. We might apply the same to FR.
        warping_path_fl = align_and_warp_audio(
            target_audio_path=processed_bluray_fl_path,
            reference_audio_path=fyc_l_path,
            output_warped_audio_path=output_warped_fl_path,
            sample_rate=TARGET_SAMPLE_RATE,
            hop_length_chroma=STFT_HOP_LENGTH # Can reuse STFT hop_length for chroma
        )

        # You would typically do the same for FR vs FYC_R
        processed_bluray_fr_path = median_filtered_dir / "processed_cornfield_FR.wav"
        fyc_r_path = fyc_prepared_dir / f"{FYC_AUDIO_PATH.stem}_R.wav"
        if processed_bluray_fr_path.exists() and fyc_r_path.exists():
            output_warped_fr_path = aligned_outputs_dir / f"warped_processed_cornfield_FR_to_FYCR.wav"
            warping_path_fr = align_and_warp_audio(
                target_audio_path=processed_bluray_fr_path,
                reference_audio_path=fyc_r_path,
                output_warped_audio_path=output_warped_fr_path,
                sample_rate=TARGET_SAMPLE_RATE,
                hop_length_chroma=STFT_HOP_LENGTH
            )
        else:
            print("Skipping FR alignment as one or both files are missing.")
            
        # Save the original FYC files to the alignment directory for easy comparison
        if fyc_l_path.exists(): shutil.copy(fyc_l_path, aligned_outputs_dir / fyc_l_path.name)
        if fyc_r_path.exists(): shutil.copy(fyc_r_path, aligned_outputs_dir / fyc_r_path.name)

    else:
        print("Cannot perform alignment: Processed Blu-ray FL or FYC L file is missing.")
        
    # ... (rest of main, with STAGE 6 and 7 still as placeholders)
    print("\n--- STAGE 6: SFX Reduction using FYC (Not Yet Implemented) ---")
    print("\n--- STAGE 7: Final Merge (Not Yet Implemented) ---")
    print("\n--- Cornfield Chase Pilot Script Finished ---")

if __name__ == "__main__":
    ensure_dir_exists(BASE_OUTPUT_DIR)
    main()