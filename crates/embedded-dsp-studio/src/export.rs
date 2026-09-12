//! Zero-dependency WAV (RIFF 16-bit PCM) & CSV exporter/importer.
//!
//! Allows exporting Signal Lab waveforms or Snapshot forensic impulse responses
//! to canonical .wav and .csv files, and importing test waveforms back into the studio.

/// Encodes an array of f32 samples into a canonical 44-byte RIFF/WAVE header and 16-bit PCM data.
pub fn export_wav_16bit(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let num_samples = samples.len();
    let subchunk2_size = (num_samples * 2) as u32; // 16-bit mono = 2 bytes per sample
    let chunk_size = 36 + subchunk2_size;
    let byte_rate = sample_rate * 2;
    let block_align = 2u16;
    let bits_per_sample = 16u16;

    let mut out = Vec::with_capacity(44 + subchunk2_size as usize);

    // RIFF chunk descriptor
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&chunk_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");

    // fmt subchunk
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // Subchunk1Size (16 for PCM)
    out.extend_from_slice(&1u16.to_le_bytes()); // AudioFormat (1 = PCM)
    out.extend_from_slice(&1u16.to_le_bytes()); // NumChannels (1 = Mono)
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());

    // data subchunk
    out.extend_from_slice(b"data");
    out.extend_from_slice(&subchunk2_size.to_le_bytes());

    // PCM samples
    for &sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let s16 = (clamped * 32767.0).round() as i16;
        out.extend_from_slice(&s16.to_le_bytes());
    }

    out
}

/// Parses a 16-bit PCM WAV file and returns (samples, sample_rate).
pub fn parse_wav_16bit(data: &[u8]) -> Result<(Vec<f32>, u32), &'static str> {
    if data.len() < 44 {
        return Err("File too short for WAV header");
    }
    if &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err("Not a valid RIFF/WAVE file");
    }

    let mut pos = 12;
    let mut sample_rate = 48000;
    let mut num_channels = 1u16;
    let mut bits_per_sample = 16u16;
    let mut audio_data_opt = None;

    while pos + 8 <= data.len() {
        let subchunk_id = &data[pos..pos + 4];
        let subchunk_size = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap()) as usize;
        pos += 8;

        if subchunk_id == b"fmt " {
            if subchunk_size < 16 || pos + 16 > data.len() {
                return Err("Invalid fmt chunk");
            }
            let format_tag = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap());
            if format_tag != 1 {
                return Err("Only uncompressed PCM WAV is supported");
            }
            num_channels = u16::from_le_bytes(data[pos + 2..pos + 4].try_into().unwrap());
            sample_rate = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap());
            bits_per_sample = u16::from_le_bytes(data[pos + 14..pos + 16].try_into().unwrap());
        } else if subchunk_id == b"data" {
            let end = (pos + subchunk_size).min(data.len());
            audio_data_opt = Some(&data[pos..end]);
            break;
        }

        pos += subchunk_size;
    }

    let audio_bytes = audio_data_opt.ok_or("Missing data chunk in WAV file")?;
    if bits_per_sample != 16 {
        return Err("Only 16-bit PCM WAV is currently supported");
    }

    let bytes_per_sample = (bits_per_sample / 8) as usize;
    let step = bytes_per_sample * (num_channels as usize);
    let mut samples = Vec::with_capacity(audio_bytes.len() / step);

    for chunk in audio_bytes.chunks_exact(step) {
        let val = i16::from_le_bytes(chunk[0..2].try_into().unwrap());
        samples.push(val as f32 / 32767.0);
    }

    Ok((samples, sample_rate))
}

/// Encodes samples to CSV: `sample_idx,time_secs,amplitude\n`
pub fn export_csv(samples: &[f32], sample_rate: f32) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(samples.len() * 24);
    out.push_str("sample_idx,time_secs,amplitude\n");
    for (i, &s) in samples.iter().enumerate() {
        let t = (i as f32) / sample_rate;
        let _ = writeln!(out, "{},{:.6},{:.6}", i, t, s);
    }
    out
}

/// Parses CSV amplitude values (handles headers if present).
pub fn parse_csv(csv_text: &str) -> Result<Vec<f32>, &'static str> {
    let mut samples = Vec::new();
    for line in csv_text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = trimmed.split(',').collect();
        let val_str = parts.last().unwrap().trim();
        if let Ok(val) = val_str.parse::<f32>() {
            samples.push(val);
        }
    }
    if samples.is_empty() {
        Err("No valid numeric samples found in CSV")
    } else {
        Ok(samples)
    }
}

/// Saves or downloads file depending on platform (native vs web).
pub fn save_or_download_file(filename: &str, #[allow(unused_variables)] mime: &str, data: &[u8]) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let ext = filename.split('.').next_back().unwrap_or("*");
        if let Some(path) = rfd::FileDialog::new()
            .set_file_name(filename)
            .add_filter(ext, &[ext])
            .save_file()
        {
            let _ = std::fs::write(path, data);
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        use eframe::wasm_bindgen::JsCast;
        if let Some(window) = web_sys::window() {
            if let Some(document) = window.document() {
                let uint8_array = js_sys::Uint8Array::from(data);
                let array = js_sys::Array::new();
                array.push(&uint8_array);
                let bag = web_sys::BlobPropertyBag::new();
                bag.set_type(mime);
                if let Ok(blob) =
                    web_sys::Blob::new_with_u8_array_sequence_and_options(&array, &bag)
                {
                    if let Ok(url) = web_sys::Url::create_object_url_with_blob(&blob) {
                        if let Ok(a) = document.create_element("a") {
                            let _ = a.set_attribute("href", &url);
                            let _ = a.set_attribute("download", filename);
                            if let Some(html_elem) = a.dyn_into::<web_sys::HtmlElement>().ok() {
                                html_elem.click();
                            }
                        }
                        let _ = web_sys::Url::revoke_object_url(&url);
                    }
                }
            }
        }
    }
}
