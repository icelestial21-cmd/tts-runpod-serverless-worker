import argparse
import numpy as np
from scipy.io.wavfile import write
from rusynth import RUSynth

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help='Path to the model directory')
    parser.add_argument('--repo-id', type=str, default='bes-dev/rusynth', help='Repository ID')
    parser.add_argument('--input', type=str, required=True, help='Text or Text File to synthesize')
    parser.add_argument('--input-path', action='store_true', help='Input is a path to a text file')
    parser.add_argument('--output', type=str, default='output.wav', help='Output audio file')
    parser.add_argument('--speaker-id', type=int, default=0, help='Speaker ID')
    parser.add_argument('--speed', type=float, default=1.0, help='Speed factor')
    parser.add_argument('--accentize', action='store_true', help='Accentize text')
    parser.add_argument('--volume', type=float, default=0.3, help='Volume')
    parser.add_argument('--low-pass-filter-cutoff', type=int, help='Low pass filter cutoff value')
    parser.add_argument('--verbose', action='store_true', help='Verbose mode')
    args = parser.parse_args()

    # create synthesizer
    synthesizer = RUSynth.from_pretrained(args.repo_id, args.model_path)
    # read text
    text = args.input
    if args.input_path:
        with open(args.input, 'r', encoding='utf-8') as f:
            text = f.read()
    # synthesize
    audio, sr = synthesizer(
        text = text,
        speaker_id = args.speaker_id,
        speed = args.speed,
        accentize = args.accentize,
        volume = args.volume,
        low_pass_filter_cutoff = args.low_pass_filter_cutoff,
        long_text = args.input_path,
        verbose = args.verbose
    )
    # write audio
    write(args.output, sr, audio.astype(np.int16))
    print(f'Output file is ready: {args.output}')

if __name__ == '__main__':
    main()
