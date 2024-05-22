rm -f test/results.csv
touch test/results.csv
echo ";input;noise;input_snr;output_snr;input_mse;output_mse;execution_time" > test/results.csv

python3 main.py "speech-librivox-0005" "noise-free-sound-0001"
python3 main.py "speech-librivox-0005" "noise-free-sound-0165"
python3 main.py "speech-librivox-0005" "noise-free-sound-0271"
python3 main.py "speech-librivox-0005" "noise-free-sound-0278"
python3 main.py "speech-librivox-0005" "white_noise_5db"
python3 main.py "speech-librivox-0005" "white_noise_10db"

python3 main.py "speech-librivox-0010" "noise-free-sound-0001"
python3 main.py "speech-librivox-0010" "noise-free-sound-0165"
python3 main.py "speech-librivox-0010" "noise-free-sound-0271"
python3 main.py "speech-librivox-0010" "noise-free-sound-0278"
python3 main.py "speech-librivox-0010" "white_noise_5db"
python3 main.py "speech-librivox-0010" "white_noise_10db"

python3 main.py "speech-librivox-0034" "noise-free-sound-0001"
python3 main.py "speech-librivox-0034" "noise-free-sound-0165"
python3 main.py "speech-librivox-0034" "noise-free-sound-0271"
python3 main.py "speech-librivox-0034" "noise-free-sound-0278"
python3 main.py "speech-librivox-0034" "white_noise_5db"
python3 main.py "speech-librivox-0034" "white_noise_10db"

python3 main.py "speech-librivox-0062" "noise-free-sound-0001"
python3 main.py "speech-librivox-0062" "noise-free-sound-0165"
python3 main.py "speech-librivox-0062" "noise-free-sound-0271"
python3 main.py "speech-librivox-0062" "noise-free-sound-0278"
python3 main.py "speech-librivox-0062" "white_noise_5db"
python3 main.py "speech-librivox-0062" "white_noise_10db"

python3 main.py "speech-librivox-0082" "noise-free-sound-0001"
python3 main.py "speech-librivox-0082" "noise-free-sound-0165"
python3 main.py "speech-librivox-0082" "noise-free-sound-0271"
python3 main.py "speech-librivox-0082" "noise-free-sound-0278"
python3 main.py "speech-librivox-0082" "white_noise_5db"
python3 main.py "speech-librivox-0082" "white_noise_10db"

python3 main.py "speech-librivox-0117" "noise-free-sound-0001"
python3 main.py "speech-librivox-0117" "noise-free-sound-0165"
python3 main.py "speech-librivox-0117" "noise-free-sound-0271"
python3 main.py "speech-librivox-0117" "noise-free-sound-0278"
python3 main.py "speech-librivox-0117" "white_noise_5db"
python3 main.py "speech-librivox-0117" "white_noise_10db"