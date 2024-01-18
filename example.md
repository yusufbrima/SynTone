python create.py --sample_rate 16000 --freq_start 400 --freq_end 8000 --num_freqs 100 --amp_start 0.1 --amp_end 1.0 --num_amps 10 --file_path "/net/projects/scratch/summer/valid_until_31_January_2024/ybrima/data/learning/SynTone/my_dataset.npz"

python trainer.py --file_path "/net/projects/scratch/summer/valid_until_31_January_2024/ybrima/data/learning/SynTone/my_dataset.npz" --batch_size 16 --epochs 1 --latent_dim 8

scp ybrima@beam.cv.uos.de:/net/store/cv/users/ybrima/RTGCompCog/SyncSpeech/environment.yml ./environment.yml


conda env export -n eval -f environment.yml --no-builds
conda env export --file environment.yml --no-builds --from-history

scp -r /Users/yusuf/Desktop/SynTone ybrima@beam.cv.uos.de:/net/store/cv/users/ybrima/RTGCompCog/SynTone

pip install Cython==0.29.36
pip install scikit-learn==0.23 --no-build-isolation

python your_script.py --file_path "/path/to/dataset.npz" --batch_size 128 --num_experiments 10 --latent_dim 10 --model_list "model1.pth" "model2.pth" "model3.pth" "model4.pth"
