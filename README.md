My attempt at a neural network that denoises audio ( specifically from a radio ),

Use the uv package manger
Use 'uv run src/prepare.py' to download and perapre the dataset for training
Use 'uv run src/training.py' to start the training
Edit 'src/config.py' to configurate

TODO:

- Use a better dataset ( maybe split training and eval )
- Cache data generation for the dataset, beacuse some filters are expensive on the cpu ( maybe pre-procces even? )
- Implement real noise instead of synthetic noise by mixing in recorded hf noise / Make data synthesis randmoness pre determenistic for better training eval
