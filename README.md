TODO:

- Make data synthesis randmoness pre determenistic for better training eval
- Use a better dataset ( maybe split training and eval )
- Implement better saving method during training
- Delete old data before preparing new data to prevent contamanation
- Cache data generation for the dataset, beacuse soem filters are expensive on the cpu ( maybe pre-procces even? )
- Implement real noise instead of synthetic noise by mixing in recorded hf noise
- Tweak config values
- 
